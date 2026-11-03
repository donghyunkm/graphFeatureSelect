"""Primary model: a GNN backbone (GATv2/SAGE/GCN) plus a learnable feature-selection layer
(``fs_method``: ``persist`` Gumbel-softmax gene mask / ``scGist`` / ``none``), wrapped in the
``LitGnnFs`` Lightning module. This is the model used across the main experiments; the ``antelope_*``
siblings differ only in which feature-selection method they swap in. ``trainmode`` (0/1/2) selects which
patch nodes contribute to the training loss."""

import csv
import os

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision.ops import sigmoid_focal_loss

from gfs.models.transforms import HalfHop


class MLP(nn.Module):
    """
    Multilayer perceptron (MLP) model.

    Args:
      input_size: number of inputs.
      output_size: number of outputs.
      hidden: list of hidden layer widths.
      activation: nonlinearity between layers.
    """

    def __init__(self, input_size, output_size, hidden, activation=nn.ReLU()):
        super().__init__()

        # Fully connected layers.
        self.input_size = input_size
        self.output_size = output_size
        fc_layers = [nn.Linear(d_in, d_out) for d_in, d_out in zip([input_size] + hidden, hidden + [output_size])]
        self.fc = nn.ModuleList(fc_layers)

        # Activation function.
        self.activation = activation

    def forward(self, x):
        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.activation(x)

        return self.fc[-1](x)


class FeatureRegularizer(nn.Module):
    def __init__(
        self, l1=0.1, panel_size=None, priority_score=None, pairs=None, alpha=0.5, beta=0.5, gamma=0.5, strict=True
    ):
        super().__init__()
        self.l1 = 0.01 if l1 is None else l1
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.n_features = panel_size if panel_size else None
        if pairs is not None:
            self.pairs = pairs
        else:
            self.pairs = None

        self.strict = strict

    def forward(self, x):
        abs_x = torch.abs(x)
        reg = torch.tensor(0.0, dtype=x.dtype, device=x.device)

        # Force weights toward 0 or 1
        reg += torch.sum(abs_x * torch.abs(x - 1))

        # Panel size constraint
        if self.n_features is not None:
            if self.strict:
                reg += torch.abs(torch.sum(abs_x) - self.n_features) * self.alpha
            else:
                reg += torch.max(torch.sum(abs_x) - self.n_features, 0) * self.alpha

        # Pairwise selection
        if self.pairs is not None:
            # Similar to tf.tensordot(abs_x, pairs, axes=1)
            pair_reg = torch.matmul(abs_x, self.pairs)
            reg += torch.sum(torch.abs(pair_reg)) * self.gamma

        return self.l1 * reg


class GnnFs(torch.nn.Module):
    """
    This model features:
    1. a GNN backbone adapted from `Classic GNNs are strong baselines...` (Luo et al. 2024)
    2. a feature selection layer based on concrete variables (Jang. et al. 2016)

    Args:
        gene_ch (int): number of input features
        spatial_ch (int): number of spatial features
        hid_ch (int): number of hidden features
        out_ch (int): number of output features
        n_select (int): number of features to select
        local_layers (int): gnn depth
        dropout (float): dropout rate
        heads (int): number of heads
        pre_linear (bool): if True, use pre-linear layer
        res (bool): if True, use residual connections
        ln (bool): if True, use layer normalization
        bn (bool): if True, use batch normalization
        jk (bool): if True, use skip connections
        x_res (bool): if True, use x residual connections
        gnn (str): "gat", "sage", or "gcn"
        xyz_status (bool): if True, use xyz status
    """

    def __init__(
        self,
        gene_ch,
        spatial_ch,
        hid_ch,
        out_ch,
        n_select,
        local_layers,
        dropout,
        heads,
        pre_linear,
        res,
        ln,
        bn,
        jk,
        x_res,
        gnn,
        xyz_status,
        fs_method,
        focal_loss,
        hard_training,
    ):
        super(GnnFs, self).__init__()
        self.fs_method = fs_method
        self.gnn_type = gnn

        if self.fs_method == "persist":
            self.logits = nn.Parameter(torch.randn(n_select, gene_ch))
        elif self.fs_method == "scGist":
            self.logits = nn.Parameter(torch.full((1, gene_ch), 0.5))
        elif self.fs_method == "none":
            # no selection module: inputs are an already-frozen gene panel (panel-baseline eval).
            # no `logits` parameter is created, so nothing selection-related can train.
            pass
        self.dropout = dropout

        self.pre_linear = pre_linear
        self.res = res
        self.ln = ln
        self.bn = bn
        self.jk = jk
        self.x_res = x_res
        self.xyz_status = xyz_status
        self.gnn_layers = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.hard_training = hard_training
        self.lin_in = torch.nn.Linear(gene_ch, hid_ch)
        self.focal_loss = focal_loss
        if self.fs_method == "scGist":
            self.feature_regularizer = FeatureRegularizer(l1=0.1, panel_size=n_select)

        if not self.pre_linear:
            if gnn == "gat":
                self.gnn_layers.append(
                    GATv2Conv(gene_ch, hid_ch, heads=heads, concat=False, add_self_loops=False, bias=False)
                )
            elif gnn == "sage":
                self.gnn_layers.append(SAGEConv(gene_ch, hid_ch))
            else:
                self.gnn_layers.append(GCNConv(gene_ch, hid_ch, cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(gene_ch, hid_ch))
            self.lns.append(torch.nn.LayerNorm(hid_ch))
            self.bns.append(torch.nn.BatchNorm1d(hid_ch))
            local_layers = local_layers - 1

        for _ in range(local_layers):
            if gnn == "gat":
                self.gnn_layers.append(
                    GATv2Conv(hid_ch, hid_ch, heads=heads, concat=False, add_self_loops=False, bias=False)
                )
            elif gnn == "sage":
                self.gnn_layers.append(SAGEConv(hid_ch, hid_ch))
            else:
                self.gnn_layers.append(GCNConv(hid_ch, hid_ch, cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(hid_ch, hid_ch))
            self.lns.append(torch.nn.LayerNorm(hid_ch))
            self.bns.append(torch.nn.BatchNorm1d(hid_ch))

        self.pred_local = torch.nn.Linear(hid_ch, out_ch)

        if self.x_res:
            # self.res_lin = torch.nn.Linear(gene_ch, out_ch) old
            self.res_lin = MLP(gene_ch, out_ch, [128, 128], nn.ReLU())  # new (from Persist)
        if self.xyz_status:
            self.xyz_lin = torch.nn.Linear(spatial_ch, out_ch)

    def reset_parameters(self):
        for layer in self.gnn_layers:
            layer.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()
        self.res_lin.reset_parameters()
        self.xyz_lin.reset_parameters()
        self.logits.reset_parameters()

    def get_mask(self, tau, subgraph_id):
        """
        Get soft k-hot mask
        """
        if self.training and not self.hard_training:
            # print("soft")
            # sample soft k-hot vectors for each subgraph in the batch during training
            # extra samples are harmless, makes indexing more straightforward;
            # case: e.g. subgraph_id.unique() = [0, 2, 3, 4]
            n_subgraphs = subgraph_id.max() + 1
            # sample has dims (n_selections, n_subgraphs, n_features) --> chooses 1 gene for each of n_selections
            sample = F.gumbel_softmax(self.logits.unsqueeze(1).repeat(1, n_subgraphs, 1), tau=tau, hard=False, dim=-1)
            # print("SAMPLE ", sample.shape)
            # k_hot has dims (n_subgraphs, n_features) --> uses max operation to combine/sum the n_selections into 1 dimension
            k_hot = torch.max(sample, dim=0).values
            # print("KHOT ", k_hot.shape)
            # repeat k-hot masks for each node based on their subgraph membership
            return k_hot[subgraph_id]
        else:
            # print("hard")
            # return hard k-hot mask for evaluation
            k_hot_ind = torch.argmax(self.logits, dim=1)
            k_hot = torch.zeros(1, self.logits.size(1), device=self.logits.device)
            k_hot.scatter_(1, k_hot_ind.unsqueeze(0), 1)
            # k_hot has dims (1, n_features)
            return k_hot

    def get_mask_indices(self):
        """
        Get indices of highest probability features.
        """
        probs = F.softmax(self.logits, dim=1)
        mask_indices = torch.argmax(probs, dim=1)
        mask_probs = torch.gather(probs, dim=1, index=mask_indices.unsqueeze(1))
        return mask_indices, mask_probs.squeeze()

    def forward(self, x, edge_index, xyz, subgraph_id, tau, hard_, return_attention_weights=False):
        # When return_attention_weights=True (and gnn=="gat"), also return the post-softmax GATv2
        # attention coefficients from each GAT layer as a list of (att_edge_index, alpha) tuples.
        # Default False keeps the ordinary forward API/behaviour used at inference and by non-GAT gnns.

        # per-subgraph mean-centring of xyz. Vectorized; equivalent to the per-id python loop it
        # replaces, including its quirk of using a single scalar mean over all 3 coords (the
        # original called torch.mean with no dim) rather than a per-axis centroid.
        # The loop cost O(n_subgraphs) masked ops, which is fine for 64 patches but ~630 ms/step
        # on GPU when every node is its own subgraph (panel-baseline eval, 4147 patches).
        n_sub = int(subgraph_id.max()) + 1
        ones = torch.ones_like(subgraph_id, dtype=xyz.dtype)
        sums = torch.zeros(n_sub, dtype=xyz.dtype, device=xyz.device).index_add_(0, subgraph_id, xyz.sum(dim=1))
        counts = torch.zeros(n_sub, dtype=xyz.dtype, device=xyz.device).index_add_(0, subgraph_id, ones)
        # clamp guards subgraph ids absent from this batch (never indexed below, but avoids 0/0)
        means = sums / (counts.clamp(min=1) * xyz.shape[1])
        xyz = xyz - means[subgraph_id].unsqueeze(1)
        # print("x shape ", x.shape) x shape  torch.Size([10602, 500])
        # print("xyz shape ", xyz.shape)
        # print("mask ", mask.shape)

        if self.fs_method == "persist":
            mask = self.get_mask(tau, subgraph_id)
            # print("256mask ", mask)
            # print("254persist ", mask.shape, x.shape) #254persist  torch.Size([16418, 500]) torch.Size([16418, 500])
            x = mask * x
        if self.fs_method == "scGist":
            x = x * self.logits

        if self.x_res:
            x_to_add = self.res_lin(x)
        if self.xyz_status:
            xyz = self.xyz_lin(xyz)

        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_final = 0

        collect_attention = return_attention_weights and self.gnn_type == "gat"
        attention_records = []

        for i, layer in enumerate(self.gnn_layers):
            x_in = x
            # Only request attention weights when explicitly asked (and only for GAT layers), so the
            # standard/zero-weight path stays byte-for-byte the current implementation.
            if collect_attention:
                conv_out, (att_edge_index, alpha) = layer(x_in, edge_index, return_attention_weights=True)
                # store per-batch only; not persistent module state (no attention kept across batches).
                attention_records.append((att_edge_index, alpha))
            else:
                conv_out = layer(x_in, edge_index)
            if self.res:
                # residual projection uses x_in (the layer input), matching the original semantics.
                x = conv_out + self.lins[i](x_in)
            else:
                x = conv_out
            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_final = x_final + x
            else:
                x_final = x

        x = self.pred_local(x_final)

        if self.x_res:
            x = x + x_to_add

        # TODO: why add here?
        if self.xyz_status:
            x = x + xyz

        if return_attention_weights:
            return x, attention_records
        return x


class LitGnnFs(L.LightningModule):
    def __init__(self, config):
        super(LitGnnFs, self).__init__()

        self.save_hyperparameters(config)

        # model
        cfg = self.hparams.model

        # GR2ST-style GATv2 attention-entropy regularization (auxiliary training loss).
        # Owned by the LightningModule; strength controlled by the config coefficient. Unrelated to
        # `tau` (the Gumbel-softmax feature-selection temperature).
        # `.get(...)` keeps configs that predate these keys backward-compatible (default: disabled).
        self.attention_entropy_weight = float(cfg.get("attention_entropy_weight", 0.0))
        self.attention_entropy_eps = float(cfg.get("attention_entropy_eps", 1.0e-9))
        # `log_attention_stats` decouples MEASURING attention stats from APPLYING the entropy loss: set it
        # true (with weight=0) on a baseline run to log the same entropy/neighbourhood diagnostics without
        # regularizing, so enabled-vs-disabled models are directly comparable. Requesting attention weights
        # is numerically free (bit-identical logits), so this does not change model behaviour, only logging.
        self.log_attention_stats = bool(cfg.get("log_attention_stats", False))
        self.gnn_type = cfg.gnn
        if (self.attention_entropy_weight > 0 or self.log_attention_stats) and self.gnn_type != "gat":
            raise ValueError("attention_entropy_weight > 0 / log_attention_stats=true requires gnn='gat'")

        self.model = GnnFs(
            gene_ch=cfg.gene_ch,
            spatial_ch=cfg.spatial_ch,
            hid_ch=cfg.hid_ch,
            out_ch=cfg.out_ch,
            n_select=cfg.n_select,
            local_layers=cfg.local_layers,
            dropout=cfg.dropout,
            heads=cfg.heads,
            pre_linear=cfg.pre_linear,
            res=cfg.res,
            ln=cfg.ln,
            bn=cfg.bn,
            jk=cfg.jk,
            x_res=cfg.x_res,
            gnn=cfg.gnn,
            xyz_status=cfg.xyz_status,
            fs_method=cfg.fs_method,
            focal_loss=cfg.focal_loss,
            hard_training=cfg.hard_training,
        )

        # related to training step
        self.halfhop = cfg.halfhop
        if self.halfhop:
            self.transform = HalfHop(alpha=0.5, p=0.5)
        else:
            self.transform = lambda x: x

        self.tautype = cfg.tautype
        self.trainmode = cfg.trainmode  # 0: all patch nodes; 1: root only; 2: nodes within loss_hops of a seed (loss_mask, needs hemisphere_new_4hop)

        # losses
        self.loss_ce = nn.CrossEntropyLoss()

        # metrics
        # options = {"num_classes": cfg.out_ch, "top_k": 1, "multidim_average": "global"}
        options = {"num_classes": cfg.out_ch, "top_k": 1, "multidim_average": "global"}

        self.train_overall_acc = MulticlassAccuracy(average="weighted", **options)
        self.val_overall_acc = MulticlassAccuracy(average="weighted", **options)
        self.val_f1_overall = MulticlassF1Score(average="weighted", **options)

        self.test_overall_acc = MulticlassAccuracy(average="weighted", **options)
        self.test_macro_acc = MulticlassAccuracy(average="macro", **options)
        self.test_micro_acc = MulticlassAccuracy(average="micro", **options)
        self.test_f1_overall = MulticlassF1Score(average="weighted", **options)
        self.test_f1_macro = MulticlassF1Score(average="macro", **options)
        self.test_f1_micro = MulticlassF1Score(average="micro", **options)

        self.test_pred = []
        self.test_labels = []

        self.train_macro_acc = MulticlassAccuracy(average="macro", **options)
        self.val_macro_acc = MulticlassAccuracy(average="macro", **options)

    def forward(self, gene_exp, edge_index, xyz, subgraph_id, tau, hard_, return_attention_weights=False):
        """
        Args:
            gene_exp (torch.Tensor): (n_nodes, n_genes)
            edge_index (torch.Tensor): (2, n_edges)
            xyz (torch.Tensor): (n_nodes, 3)
            subgraph_id (torch.Tensor): (n_nodes)
            tau (float): temperature for Gumbel-Softmax
            hard_ (bool): if True, use hard Gumbel-Softmax
            return_attention_weights (bool): if True (and gnn=="gat"), also return the per-GAT-layer
                attention records [(att_edge_index, alpha), ...]. Default False preserves the ordinary
                inference API (returns only the celltype logits).
        """

        return self.model(
            gene_exp, edge_index, xyz, subgraph_id, tau, hard_, return_attention_weights=return_attention_weights
        )

    def _gr2st_attention_entropy(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Literal GR2ST attention-entropy loss (Zhang et al., GR2ST):

            L = -mean_e [ sum_h alpha_{e,h} * log(alpha_{e,h} + eps) ]

        `alpha` is the post-softmax GATv2 attention tensor of shape [E, H] (E edges, H heads); the sum
        over dim=1 sums per-head entropy contributions, then we mean over edges. This is deliberately
        NOT the neighbourhood-normalized entropy (that is only a diagnostic below) — it reproduces the
        released GR2ST implementation so the auxiliary loss stays faithful. Multi-head `alpha` is passed
        in whole (heads are NOT averaged) so each head can develop its own selectivity. Do not detach.
        """
        return -torch.mean(torch.sum(alpha * torch.log(alpha + self.attention_entropy_eps), dim=1))

    def _attention_sharpness_diagnostics(self, attention_records):
        """
        Interpretable, per-destination-node/per-head attention-sharpness diagnostics (detached, no grad).

        For each returned GAT layer, group edges by destination node (see below for which edge_index row
        is the destination under the installed PyG version) and, per (node, head):
          - top1 mass      = max_j alpha_{ij,h}                        (higher => sharper)
          - neighbourhood entropy H_{i,h} = -sum_j alpha_{ij,h} log(alpha_{ij,h}+eps)
          - effective neighbours = exp(H_{i,h})                        (lower  => sharper)
          - normalized entropy  = H_{i,h} / log(deg_i)  (deg_i > 1 only; ~0 concentrated, ~1 uniform)
        Values are averaged over valid (node, head) pairs, then over layers. Returns a dict of python
        floats (safe to log; retains no computation graph).
        """
        eps = self.attention_entropy_eps
        top1_list, eff_list, norm_list = [], [], []
        for att_edge_index, alpha in attention_records:
            alpha = alpha.detach()
            if alpha.dim() == 1:
                alpha = alpha.unsqueeze(-1)
            # PyG message passing (flow="source_to_target") aggregates messages at row 1 (target/dest).
            dst = att_edge_index[1]
            num_nodes = int(dst.max().item()) + 1 if dst.numel() > 0 else 0
            H = alpha.size(1)
            if num_nodes == 0:
                continue

            # per-(dst,head): top-1 mass via scatter-max
            top1 = torch.zeros(num_nodes, H, device=alpha.device, dtype=alpha.dtype)
            top1.scatter_reduce_(0, dst.unsqueeze(1).expand(-1, H), alpha, reduce="amax", include_self=False)

            # per-(dst,head): entropy H_{i,h} = -sum_j a log(a+eps)
            contrib = -(alpha * torch.log(alpha + eps))
            ent = torch.zeros(num_nodes, H, device=alpha.device, dtype=alpha.dtype)
            ent.index_add_(0, dst, contrib)

            # per-dst degree (edge count into that node)
            deg = torch.zeros(num_nodes, device=alpha.device, dtype=alpha.dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=alpha.dtype))
            valid = deg > 0  # destinations actually present in this batch

            eff = torch.exp(ent)
            top1_list.append(top1[valid].mean())
            eff_list.append(eff[valid].mean())

            # normalized entropy only defined for deg > 1 (log(1)=0)
            multi = deg > 1
            if multi.any():
                norm = ent[multi] / torch.log(deg[multi]).unsqueeze(1)
                norm_list.append(norm.mean())

        diagnostics = {}
        if top1_list:
            diagnostics["train_attention_top1_mass"] = torch.stack(top1_list).mean().item()
            diagnostics["train_attention_effective_neighbors"] = torch.stack(eff_list).mean().item()
        if norm_list:
            diagnostics["train_attention_normalized_entropy"] = torch.stack(norm_list).mean().item()
        return diagnostics

    def training_step(self, batch, batch_idx):
        # calculate losses and metrics for all training nodes in the batch.
        batch_size = torch.sum(batch.train_mask)
        # print("381 BEFORE HALFHOP ", batch.x.shape) # 381 BEFORE HALFHOP  torch.Size([4258, 503])

        data = self.transform(batch)

        if self.trainmode == 0:
            # backprop/metrics for all nodes in batch (including neighbors)
            idx = data.train_mask
        elif self.trainmode == 1:
            # backprop/metrics for only "root" nodes, not neighbors
            idx = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]
        elif self.trainmode == 2:
            # backprop/metrics for nodes within loss_hops of a seed (== GNN depth), i.e.
            # nodes with a COMPLETE receptive field under deep sampling. Requires a data
            # module that attaches batch.loss_mask (hemisphere_new_4hop).
            idx = torch.where(batch.loss_mask)[0]
        # print("389 ", data.x.shape, len(data.gene_exp_ind)) # torch.Size([18503, 503]) 500
        # Request GAT attention weights when the entropy regularizer is active OR when only logging attention
        # stats (log_attention_stats). Otherwise take the ordinary (cheaper) plain forward path. Requesting
        # attention weights is numerically free (bit-identical logits). `use_attention_entropy` controls the
        # LOSS term; `collect_attention` controls whether we gather attention to measure/log at all. NB:
        # entropy is computed on `data.edge_index` -- the exact (possibly HalfHop-perturbed) graph used to
        # produce this batch's predictions, not a separate pass.
        use_attention_entropy = self.gnn_type == "gat" and self.attention_entropy_weight > 0
        collect_attention = self.gnn_type == "gat" and (self.attention_entropy_weight > 0 or self.log_attention_stats)
        if collect_attention:
            celltype_pred, attention_records = self.forward(
                gene_exp=data.x[:, data.gene_exp_ind],
                edge_index=data.edge_index,
                xyz=data.x[:, data.xyz_ind],
                subgraph_id=data.subgraph_id,
                tau=tau_schedule(self.tautype, self.current_epoch, self.trainer.max_epochs),
                hard_=False,
                return_attention_weights=True,
            )
        else:
            celltype_pred = self.forward(
                gene_exp=data.x[:, data.gene_exp_ind],
                edge_index=data.edge_index,
                xyz=data.x[:, data.xyz_ind],
                subgraph_id=data.subgraph_id,
                tau=tau_schedule(self.tautype, self.current_epoch, self.trainer.max_epochs),
                hard_=False,
                return_attention_weights=False,
            )
            attention_records = None
        # conditionally remove "slow nodes" (from halfhop)
        if hasattr(data, "slow_node_mask"):
            celltype_pred = celltype_pred[~data.slow_node_mask]

        # ---- classification loss ----
        if self.model.focal_loss:
            train_loss_cls = sigmoid_focal_loss(
                inputs=celltype_pred[idx],
                targets=F.one_hot(data.celltype[idx], num_classes=self.hparams.model.out_ch).to(torch.float),
                # targets = data.celltype[idx],
                alpha=0.25,
                gamma=2.0,
                reduction="mean",
            )
        else:
            train_loss_cls = self.loss_ce(celltype_pred[idx], data.celltype[idx])

        # Total objective = classification + existing feature reg + weighted attention entropy.
        train_loss_total = train_loss_cls

        # ---- existing scGist feature regularization ----
        train_loss_reg = None
        if self.model.fs_method == "scGist":
            train_loss_reg = self.model.feature_regularizer(self.model.logits) * 100
            train_loss_total = train_loss_total + train_loss_reg

        # ---- GR2ST-style attention-entropy regularization ----
        # Computed over ALL GAT edges in the sampled forward pass (incl. HalfHop slow-node edges and, under
        # trainmode=1/2, context edges), even though the classification loss is restricted to a node subset:
        # those edges all shape the root-node representations, so this is the closest analogue to regularizing
        # the GAT itself. Aggregate across stacked GAT layers with the MEAN (not sum) so changing local_layers
        # does not rescale the effective attention_entropy_weight -- an intentional adaptation of GR2ST (which
        # sums the entropies of its two single-GAT branches) to Antelope's stacked-GAT backbone.
        train_attention_entropy = None
        train_loss_attention_entropy_weighted = None
        if collect_attention:
            if use_attention_entropy:
                # differentiable: this term is added to the objective
                layer_entropies = [self._gr2st_attention_entropy(alpha) for _, alpha in attention_records]
                train_attention_entropy = torch.stack(layer_entropies).mean()
                train_loss_attention_entropy_weighted = self.attention_entropy_weight * train_attention_entropy
                train_loss_total = train_loss_total + train_loss_attention_entropy_weighted
            else:
                # log-only (log_attention_stats with weight=0): measure entropy, do NOT add to the loss and
                # do NOT retain the attention subgraph.
                with torch.no_grad():
                    layer_entropies = [self._gr2st_attention_entropy(alpha) for _, alpha in attention_records]
                    train_attention_entropy = torch.stack(layer_entropies).mean()

        # calculate metrics

        self.train_overall_acc(preds=celltype_pred[idx], target=data.celltype[idx])
        self.train_macro_acc(preds=celltype_pred[idx], target=data.celltype[idx])

        # log losses and metrics
        options = {
            "on_step": self.hparams.logging.on_step,
            "on_epoch": self.hparams.logging.on_epoch,
            "prog_bar": self.hparams.logging.prog_bar,
            "logger": self.hparams.logging.logger,
            "batch_size": batch_size,
        }
        # `train_loss_ce` kept as a backward-compatible alias = classification (+ scGist reg), matching the
        # previous semantics of the returned loss when entropy is disabled. `train_loss_total` is the true
        # optimized objective (adds weighted attention entropy).
        train_loss_ce = (
            train_loss_total
            if not use_attention_entropy
            else (train_loss_total - train_loss_attention_entropy_weighted)
        )
        self.log("train_loss_ce", train_loss_ce, **options)
        self.log("train_loss_cls", train_loss_cls, **options)
        self.log("train_loss_total", train_loss_total, **options)
        self.log("train_loss_reg", train_loss_reg, **options) if self.model.fs_method == "scGist" else None
        if collect_attention:
            # entropy + neighbourhood sharpness diagnostics; logged whenever attention is collected so a
            # weight=0 (log-only) run is directly comparable to a regularized one.
            self.log("train_attention_entropy", train_attention_entropy, **options)
            if use_attention_entropy:
                self.log("train_loss_attention_entropy_weighted", train_loss_attention_entropy_weighted, **options)
            for key, value in self._attention_sharpness_diagnostics(attention_records).items():
                self.log(key, value, **options)
        self.log("train_overall_acc", self.train_overall_acc, **options)
        self.log("train_macro_acc", self.train_macro_acc, **options)
        self.log("tau", tau_schedule(self.tautype, self.current_epoch, self.trainer.max_epochs), **options)
        return train_loss_total

    def on_train_epoch_end(self):
        # log gene selections and probabilities at the end of each epoch
        if self.model.fs_method == "persist":
            mask_indices, mask_probs = self.model.get_mask_indices()
            metrics = {"epoch": self.current_epoch}

            for i in range(mask_indices.size(0)):
                metrics[f"sel_{i}"] = mask_indices[i].item()

            for i in range(mask_indices.size(0)):
                metrics[f"prob_{i}"] = mask_probs[i].item()

            # get filepath of logger
            path = self.logger._root_dir + "/selections.csv"
            file_exists = os.path.isfile(path)
            with open(path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(metrics.keys())
                writer.writerow([v for v in metrics.values()])
            return

    def validation_step(self, batch, batch_idx):
        # calculate losses and metrics for all nodes in the batch.

        batch_size = batch.input_id.size(0)
        idx = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]
        # metrics calculated for only "root" nodes, not neighbors

        # halfhop or pass-through
        data = self.transform(batch)
        # print("385 ", data.x.shape, len(data.gene_exp_ind)) 385  torch.Size([10602, 503]) 500

        # Optionally collect attention as a DIAGNOSTIC only (never added to val_loss_ce), to check whether
        # learned sharpness generalizes to validation graphs. Collected when regularizing OR when only
        # logging stats (log_attention_stats). Validation classification stays identical either way.
        collect_attention = self.gnn_type == "gat" and (self.attention_entropy_weight > 0 or self.log_attention_stats)
        if collect_attention:
            celltype_pred, attention_records = self.forward(
                gene_exp=data.x[:, data.gene_exp_ind],
                edge_index=data.edge_index,
                xyz=data.x[:, data.xyz_ind],
                subgraph_id=data.subgraph_id,
                tau=tau_schedule(self.tautype, self.current_epoch, self.trainer.max_epochs),
                hard_=True,
                return_attention_weights=True,
            )
        else:
            celltype_pred = self.forward(
                gene_exp=data.x[:, data.gene_exp_ind],
                edge_index=data.edge_index,
                xyz=data.x[:, data.xyz_ind],
                subgraph_id=data.subgraph_id,
                tau=tau_schedule(self.tautype, self.current_epoch, self.trainer.max_epochs),
                hard_=True,
            )
            attention_records = None

        # conditionally remove "slow nodes" (from halfhop)
        if hasattr(data, "slow_node_mask"):
            celltype_pred = celltype_pred[~data.slow_node_mask]

        # calculate losses and metrics
        if self.model.focal_loss:
            val_loss_ce = sigmoid_focal_loss(
                inputs=celltype_pred[idx],
                targets=F.one_hot(batch.celltype[idx], num_classes=self.hparams.model.out_ch).to(torch.float),
                # targets = batch.celltype[idx],
                alpha=0.25,
                gamma=2.0,
                reduction="mean",
            )
        else:
            val_loss_ce = self.loss_ce(celltype_pred[idx], batch.celltype[idx])

        if self.model.fs_method == "scGist":
            val_loss_reg = self.model.feature_regularizer(self.model.logits)
            val_loss_ce += val_loss_reg

        self.val_overall_acc(preds=celltype_pred[idx], target=batch.celltype[idx])  # original
        self.val_f1_overall(preds=celltype_pred[idx], target=batch.celltype[idx])

        self.val_macro_acc(preds=celltype_pred[idx], target=batch.celltype[idx])

        # log losses and metrics
        options = {
            "on_step": self.hparams.logging.on_step,
            "on_epoch": self.hparams.logging.on_epoch,
            "prog_bar": self.hparams.logging.prog_bar,
            "logger": self.hparams.logging.logger,
            "batch_size": batch_size,
        }

        self.log("val_loss_ce", val_loss_ce, **options)
        self.log("val_loss_reg", val_loss_reg, **options) if self.model.fs_method == "scGist" else None
        self.log("val_overall_acc", self.val_overall_acc, **options)
        self.log("val_macro_acc", self.val_macro_acc, **options)
        self.log("val_f1_overall", self.val_f1_overall, **options)

        # attention-sharpness diagnostics (never affects val_loss_ce or checkpoint selection)
        if collect_attention:
            with torch.no_grad():
                val_attention_entropy = torch.stack([
                    self._gr2st_attention_entropy(alpha) for _, alpha in attention_records
                ]).mean()
            self.log("val_attention_entropy", val_attention_entropy, **options)
            for key, value in self._attention_sharpness_diagnostics(attention_records).items():
                self.log(key.replace("train_", "val_"), value, **options)

    def on_validation_epoch_end(self):
        pass

    def predict_step(self, batch, batch_idx):
        # calculate losses and metrics for all nodes in the batch.

        batch_size = batch.input_id.size(0)
        idx = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]

        # halfhop or pass-through
        data = self.transform(batch)

        celltype_pred = self.forward(
            gene_exp=data.x[:, data.gene_exp_ind],
            edge_index=data.edge_index,
            xyz=data.x[:, data.xyz_ind],
            subgraph_id=data.subgraph_id,
            tau=tau_schedule(self.tautype, self.current_epoch, self.trainer.max_epochs),
            hard_=True,
        )

        # conditionally remove "slow nodes" (from halfhop)
        if hasattr(data, "slow_node_mask"):
            celltype_pred = celltype_pred[~data.slow_node_mask]

        return [celltype_pred[idx], batch.celltype[idx]]

    def on_predict_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        # calculate losses and metrics for all nodes in the batch.

        batch_size = batch.input_id.size(0)
        idx = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]
        # metrics calculated for only "root" nodes, not neighbors

        # halfhop or pass-through
        data = self.transform(batch)

        celltype_pred = self.forward(
            gene_exp=data.x[:, data.gene_exp_ind],
            edge_index=data.edge_index,
            xyz=data.x[:, data.xyz_ind],
            subgraph_id=data.subgraph_id,
            tau=tau_schedule(self.tautype, self.current_epoch, self.trainer.max_epochs),
            hard_=True,
        )

        # conditionally remove "slow nodes" (from halfhop)
        if hasattr(data, "slow_node_mask"):
            celltype_pred = celltype_pred[~data.slow_node_mask]

        # calculate losses and metrics
        if self.model.focal_loss:
            test_loss_ce = sigmoid_focal_loss(
                inputs=celltype_pred[idx],
                targets=F.one_hot(batch.celltype[idx], num_classes=self.hparams.model.out_ch).to(torch.float),
                # targets = batch.celltype[idx],
                alpha=0.25,
                gamma=2.0,
                reduction="mean",
            )
        else:
            test_loss_ce = self.loss_ce(celltype_pred[idx], batch.celltype[idx])

        self.test_overall_acc(preds=celltype_pred[idx], target=batch.celltype[idx])
        self.test_macro_acc(preds=celltype_pred[idx], target=batch.celltype[idx])
        self.test_micro_acc(preds=celltype_pred[idx], target=batch.celltype[idx])

        self.test_f1_overall(preds=celltype_pred[idx], target=batch.celltype[idx])
        self.test_f1_macro(preds=celltype_pred[idx], target=batch.celltype[idx])
        self.test_f1_micro(preds=celltype_pred[idx], target=batch.celltype[idx])

        # log losses and metrics
        options = {
            "on_step": self.hparams.logging.on_step,
            "on_epoch": self.hparams.logging.on_epoch,
            "prog_bar": self.hparams.logging.prog_bar,
            "logger": self.hparams.logging.logger,
            "batch_size": batch_size,
        }

        self.log("test_loss_ce", test_loss_ce, **options)
        self.log("test_overall_acc", self.test_overall_acc, **options)
        self.log("test_macro_acc", self.test_macro_acc, **options)
        self.log("test_micro_acc", self.test_micro_acc, **options)

        self.log("test_f1_overall", self.test_f1_overall, **options)
        self.log("test_f1_macro", self.test_f1_macro, **options)
        self.log("test_f1_micro", self.test_f1_micro, **options)
        self.test_pred.append(celltype_pred[idx])
        self.test_labels.append(batch.celltype[idx])

    def on_test_epoch_end(self):
        all_predictions = []
        all_labels = []

        for i in range(len(self.test_pred)):
            all_predictions.append(self.test_pred[i])
            all_labels.append(self.test_labels[i])

        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        # Now you can save these tensors to a file
        # For example, saving to a .pt file
        path = self.logger._root_dir + "/test_pred.pt"
        torch.save({"predictions": all_predictions, "labels": all_labels}, path)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.trainer.lr)

        multistep_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)

        if self.hparams.trainer.lr_scheduler == "multistep":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": multistep_scheduler,
                    "interval": "epoch",
                    # interval="epoch" means scheduler is called at epoch boundaries
                    # frequency=1 means scheduler is called every interval (every epoch)
                    # e.g. frequency=2 would call scheduler every 2 epochs
                    "frequency": 1,
                },
            }

        elif self.hparams.trainer.lr_scheduler == "constant":
            return optimizer


def tau_schedule(type, epoch, total_epoch):

    if type == "old":
        start_tau = 10
        end_tau = 0.01
        tau = start_tau * (end_tau / start_tau) ** (epoch / total_epoch)
        return tau

    start_tau = 10.0

    mid_tau = 2.0

    end_tau = 0.05

    explore_frac = 0.6

    progress = epoch / max(total_epoch - 1, 1)

    if progress < explore_frac:
        p = progress / explore_frac

        return start_tau * (mid_tau / start_tau) ** p

    p = (progress - explore_frac) / (1 - explore_frac)

    return mid_tau * (end_tau / mid_tau) ** p


def old_tau_schedule(type, epoch, total_epoch):
    start_tau = 10
    end_tau = 0.01
    # end_tau = 0.1

    if type == "exp":
        tau = start_tau * (end_tau / start_tau) ** (epoch / total_epoch)
    elif type == "constant":
        tau = 0.1
    return tau
