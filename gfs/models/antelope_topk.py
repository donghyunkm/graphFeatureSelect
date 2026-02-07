import csv
import os

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision.ops import sigmoid_focal_loss

from gfs.models.transforms import HalfHop
from gfs.models.get_sampler import SamplerArgs, get_sampler
from gfs.models.samplers.sfess.sfess import score_function_estimator

from functools import partial


class SubsetLayer(nn.Module):

    def __init__(self, subset_layer, k, num_samples):
        super(SubsetLayer, self).__init__()
        self.subset = subset_layer
        self.k = k
        self.num_samples = num_samples

    def forward(self, logits, tau):
        if self.training:
            res = self.subset(logits, tau)
            return res
        else:
            indices = torch.topk(logits.squeeze(-1), self.k, dim=1)[1]
            khot = F.one_hot(indices, logits.size(1)).sum(1).float()
            khot = khot.unsqueeze(0).unsqueeze(-1).expand(self.num_samples, -1, -1, -1)
            return khot, None


def get_subset_layer(k, args):
    name = {
        "sfess": "sfess",
        "sfess-v": "sfess",
        "gumbel": "gumbel",
        "st-gumbel": "gumbel",
        "simple": "simple",
        "imle": "imle",
        "pps": "pps",
    }[args.sampler]
    sampler_args = SamplerArgs(
        name=name,
        sample_k=k,
        n_samples=args.num_samples,
        noise_scale=args.noise_scale,
        beta=args.beta,
        tau=args.tau,
        hard=args.sampler != "gumbel",
        pps_gradient=args.pps_gradient,
        pps_activation=args.pps_activation,
        pps_sample=args.pps_sample,
    )
    sampler = get_sampler(sampler_args, device=args.device)
    subset_layer = SubsetLayer(sampler, k, args.num_samples)
    return subset_layer

def get_sfe(args):
    estimator = {
        "sfess": "reinforce",
        "sfess-v": "vimco",
        "gumbel": None,
        "st-gumbel": None,
        "simple": None,
        "imle": None,
        "pps": None,
    }[args.sampler]
    return partial(score_function_estimator, estimator=estimator)

class MLP(nn.Module):
    '''
    Multilayer perceptron (MLP) model.

    Args:
      input_size: number of inputs.
      output_size: number of outputs.
      hidden: list of hidden layer widths.
      activation: nonlinearity between layers.
    '''

    def __init__(self,
                 input_size,
                 output_size,
                 hidden,
                 activation=nn.ReLU()):
        super().__init__()

        # Fully connected layers.
        self.input_size = input_size
        self.output_size = output_size
        fc_layers = [nn.Linear(d_in, d_out) for d_in, d_out in
                     zip([input_size] + hidden, hidden + [output_size])]
        self.fc = nn.ModuleList(fc_layers)

        # Activation function.
        self.activation = activation

    def forward(self, x):
        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.activation(x)

        return self.fc[-1](x)


class FeatureRegularizer(nn.Module):
    def __init__(self, l1=0.1, panel_size=None, priority_score=None, pairs=None, alpha=0.5, beta=0.5, gamma=0.5, strict=True):
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
        reg = torch.tensor(0., dtype=x.dtype, device=x.device)

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
        k,
        num_samples,
        cfg_topk
    ):
        super(GnnFs, self).__init__()
        self.fs_method = fs_method

        if self.fs_method == "persist":
            self.logits = nn.Parameter(torch.randn(n_select, gene_ch))
        elif self.fs_method == "scGist":
            self.logits = nn.Parameter(torch.full((1, gene_ch), 0.5)) 
        elif self.fs_method == "topk":
            self.logits = nn.Parameter(torch.randn(gene_ch))
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
        self.k = k
        self.num_samples = num_samples
        self.cfg_topk = cfg_topk
        self.extra = None
        self.lin_in = torch.nn.Linear(gene_ch, hid_ch)
        self.focal_loss = focal_loss
        if self.fs_method == "scGist":
            self.feature_regularizer = FeatureRegularizer(l1=0.1, panel_size=n_select)

        if not self.pre_linear:
            if gnn == "gat":
                self.gnn_layers.append(
                    GATConv(gene_ch, hid_ch, heads=heads, concat=False, add_self_loops=False, bias=False)
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
                    GATConv(hid_ch, hid_ch, heads=heads, concat=False, add_self_loops=False, bias=False)
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
            self.res_lin = MLP(gene_ch, out_ch, [128, 128], nn.ReLU()) # new (from Persist)
        if self.xyz_status:
            self.xyz_lin = torch.nn.Linear(spatial_ch, out_ch)

        self.subset_layer = get_subset_layer(self.k, self.cfg_topk)


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

    def sample_mask(self, logits, batch, subgraph_id, tau):
        if logits.dim() == 1:
            logits = logits.expand(batch, -1)


        if self.training:
            n_subgraphs = subgraph_id.max() + 1
            mask_list = []
            for i in range(n_subgraphs):
                mask, extra = self.subset_layer(logits.unsqueeze(-1), tau)
                mask = mask.squeeze(-1)
                # print("MASK295295 ", mask.shape) MASK295295  torch.Size([1, 1, 500])
                # print("MASK295HARD ", mask) 10 1s,490 0s when hard sampling is used
                mask_list.append(mask)
                self.extra = extra #??

            subgraph_id = subgraph_id.tolist()
            # return_mask = mask_list[subgraph_id]
            return_mask = [mask_list[i] for i in subgraph_id]
            return_mask = torch.stack(return_mask)
            return_mask = return_mask.squeeze()
            return return_mask
        else:
            indices = torch.topk(logits, self.k, dim=-1)[1] 
            mask = F.one_hot(indices, logits.size(-1)).sum(1)
            mask = mask.float()
            mask = mask.expand(self.num_samples, -1, -1)
            mask = mask.squeeze()
            return mask

# SAMPLEMAASK
# torch.Size([500]) 1
# logits2  torch.Size([1, 500])
# MASKVAL  torch.Size([1, 1, 500])

# SAMPLEMAASK
# torch.Size([500]) 1
# logits2  torch.Size([1, 500])
# MASKTRAIN  torch.Size([1, 1, 500])

    def get_mask(self, tau, subgraph_id):
        """
        Get soft k-hot mask
        """
        if self.training:
            # sample soft k-hot vectors for each subgraph in the batch during training
            # extra samples are harmless, makes indexing more straightforward;
            # case: e.g. subgraph_id.unique() = [0, 2, 3, 4]
            n_subgraphs = subgraph_id.max() + 1
            # sample has dims (n_selections, n_subgraphs, n_features)
            sample = F.gumbel_softmax(self.logits.unsqueeze(1).repeat(1, n_subgraphs, 1), tau=tau, hard=False, dim=-1)
            # k_hot has dims (n_subgraphs, n_features)
            k_hot = torch.max(sample, dim=0).values
            # repeat k-hot masks for each node based on their subgraph membership
            return k_hot[subgraph_id]
        else:
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

    def forward(self, x, edge_index, xyz, subgraph_id, tau, hard_):

        for id in subgraph_id.unique():
            xyz_mean = torch.mean(xyz[subgraph_id == id])
            xyz[subgraph_id == id] = xyz[subgraph_id == id] - xyz_mean

        mask = self.sample_mask(self.logits, 1, subgraph_id, tau)
        print("361XX ", x.shape)
        print("MASK361 ", mask)

        x = mask * x
        x = torch.squeeze(x)

        if self.x_res:
            x_to_add = self.res_lin(x)
        if self.xyz_status:
            xyz = self.xyz_lin(xyz)

        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_final = 0

        for i, layer in enumerate(self.gnn_layers):
            if self.res:
                x = layer(x, edge_index) + self.lins[i](x)
            else:
                x = layer(x, edge_index)
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

        return x


class LitGnnFs(L.LightningModule):
    def __init__(self, config):
        super(LitGnnFs, self).__init__()

        self.save_hyperparameters(config)

        # model
        cfg = self.hparams.model
        cfg_topk = self.hparams.topk
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
            fs_method = cfg.fs_method,
            focal_loss = cfg.focal_loss,
            k = cfg_topk.k,
            num_samples = cfg_topk.num_samples,
            cfg_topk = cfg_topk
        )

        # related to training step
        self.halfhop = cfg.halfhop
        if self.halfhop:
            self.transform = HalfHop(alpha=0.5, p=0.5)
        else:
            self.transform = lambda x: x

        self.tautype = cfg.tautype

        self.trainmode = cfg.trainmode # 0 uses all training nodes; 1 uses only root nodes

        
        self.sfe = get_sfe(cfg_topk)
        
        self.k = cfg_topk.k
        # losses
        self.loss_ce = nn.CrossEntropyLoss()

        # metrics
        # options = {"num_classes": cfg.out_ch, "top_k": 1, "multidim_average": "global"}
        options = {"num_classes": cfg.out_ch, "top_k": 1, "multidim_average": "global"}

        self.train_overall_acc = MulticlassAccuracy(average="weighted", **options)
        self.val_overall_acc = MulticlassAccuracy(average="weighted", **options)

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

    def forward(self, gene_exp, edge_index, xyz, subgraph_id, tau, hard_):
        """
        Args:
            gene_exp (torch.Tensor): (n_nodes, n_genes)
            edge_index (torch.Tensor): (2, n_edges)
            xyz (torch.Tensor): (n_nodes, 3)
            subgraph_id (torch.Tensor): (n_nodes)
            tau (float): temperature for Gumbel-Softmax
            hard_ (bool): if True, use hard Gumbel-Softmax
        """

        celltype = self.model(gene_exp, edge_index, xyz, subgraph_id, tau, hard_)
        return celltype

    def training_step(self, batch, batch_idx):
        # calculate losses and metrics for all training nodes in the batch.
        batch_size = torch.sum(batch.train_mask)
        data = self.transform(batch)

        if self.trainmode == 0:
            # backprop/metrics for all nodes in batch (including neighbors)
            idx = data.train_mask
        elif self.trainmode == 1:
            # backprop/metrics for only "root" nodes, not neighbors
            idx = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]
        celltype_pred = self.forward(
            gene_exp=data.x[:, data.gene_exp_ind],
            edge_index=data.edge_index,
            xyz=data.x[:, data.xyz_ind],
            subgraph_id=data.subgraph_id,
            tau=tau_schedule(self.tautype, self.current_epoch, self.trainer.max_epochs),
            hard_=False,
        )

        # conditionally remove "slow nodes" (from halfhop)
        if hasattr(data, "slow_node_mask"):
            celltype_pred = celltype_pred[~data.slow_node_mask]

        # calculate loss
        if self.model.focal_loss:
            train_loss_ce = sigmoid_focal_loss(
                inputs=celltype_pred[idx],
                targets=F.one_hot(data.celltype[idx], num_classes=self.hparams.model.out_ch).to(torch.float),
                # targets = data.celltype[idx],
                alpha=0.25,
                gamma=2.0,
                reduction="mean",
            )
        else:
            train_loss_ce = self.loss_ce(celltype_pred[idx], data.celltype[idx])

        if self.model.fs_method == "scGist":
            train_loss_reg = self.model.feature_regularizer(self.model.logits) * 100
            train_loss_ce += train_loss_reg 

        train_loss_ce = train_loss_ce.unsqueeze(0)
        train_loss_ce = self.sfe(train_loss_ce, self.model.extra) # only methods that require score function estimates return something for .extra
        train_loss_ce = train_loss_ce.squeeze()

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

        self.log("train_loss_ce", train_loss_ce, **options)
        self.log("train_loss_reg", train_loss_reg, **options) if self.model.fs_method == "scGist" else None
        self.log("train_overall_acc", self.train_overall_acc, **options)
        self.log("train_macro_acc", self.train_macro_acc, **options)
        self.log("tau", tau_schedule(self.tautype, self.current_epoch, self.trainer.max_epochs), **options)
        return train_loss_ce

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


        self.val_overall_acc(preds=celltype_pred[idx], target=batch.celltype[idx]) # original


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

        multistep_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[100],
            gamma=0.1 
        )

        if self.hparams.trainer.lr_scheduler == "multistep":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": multistep_scheduler,
                    "interval": "epoch",
                    # interval="epoch" means scheduler is called at epoch boundaries
                    # frequency=1 means scheduler is called every interval (every epoch)
                    # e.g. frequency=2 would call scheduler every 2 epochs
                    "frequency": 1
                }
            }
            
        elif self.hparams.trainer.lr_scheduler == "constant":
            return optimizer

def tau_schedule(type, epoch, total_epoch):
    start_tau = 10
    end_tau = 0.01 
    # end_tau = 0.1

    if type == 'exp':
        tau = start_tau * (end_tau / start_tau) ** (epoch / total_epoch)
    elif type == "constant":
        tau = 0.1
    return tau