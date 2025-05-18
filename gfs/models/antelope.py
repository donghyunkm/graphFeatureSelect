import csv
import os

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torchmetrics.classification import MulticlassAccuracy

from gfs.models.transforms import HalfHop

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
    ):
        super(GnnFs, self).__init__()

        self.logits = nn.Parameter(torch.randn(n_select, gene_ch))
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

        self.lin_in = torch.nn.Linear(gene_ch, hid_ch)

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
        mask = self.get_mask(tau, subgraph_id)
        x = mask * x

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
        )

        # related to training step
        self.halfhop = cfg.halfhop
        if self.halfhop:
            self.transform = HalfHop(alpha=0.5, p=0.5)
        else:
            self.transform = lambda x: x

        self.tautype = cfg.tautype

        self.trainmode = cfg.trainmode # 0 uses all training nodes; 1 uses only root nodes

        # losses
        self.loss_ce = nn.CrossEntropyLoss()

        # metrics
        # options = {"num_classes": cfg.out_ch, "top_k": 1, "multidim_average": "global"}
        options = {"num_classes": cfg.out_ch, "top_k": 1, "multidim_average": "global"}

        self.train_overall_acc = MulticlassAccuracy(average="weighted", **options)
        self.val_overall_acc = MulticlassAccuracy(average="weighted", **options)

        # self.train_macro_acc = MulticlassAccuracy(average="macro", **options)
        # self.val_macro_acc = MulticlassAccuracy(average="macro", **options)

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
            idx = data.train_mask
        elif self.trainmode == 1:
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
        train_loss_ce = self.loss_ce(celltype_pred[idx], data.celltype[idx])

        # calculate metrics

        self.train_overall_acc(preds=celltype_pred[idx], target=data.celltype[idx])
        # self.train_macro_acc(preds=celltype_pred[idx], target=data.celltype[idx])


        # log losses and metrics
        options = {
            "on_step": self.hparams.logging.on_step,
            "on_epoch": self.hparams.logging.on_epoch,
            "prog_bar": self.hparams.logging.prog_bar,
            "logger": self.hparams.logging.logger,
            "batch_size": batch_size,
        }

        self.log("train_loss_ce", train_loss_ce, **options)
        self.log("train_overall_acc", self.train_overall_acc, **options)
        # self.log("train_macro_acc", self.train_macro_acc, **options)
        self.log("tau", tau_schedule(self.tautype, self.current_epoch, self.trainer.max_epochs), **options)
        return train_loss_ce

    def on_train_epoch_end(self):
        # log gene selections and probabilities at the end of each epoch
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
        val_loss_ce = self.loss_ce(celltype_pred[idx], batch.celltype[idx])
        self.val_overall_acc(preds=celltype_pred[idx], target=batch.celltype[idx])
        # self.val_macro_acc(preds=celltype_pred[idx], target=batch.celltype[idx])

        # log losses and metrics
        options = {
            "on_step": self.hparams.logging.on_step,
            "on_epoch": self.hparams.logging.on_epoch,
            "prog_bar": self.hparams.logging.prog_bar,
            "logger": self.hparams.logging.logger,
            "batch_size": batch_size,
        }

        self.log("val_loss_ce", val_loss_ce, **options)
        self.log("val_overall_acc", self.val_overall_acc, **options)
        # self.log("val_macro_acc", self.val_macro_acc, **options)

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