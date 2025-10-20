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


class Persist(torch.nn.Module):
    """
    This model features:
    1. a GNN backbone adapted from `Classic GNNs are strong baselines...` (Luo et al. 2024)
    2. a feature selection layer based on concrete variables (Jang. et al. 2016)

    Args:
        gene_ch (int): number of input features
        hid_ch (int): number of hidden features
        out_ch (int): number of output features
        n_select (int): number of features to select
    """

    def __init__(
        self,
        gene_ch,
        hid_ch,
        out_ch,
        n_select,
        mask_type
    ):
        super(Persist, self).__init__()

        self.logits = nn.Parameter(torch.randn(n_select, gene_ch))
        self.mlp = MLP(gene_ch, out_ch, [128, 128], nn.ReLU())
        self.mask_type = mask_type

    def get_mask(self, tau, subgraph_id, x):
        """
        Get soft k-hot mask
        """
        if self.training:
            # sample soft k-hot vectors for each subgraph in the batch during training
            # extra samples are harmless, makes indexing more straightforward;
            # case: e.g. subgraph_id.unique() = [0, 2, 3, 4]
            n_subgraphs = subgraph_id.max() + 1
            # sample has dims (n_selections, n_subgraphs, n_features)

            if self.mask_type == 0:
                sample = F.gumbel_softmax(self.logits.unsqueeze(1).repeat(1, n_subgraphs, 1), tau=tau, hard=False, dim=-1) # -> original (1 mask per subgraph)
            else:
                sample = F.gumbel_softmax(self.logits.unsqueeze(1).repeat(1, len(x), 1), tau=tau, hard=False, dim=-1) # use this to sample 1 mask for each cell
            # sample = F.gumbel_softmax(self.logits.unsqueeze(1).repeat(1, 1, 1), tau=tau, hard=False, dim=-1) # use this to just sample 1 mask!
            # sample = F.gumbel_softmax(self.logits.unsqueeze(1).repeat(1, len(x), 1), tau=tau, hard=False, dim=-1) # use this to sample 1 mask for each cell
            # k_hot has dims (n_subgraphs, n_features)
            k_hot = torch.max(sample, dim=0).values
            # repeat k-hot masks for each node based on their subgraph membership
            if self.mask_type == 0:
                return k_hot[subgraph_id] # -> original (1 mask per subgraph)
            else:
                return k_hot # use this to return 1 same mask for all or to return 1 mask per cell
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

    def forward(self, x, subgraph_id, tau):
        mask = self.get_mask(tau, subgraph_id, x)
        x = mask * x
        pred = self.mlp(x)
        return pred



class LitPersist(L.LightningModule):
    def __init__(self, config):
        super(LitPersist, self).__init__()

        self.save_hyperparameters(config)

        # model
        cfg = self.hparams.model
        self.model = Persist(gene_ch=cfg.gene_ch,
                             hid_ch=cfg.hid_ch,
                             out_ch=cfg.out_ch,
                             n_select=cfg.n_select,
                             mask_type=cfg.mask_type)



        self.tautype = cfg.tautype
        # losses
        self.loss_ce = nn.CrossEntropyLoss()
        # metrics
        options = {"num_classes": cfg.out_ch, "top_k": 1, "multidim_average": "global"}
        
        self.train_overall_acc = MulticlassAccuracy(average="weighted", **options)
        self.val_overall_acc = MulticlassAccuracy(average="weighted", **options)


        self.transform = lambda x: x


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

        celltype = self.model(gene_exp, subgraph_id, tau)
        return celltype

    def training_step(self, batch, batch_idx):
        # calculate losses and metrics for all training nodes in the batch.
        batch_size = torch.sum(batch.train_mask)
        data = self.transform(batch)
        idx = torch.where(data.n_id == data.input_id.unsqueeze(-1))[0]

        celltype_pred = self.forward(
            gene_exp=data.x[:, data.gene_exp_ind],
            edge_index=data.edge_index,
            xyz=data.x[:, data.xyz_ind],
            subgraph_id=data.subgraph_id,
            tau=self.tau_schedule(self.tautype, self.current_epoch, self.trainer.max_epochs),
            hard_=False,
        )

        # conditionally remove "slow nodes" (from halfhop)
        if hasattr(data, "slow_node_mask"):
            celltype_pred = celltype_pred[~data.slow_node_mask]

        # calculate loss
        train_loss_ce = self.loss_ce(celltype_pred[idx], data.celltype[idx])

        # calculate metrics
        self.train_overall_acc(
            preds=celltype_pred[idx], target=data.celltype[idx]
        )

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
        self.log("tau", self.tau_schedule(self.tautype, self.current_epoch, self.trainer.max_epochs), **options)
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
            tau=self.tau_schedule(self.tautype, self.current_epoch, self.trainer.max_epochs),
            hard_=True,
        )

        # conditionally remove "slow nodes" (from halfhop)
        if hasattr(data, "slow_node_mask"):
            celltype_pred = celltype_pred[~data.slow_node_mask]

        # calculate losses and metrics
        val_loss_ce = self.loss_ce(celltype_pred[idx], batch.celltype[idx])
        self.val_overall_acc(preds=celltype_pred[idx], target=batch.celltype[idx])

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
            tau=self.tau_schedule(self.tautype, self.current_epoch, self.trainer.max_epochs),
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
        
        lr_factor = 0.5
        lookback = 10
        min_lr = 1e-5
        verbose = True
        persist_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lr_factor, patience=lookback // 2, min_lr=min_lr,
            verbose=verbose)


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
        elif self.hparams.trainer.lr_scheduler == "persist":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": persist_scheduler,
                    "interval": "epoch",
                    # interval="epoch" means scheduler is called at epoch boundaries
                    # frequency=1 means scheduler is called every interval (every epoch)
                    # e.g. frequency=2 would call scheduler every 2 epochs
                    "frequency": 1,
                    "monitor": "val_loss_ce"
                }
            }
        
    def tau_schedule(self, type, epoch, total_epoch):
        start_tau = 10
        end_tau = 0.01 
        # end_tau = 0.1
        mbsize = self.hparams.data.batch_size
        max_nepochs = self.hparams.trainer.max_epochs

        if type == 'exp':
            tau = start_tau * (end_tau / start_tau) ** (epoch / total_epoch)
        elif type == "constant":
            tau = 0.1
        elif type == "persist":
            tau = start_tau * (end_tau / start_tau) ** ( (epoch * mbsize) / (46333 * max_nepochs) )


        return tau
