"""Lightning module for GNN with feature selection."""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from gfs.models.backbone import GNNBackbone
from gfs.models.heads import ClassificationHead, ReconstructionHead
from gfs.models.feature_selection import get_feature_selector


class LitGnnFs(L.LightningModule):
    """Lightning module wiring FeatureSelector + GNNBackbone + TaskHead.

    Args:
        config: Hydra DictConfig with keys: backbone, feature_selection, task,
                data, trainer, logging, and global flags (n_select, trainmode, lam, halfhop)
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        cfg = self.hparams

        # These will be set by the training script after DataModule.setup()
        self.n_genes: int = 0
        self.n_classes: int = 0
        self.spatial_ch: int = len(cfg.data.spatial_cols)

    def setup_model(self, n_genes: int, n_classes: int):
        """Initialize model components. Called after DataModule provides dimensions.

        This is separate from __init__ because n_genes and n_classes come from the data,
        not from the config (they depend on filtering and label encoding).
        """
        cfg = self.hparams
        self.n_genes = n_genes
        self.n_classes = n_classes

        # Feature selector
        fs_cfg = cfg.feature_selection
        self.feature_selector = get_feature_selector(
            fs_method=fs_cfg.method,
            n_genes=n_genes,
            n_select=cfg.n_select,
            sigma=getattr(fs_cfg, 'sigma', 0.5),
            l1=getattr(fs_cfg, 'l1', 0.1),
        )

        # GNN backbone
        bb_cfg = cfg.backbone
        self.backbone = GNNBackbone(
            gene_ch=n_genes,
            spatial_ch=self.spatial_ch,
            hid_ch=bb_cfg.hid_ch,
            n_layers=bb_cfg.n_layers,
            gnn_type=bb_cfg.gnn_type,
            dropout=bb_cfg.dropout,
            heads=bb_cfg.heads,
            pre_linear=bb_cfg.pre_linear,
            residual=bb_cfg.residual,
            layer_norm=bb_cfg.layer_norm,
            batch_norm=bb_cfg.batch_norm,
            jk=bb_cfg.jk,
            xyz_proj=bb_cfg.xyz_proj,
            x_residual=bb_cfg.x_residual,
        )

        # Task head
        task_cfg = cfg.task
        if task_cfg.name == "classification":
            self.task_head = ClassificationHead(bb_cfg.hid_ch, n_classes)
            self.loss_fn = nn.CrossEntropyLoss()
        elif task_cfg.name == "reconstruction":
            hidden = list(task_cfg.get("hidden", [128, 128]))
            self.task_head = ReconstructionHead(bb_cfg.hid_ch, n_genes, hidden)
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task: {task_cfg.name}")

        # Metrics (classification only)
        if task_cfg.name == "classification":
            opts = {"num_classes": n_classes, "top_k": 1, "multidim_average": "global"}
            self.train_acc = MulticlassAccuracy(average="weighted", **opts)
            self.train_macro_acc = MulticlassAccuracy(average="macro", **opts)
            self.val_acc = MulticlassAccuracy(average="weighted", **opts)
            self.val_macro_acc = MulticlassAccuracy(average="macro", **opts)
            self.test_acc = MulticlassAccuracy(average="weighted", **opts)
            self.test_macro_acc = MulticlassAccuracy(average="macro", **opts)
            self.test_f1_macro = MulticlassF1Score(average="macro", **opts)

        # Config references
        self.trainmode = cfg.trainmode
        self.lam = cfg.lam
        self.tautype = getattr(cfg.feature_selection, 'tautype', 'constant')

    def forward(self, gene_exp, edge_index, xyz, subgraph_id=None, tau=1.0):
        """Full forward: feature selection -> backbone -> task head."""
        # Default: treat entire batch as one subgraph (uniform mask)
        if subgraph_id is None:
            subgraph_id = torch.zeros(gene_exp.size(0), dtype=torch.long, device=gene_exp.device)
        masked_exp = self.feature_selector(gene_exp, tau=tau, subgraph_id=subgraph_id)
        embeddings = self.backbone(masked_exp, edge_index, xyz, subgraph_id)
        return self.task_head(embeddings)

    def _get_tau(self):
        """Compute temperature for current epoch."""
        return tau_schedule(
            self.tautype,
            self.current_epoch,
            self.trainer.max_epochs,
        )

    def _seed_node_idx(self, batch):
        """Find indices of seed nodes in the batch.

        NeighborLoader puts seed nodes first: indices 0..batch_size-1.
        """
        return torch.arange(batch.input_id.size(0), device=batch.y.device)

    def _log_metrics(self, metrics: dict, batch_size: int):
        """DRY logging helper."""
        log_cfg = self.hparams.logging
        opts = {
            "on_step": log_cfg.on_step,
            "on_epoch": log_cfg.on_epoch,
            "prog_bar": log_cfg.prog_bar,
            "logger": log_cfg.logger,
            "batch_size": batch_size,
        }
        for name, value in metrics.items():
            self.log(name, value, **opts)

    def training_step(self, batch, batch_idx):
        tau = self._get_tau()
        pred = self.forward(
            batch.gene_exp, batch.edge_index, batch.xyz,
            subgraph_id=getattr(batch, 'subgraph_id', None),
            tau=tau,
        )

        # Select which nodes to compute loss on
        if self.trainmode == 0:
            idx = batch.train_mask
        else:
            idx = self._seed_node_idx(batch)

        # Compute loss
        task_cfg = self.hparams.task
        if task_cfg.name == "classification":
            loss_task = self.loss_fn(pred[idx], batch.y[idx])
        else:
            loss_task = self.loss_fn(pred[idx], batch.gene_exp[idx])

        reg_loss = self.feature_selector.regularization_loss()
        total_loss = loss_task + self.lam * reg_loss

        # Log
        batch_size = batch.input_id.size(0)
        metrics = {"train_loss": loss_task, "reg_loss": reg_loss, "tau": tau}
        if task_cfg.name == "classification":
            self.train_acc(pred[idx], batch.y[idx])
            self.train_macro_acc(pred[idx], batch.y[idx])
            metrics["train_acc"] = self.train_acc
            metrics["train_macro_acc"] = self.train_macro_acc
        self._log_metrics(metrics, batch_size)

        return total_loss

    def validation_step(self, batch, batch_idx):
        tau = self._get_tau()
        pred = self.forward(
            batch.gene_exp, batch.edge_index, batch.xyz,
            subgraph_id=getattr(batch, 'subgraph_id', None),
            tau=tau,
        )

        # Seed-node-only eval
        idx = self._seed_node_idx(batch)

        task_cfg = self.hparams.task
        if task_cfg.name == "classification":
            loss_task = self.loss_fn(pred[idx], batch.y[idx])
        else:
            loss_task = self.loss_fn(pred[idx], batch.gene_exp[idx])

        batch_size = batch.input_id.size(0)
        metrics = {"val_loss": loss_task}
        if task_cfg.name == "classification":
            self.val_acc(pred[idx], batch.y[idx])
            self.val_macro_acc(pred[idx], batch.y[idx])
            metrics["val_acc"] = self.val_acc
            metrics["val_macro_acc"] = self.val_macro_acc
        self._log_metrics(metrics, batch_size)

    def test_step(self, batch, batch_idx):
        tau = self._get_tau()
        pred = self.forward(
            batch.gene_exp, batch.edge_index, batch.xyz,
            subgraph_id=getattr(batch, 'subgraph_id', None),
            tau=tau,
        )

        idx = self._seed_node_idx(batch)

        task_cfg = self.hparams.task
        if task_cfg.name == "classification":
            loss_task = self.loss_fn(pred[idx], batch.y[idx])
        else:
            loss_task = self.loss_fn(pred[idx], batch.gene_exp[idx])

        batch_size = batch.input_id.size(0)
        metrics = {"test_loss": loss_task}
        if task_cfg.name == "classification":
            self.test_acc(pred[idx], batch.y[idx])
            self.test_macro_acc(pred[idx], batch.y[idx])
            self.test_f1_macro(pred[idx], batch.y[idx])
            metrics["test_acc"] = self.test_acc
            metrics["test_macro_acc"] = self.test_macro_acc
            metrics["test_f1_macro"] = self.test_f1_macro
        self._log_metrics(metrics, batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.trainer.lr)

        if self.hparams.trainer.lr_scheduler == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[100], gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        return optimizer


def tau_schedule(tautype: str, epoch: int, max_epochs: int) -> float:
    """Temperature schedule for Gumbel-softmax."""
    if tautype == "exp":
        start, end = 10.0, 0.01
        # Avoid division by zero
        progress = epoch / max(max_epochs, 1)
        return start * (end / start) ** progress
    else:
        return 0.1
