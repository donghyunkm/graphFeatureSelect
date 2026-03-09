import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision.ops import sigmoid_focal_loss

from gfs.models.backbone import GnnFs
from gfs.models.transforms import HalfHop


class LitGnnFs(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters(config)

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
            fs_method=cfg.fs_method,
            focal_loss=cfg.focal_loss,
            sigma=cfg.sigma,
            lam=cfg.lam,
        )

        self.halfhop = cfg.halfhop
        if self.halfhop:
            self.transform = HalfHop(alpha=0.5, p=0.5)
        else:
            self.transform = lambda x: x

        self.tautype = cfg.tautype
        self.trainmode = cfg.trainmode

        self.loss_ce = nn.CrossEntropyLoss()

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
        celltype = self.model(gene_exp, edge_index, xyz, subgraph_id, tau, hard_)
        return celltype

    def _compute_loss(self, celltype_pred, celltype, idx):
        if self.model.focal_loss:
            loss_ce = sigmoid_focal_loss(
                inputs=celltype_pred[idx],
                targets=F.one_hot(celltype[idx], num_classes=self.hparams.model.out_ch).to(torch.float),
                alpha=0.25,
                gamma=2.0,
                reduction="mean",
            )
        else:
            loss_ce = self.loss_ce(celltype_pred[idx], celltype[idx])

        reg_loss = self.model.feature_selector.regularization_loss()
        total_loss = loss_ce + self.model.lam * reg_loss

        return total_loss, loss_ce, reg_loss

    def _forward_batch(self, batch, hard_):
        data = self.transform(batch)
        tau = tau_schedule(self.tautype, self.current_epoch, self.trainer.max_epochs)

        celltype_pred = self.forward(
            gene_exp=data.x[:, data.gene_exp_ind],
            edge_index=data.edge_index,
            xyz=data.x[:, data.xyz_ind],
            subgraph_id=data.subgraph_id,
            tau=tau,
            hard_=hard_,
        )

        if hasattr(data, "slow_node_mask"):
            celltype_pred = celltype_pred[~data.slow_node_mask]

        return data, celltype_pred, tau

    def training_step(self, batch, batch_idx):
        batch_size = torch.sum(batch.train_mask)

        if self.trainmode == 0:
            data, celltype_pred, tau = self._forward_batch(batch, hard_=False)
            idx = data.train_mask
        elif self.trainmode == 1:
            data, celltype_pred, tau = self._forward_batch(batch, hard_=False)
            idx = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]

        total_loss, loss_ce, reg_loss = self._compute_loss(celltype_pred, data.celltype, idx)

        self.train_overall_acc(preds=celltype_pred[idx], target=data.celltype[idx])
        self.train_macro_acc(preds=celltype_pred[idx], target=data.celltype[idx])

        options = {
            "on_step": self.hparams.logging.on_step,
            "on_epoch": self.hparams.logging.on_epoch,
            "prog_bar": self.hparams.logging.prog_bar,
            "logger": self.hparams.logging.logger,
            "batch_size": batch_size,
        }

        self.log("train_loss_ce", loss_ce, **options)
        self.log("reg", reg_loss, **options)
        self.log("train_overall_acc", self.train_overall_acc, **options)
        self.log("train_macro_acc", self.train_macro_acc, **options)
        self.log("tau", tau, **options)
        return total_loss

    def on_train_epoch_end(self):
        self.model.feature_selector.on_train_epoch_end(
            self.logger._root_dir, self.current_epoch
        )

    def validation_step(self, batch, batch_idx):
        batch_size = batch.input_id.size(0)
        idx = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]

        data, celltype_pred, tau = self._forward_batch(batch, hard_=True)

        total_loss, loss_ce, reg_loss = self._compute_loss(celltype_pred, batch.celltype, idx)

        self.val_overall_acc(preds=celltype_pred[idx], target=batch.celltype[idx])
        self.val_macro_acc(preds=celltype_pred[idx], target=batch.celltype[idx])

        options = {
            "on_step": self.hparams.logging.on_step,
            "on_epoch": self.hparams.logging.on_epoch,
            "prog_bar": self.hparams.logging.prog_bar,
            "logger": self.hparams.logging.logger,
            "batch_size": batch_size,
        }

        self.log("val_loss_ce", loss_ce, **options)
        self.log("reg", reg_loss, **options)
        self.log("val_overall_acc", self.val_overall_acc, **options)
        self.log("val_macro_acc", self.val_macro_acc, **options)

    def on_validation_epoch_end(self):
        pass

    def predict_step(self, batch, batch_idx):
        batch_size = batch.input_id.size(0)
        idx = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]

        data, celltype_pred, tau = self._forward_batch(batch, hard_=True)

        return [celltype_pred[idx], batch.celltype[idx]]

    def on_predict_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        batch_size = batch.input_id.size(0)
        idx = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]

        data, celltype_pred, tau = self._forward_batch(batch, hard_=True)

        total_loss, loss_ce, reg_loss = self._compute_loss(celltype_pred, batch.celltype, idx)

        self.test_overall_acc(preds=celltype_pred[idx], target=batch.celltype[idx])
        self.test_macro_acc(preds=celltype_pred[idx], target=batch.celltype[idx])
        self.test_micro_acc(preds=celltype_pred[idx], target=batch.celltype[idx])

        self.test_f1_overall(preds=celltype_pred[idx], target=batch.celltype[idx])
        self.test_f1_macro(preds=celltype_pred[idx], target=batch.celltype[idx])
        self.test_f1_micro(preds=celltype_pred[idx], target=batch.celltype[idx])

        options = {
            "on_step": self.hparams.logging.on_step,
            "on_epoch": self.hparams.logging.on_epoch,
            "prog_bar": self.hparams.logging.prog_bar,
            "logger": self.hparams.logging.logger,
            "batch_size": batch_size,
        }

        self.log("test_loss_ce", loss_ce, **options)
        self.log("reg", reg_loss, **options)
        self.log("test_overall_acc", self.test_overall_acc, **options)
        self.log("test_macro_acc", self.test_macro_acc, **options)
        self.log("test_micro_acc", self.test_micro_acc, **options)

        self.log("test_f1_overall", self.test_f1_overall, **options)
        self.log("test_f1_macro", self.test_f1_macro, **options)
        self.log("test_f1_micro", self.test_f1_micro, **options)
        self.test_pred.append(celltype_pred[idx])
        self.test_labels.append(batch.celltype[idx])

    def on_test_epoch_end(self):
        all_predictions = torch.cat(self.test_pred)
        all_labels = torch.cat(self.test_labels)

        path = self.logger._root_dir + "/test_pred.pt"
        torch.save({"predictions": all_predictions, "labels": all_labels}, path)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.trainer.lr)

        multistep_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[100],
            gamma=0.1,
        )

        if self.hparams.trainer.lr_scheduler == "multistep":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": multistep_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        elif self.hparams.trainer.lr_scheduler == "constant":
            return optimizer


def tau_schedule(type, epoch, total_epoch):
    start_tau = 10
    end_tau = 0.01

    if type == 'exp':
        tau = start_tau * (end_tau / start_tau) ** (epoch / total_epoch)
    elif type == "constant":
        tau = 0.1
    return tau
