import lightning as L
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError
from torchmetrics.classification import MulticlassAccuracy
from torch_geometric.nn import GATv2Conv
from graphFeatureSelect.MPNN import MPNNs
from graphFeatureSelect.MPNN_concrete import MPNN_conc
import torch_geometric.transforms as T

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes):
        super().__init__()
        # torch.manual_seed(1234567)
        self.hidden_channels = hidden_channels
        self.num_features = num_features
        self.num_classes = num_classes
        self.conv1 = GATv2Conv(self.num_features, self.hidden_channels, heads = 8, concat = False)
        self.conv2 = GATv2Conv(self.hidden_channels, self.num_classes, heads = 8, concat = False)
        self.lin1 = nn.Linear(self.num_features, self.num_classes)

        self.dropout = nn.Dropout(0.25)
    def forward(self, x, edge_index):
        residual1 = self.lin1(x)

        out = self.conv1(x, edge_index)
        
        out = out.relu()
        out = self.dropout(out)        
        out = self.conv2(out, edge_index)
        
        out = out + residual1

        return out

class GAT_xyz(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes):
        super().__init__()
        # torch.manual_seed(1234567)
        self.hidden_channels = hidden_channels
        self.num_features = num_features
        self.num_classes = num_classes
        self.conv1 = GATv2Conv(self.num_features, self.hidden_channels, heads = 8, concat = False)
        self.conv2 = GATv2Conv(self.hidden_channels, self.num_classes, heads = 8, concat = False)
        self.lin1 = nn.Linear(self.num_features, self.num_classes)

        self.dropout = nn.Dropout(0.25)
    def forward(self, x, edge_index, xyz):
        residual1 = self.lin1(x)
        out = self.conv1(x, edge_index)
        out = out.relu()
        out = self.dropout(out)        
        out = self.conv2(out, edge_index)
        out = out + residual1

        return out


class MLP_2layer(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes):
        super().__init__()
        # torch.manual_seed(1234567)
        self.hidden_channels = hidden_channels
        self.num_features = num_features
        self.num_classes = num_classes
        self.lin1 = nn.Linear(self.num_features, self.hidden_channels)
        self.lin2 = nn.Linear(self.hidden_channels, self.num_classes)
        self.dropout = nn.Dropout(0.25)
    def forward(self, x, edge_index):
        out = self.lin1(x)
        out = out.relu()
        out = self.dropout(out)        
        out = self.lin2(out)

        return out



class GNN(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, n_labels, lr, weight_mse=1.0, weight_ce=1.0, local_layers=2, 
                        dropout=0.5, heads=1, pre_linear=True, res=True, ln=True, bn=False, jk=True, x_res=True, gnn='gat', halfhop=False, xyz_status=True):
        super(GNN, self).__init__()
        self.lr = lr
        self.weight_mse = weight_mse
        self.weight_ce = weight_ce
        self.n_labels = n_labels
        self.halfhop = halfhop
        self.model = MPNNs(input_dim, hidden_dim, n_labels, local_layers, dropout, heads, pre_linear, res, ln, bn, jk, x_res, gnn, xyz_status)
        # losses
        self.loss_ce = nn.CrossEntropyLoss()

        # metrics
        self.metric_overall_acc = MulticlassAccuracy(
            num_classes=self.n_labels, top_k=1, average="weighted", multidim_average="global"
        )
        self.metric_macro_acc = MulticlassAccuracy(
            num_classes=self.n_labels, top_k=1, average="macro", multidim_average="global"
        )
        self.metric_multiclass_acc = MulticlassAccuracy(
            num_classes=self.n_labels, top_k=1, average=None, multidim_average="global"
        )

    def forward(self, x, edge_index, xyz):
        celltype = self.model(x, edge_index, xyz)
        return celltype

    def training_step(self, batch, batch_idx):
        # for GNN, batch size should be 1, and there isn't a batch dimension.
        data = batch
        # add half hop here 
        if not self.halfhop:
            gene_exp = data.x[:,:-2]
            edgelist = data.edge_index
            celltype = data.labels
            xyz = data.x[:, -2:]
            celltype_pred = self.forward(gene_exp, edgelist, xyz)
        else:
            transform = T.HalfHop(alpha=0.5)
            data = transform(data)
            gene_exp = data.x[:, :-2]
            edgelist = data.edge_index
            celltype = data.labels
            xyz = data.x[:, -2:]
            aug_mask = data.slow_node_mask
            celltype_pred = self.forward(gene_exp, edgelist, xyz)
            celltype_pred = celltype_pred[~aug_mask]

        # Calculate losses
        total_loss = self.loss_ce(celltype_pred, celltype.squeeze())

        # Log losses
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = 1)

        # Calculate metrics
        train_overall_acc = self.metric_overall_acc(preds=celltype_pred, target=celltype)
        train_macro_acc = self.metric_macro_acc(preds=celltype_pred, target=celltype)

        # Log metrics
        self.log("train_overall_acc", train_overall_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = 1)
        self.log("train_macro_acc", train_macro_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = 1)

        return total_loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        # for GNN, batch size should be 1, and there isn't a batch dimension.
        data = batch
        gene_exp = data.x[:, :-2]
        edgelist = data.edge_index
        celltype = data.labels
        xyz = data.x[:,-2:]

        celltype_pred = self.forward(gene_exp, edgelist, xyz)
        celltype_pred_max = celltype_pred.argmax(dim=1)
        # Calculate metrics
        val_overall_acc = self.metric_overall_acc(preds=celltype_pred_max, target=celltype)
        val_macro_acc = self.metric_macro_acc(preds=celltype_pred_max, target=celltype)
        val_metric_multiclass_acc = self.metric_multiclass_acc(preds=celltype_pred_max, target=celltype)

        self.log("val_overall_acc", val_overall_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = 1)
        self.log("val_macro_acc", val_macro_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = 1)

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-4)
        return optimizer
    

class GNN_concrete(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, n_labels, n_mask, lr, weight_mse=1.0, weight_ce=1.0, local_layers=2, 
                        dropout=0.5, heads=1, pre_linear=True, res=True, ln=True, bn=False, jk=True, x_res=True, gnn='gat', halfhop=False, xyz_status=True):
        super(GNN_concrete, self).__init__()
        self.lr = lr
        self.n_mask = n_mask
        self.weight_mse = weight_mse
        self.weight_ce = weight_ce
        self.n_labels = n_labels
        self.halfhop = halfhop
        self.model = MPNN_conc(input_dim, hidden_dim, n_labels, n_mask, local_layers, dropout, heads, pre_linear, res, ln, bn, jk, x_res, gnn, xyz_status)
        # losses
        self.loss_ce = nn.CrossEntropyLoss()

        # metrics
        self.metric_overall_acc = MulticlassAccuracy(
            num_classes=self.n_labels, top_k=1, average="weighted", multidim_average="global"
        )
        self.metric_macro_acc = MulticlassAccuracy(
            num_classes=self.n_labels, top_k=1, average="macro", multidim_average="global"
        )
        self.metric_multiclass_acc = MulticlassAccuracy(
            num_classes=self.n_labels, top_k=1, average=None, multidim_average="global"
        )

    def forward(self, x, edge_index, xyz, temp, hard_):
        celltype = self.model(x, edge_index, xyz, temp, hard_)
        return celltype

    def training_step(self, batch, batch_idx):
        # for GNN, batch size should be 1, and there isn't a batch dimension.
        data = batch
        epoch = self.current_epoch
        # add half hop here 
        if not self.halfhop:
            gene_exp = data.x[:,:-2]
            edgelist = data.edge_index
            celltype = data.labels
            xyz = data.x[:, -2:]
            celltype_pred = self.forward(gene_exp, edgelist, xyz, exp_decay_temp_schedule(epoch, self.trainer.max_epochs), False)
        else:
            transform = T.HalfHop(alpha=0.5)
            data = transform(data)
            gene_exp = data.x[:, :-2]
            edgelist = data.edge_index
            celltype = data.labels
            xyz = data.x[:, -2:]
            aug_mask = data.slow_node_mask
            celltype_pred = self.forward(gene_exp, edgelist, xyz, exp_decay_temp_schedule(epoch, self.trainer.max_epochs), False)
            celltype_pred = celltype_pred[~aug_mask]

        # Calculate losses
        total_loss = self.loss_ce(celltype_pred, celltype.squeeze())

        # Log losses
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = 1)

        # Calculate metrics
        train_overall_acc = self.metric_overall_acc(preds=celltype_pred, target=celltype)
        train_macro_acc = self.metric_macro_acc(preds=celltype_pred, target=celltype)

        # Log metrics
        self.log("train_overall_acc", train_overall_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = 1)
        self.log("train_macro_acc", train_macro_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = 1)

        return total_loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        # for GNN, batch size should be 1, and there isn't a batch dimension.
        data = batch
        gene_exp = data.x[:, :-2]
        edgelist = data.edge_index
        celltype = data.labels
        xyz = data.x[:,-2:]
        celltype_pred = self.forward(gene_exp, edgelist, xyz, 0.01, True)
        celltype_pred_max = celltype_pred.argmax(dim=1)
        # Calculate metrics
        val_overall_acc = self.metric_overall_acc(preds=celltype_pred_max, target=celltype)
        val_macro_acc = self.metric_macro_acc(preds=celltype_pred_max, target=celltype)
        val_metric_multiclass_acc = self.metric_multiclass_acc(preds=celltype_pred_max, target=celltype)

        self.log("val_overall_acc", val_overall_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = 1)
        self.log("val_macro_acc", val_macro_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = 1)

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-4)
        return optimizer

def exp_decay_temp_schedule(epoch, total_epoch):
    start_temp = 10
    end_temp = 0.01
    temp = start_temp * (end_temp / start_temp) ** (epoch / total_epoch)
    return temp
