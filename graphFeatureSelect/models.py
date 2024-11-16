import lightning as L
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError
from torchmetrics.classification import MulticlassAccuracy
from torch_geometric.nn import GATv2Conv

class GAT3(torch.nn.Module):
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

class GAT3_xyz(torch.nn.Module):
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
    def __init__(self, input_dim, hidden_dim, n_labels, weight_mse=1.0, weight_ce=1.0, model_type = "GAT3"):
        super(GNN, self).__init__()

        self.weight_mse = weight_mse
        self.weight_ce = weight_ce
        self.n_labels = n_labels
        if model_type == "GAT3":
            self.model = GAT3(hidden_channels=32, num_features = input_dim, num_classes=self.n_labels)
        elif model_type == "MLP_2layer":
            self.model = MLP_2layer(hidden_channels=32, num_features = input_dim, num_classes=self.n_labels)
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

    def forward(self, x, edge_index):
        celltype = self.model(x, edge_index)
        return celltype

    def training_step(self, batch, batch_idx):
        # for GNN, batch size should be 1, and there isn't a batch dimension.
        gene_exp, edgelist, celltype = batch
        gene_exp = gene_exp.squeeze(dim=0)
        edgelist = edgelist.squeeze(dim=0).T
        celltype = celltype.squeeze(dim=0)
        celltype_pred = self.forward(gene_exp, edgelist)

        # Calculate losses
        total_loss = self.loss_ce(celltype_pred, celltype.squeeze())

        # Log losses
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Calculate metrics
        train_overall_acc = self.metric_overall_acc(preds=celltype_pred, target=celltype)
        train_macro_acc = self.metric_macro_acc(preds=celltype_pred, target=celltype)

        # Log metrics
        self.log("train_overall_acc", train_overall_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_macro_acc", train_macro_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        # for GNN, batch size should be 1, and there isn't a batch dimension.
        gene_exp, edgelist, celltype = batch
        gene_exp = gene_exp.squeeze(dim=0)
        edgelist = edgelist.squeeze(dim=0).T
        celltype = celltype.squeeze(dim=0)

        celltype_pred = self.forward(gene_exp, edgelist)
        celltype_pred_max = celltype_pred.argmax(dim=1)
        # Calculate metrics
        val_overall_acc = self.metric_overall_acc(preds=celltype_pred_max, target=celltype)
        val_macro_acc = self.metric_macro_acc(preds=celltype_pred_max, target=celltype)
        val_metric_multiclass_acc = self.metric_multiclass_acc(preds=celltype_pred_max, target=celltype)

        self.log("val_overall_acc", val_overall_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_macro_acc", val_macro_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        return optimizer