import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

from graphFeatureSelect.datamodule import AnnDataGraphDataModule
from graphFeatureSelect.models import GNN, GNN_concrete
from graphFeatureSelect.utils import get_datetime, get_paths
import numpy as np
import random
import torch

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def linear_predict(x, y, train_idx, test_idx):
    x_train = x[train_idx, :]
    x_test = x[test_idx, :]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # scale input - regularization with logistic regression is sensitive to scale.
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, solver='saga', n_jobs=-1))
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    return acc_train, acc_test


def rand_predict(x, y, train_idx, test_idx):
    x_train = x[train_idx, :]
    x_test = x[test_idx, :]
    y_train = y[train_idx]
    y_test = y[test_idx]

    strategies = ["prior", "stratified", "uniform"]
    test_scores = {}
    for s in strategies:
        dclf = DummyClassifier(strategy=s, random_state=42)
        dclf.fit(x_train, y_train)
        score = dclf.score(x_test, y_test)
        test_scores[s] = np.round(score, 2)
    return test_scores

seed = 42
setup_seeds(seed)
# data parameters, we'll eventually obtain this from the data. 

input_dim = 500
hidden_dim = 32
n_labels = 126
local_layers = 2
dropout = 0.5
heads = 8
pre_linear = True
res = True
ln = True
bn = False 
jk = True
gnn = "gat"
x_res = True # residual connection at the end
halfhop = True
xyz_status = True
model_type = "gnn"
n_mask = 5
# lr = 0.05 for concrete
lr = 0.075 # 0.01 for gnn?
n_epochs = 1000

selected_genes = [179, 167, 111, 300, 350]
# selected_genes = [184, 159, 252, 250, 1, 18, 298, 398, 2, 25, 431, 159, 265, 237,392, 87, 357, 111, 334, 64]

model_name = "modeltype_" + str(model_type) + "_nmask_" + str(n_mask) + "_lr_" + str(lr) + "_epochs_" + str(n_epochs) + "_in_" + str(input_dim) + "_h_" + str(hidden_dim) + "_nlabels_" + str(n_labels) + "_layers_" + str(local_layers)+ "_dp_" + str(dropout) + "_hd_" + str(heads) + "_preln_" + str(pre_linear) + "_res_" + str(res)+ "_ln_" + str(ln) + "_bn_" + str(bn)+ "_jk_" + str(jk)+ "_xres_" + str(x_res) + "_gnn_" + str(gnn) + "_halfhop_" + str(halfhop) + "_xyz_" + str(xyz_status)
# paths
paths = get_paths()
expname = get_datetime(expname="VISp_1slice_" + model_name + "_s_" + str(seed))
log_path = paths["runs_root"] + f"logs/{expname}"
checkpoint_path = paths["runs_root"] + f"checkpoints/{expname}"

# helpers
tb_logger = TensorBoardLogger(save_dir=log_path)
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_path, monitor="val_overall_acc", filename="{epoch}-{val_overall_acc:.2f}", mode = 'max', save_top_k=1, every_n_epochs=1 
)

# data, model and fitting
if model_type == "concrete" or model_type == "gnn":
    datamodule = AnnDataGraphDataModule(data_dir=paths["data_root"], file_names=["VISp_nhood.h5ad"], batch_size=1, n_hops=2)
elif model_type == "persist":
    datamodule = AnnDataGraphDataModule(data_dir=paths["data_root"], file_names=["VISp_nhood.h5ad"], batch_size=512, n_hops=2, no_edges=True)


if model_type == "concrete" or model_type == "persist":
    model = GNN_concrete(input_dim, hidden_dim, n_labels, n_mask, lr, 1.0, 1.0, local_layers, dropout, heads, pre_linear, res, ln, bn, jk, x_res, gnn, halfhop, xyz_status)
else:
    model = GNN(input_dim, hidden_dim, n_labels, lr, 1.0, 1.0, local_layers, dropout, heads, pre_linear, res, ln, bn, jk, x_res, gnn, halfhop, xyz_status, selected_genes)
trainer = L.Trainer(limit_train_batches=1000, limit_val_batches=100, max_epochs=n_epochs, logger=tb_logger, callbacks=[checkpoint_callback])
trainer.fit(model=model, datamodule=datamodule)

if model_type == "concrete" or model_type == "persist":

    genes = model.model.concrete_argmax()
    # predict superclass with genes
    x = datamodule.dataset.x[:, genes]
    y = datamodule.dataset.labels
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(
        skf.split(np.arange(x.shape[0]), y)
    )
    linear_acc_train, linear_acc_test = linear_predict(x, y, train_idx, test_idx)
    rand_scores = rand_predict(x, y, train_idx, test_idx)
    rand_scores = str(rand_scores)
    write_dir = log_path + "/summary.txt"
    with open(write_dir, "w") as f:
        f.write(f"Linear accuracy train: {linear_acc_train}\n")
        f.write(f"Linear accuracy test: {linear_acc_test}\n")
        f.write(f"Dummy accuracy: {rand_scores}\n")
        f.write(f"Genes: {str(genes)}\n")

    gene_dir = log_path + "/selected_genes.pt"
    torch.save(model.model.concrete_argmax(), gene_dir)


