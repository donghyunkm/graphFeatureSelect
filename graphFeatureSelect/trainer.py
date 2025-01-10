import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

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


seed = 42
model_type = "gat"
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
concrete = True
n_mask = 10
lr = 0.05
n_epochs = 20
model_name = "conc_" + str(concrete) + "_nmask_" + str(n_mask) + "_lr_" + str(lr) + "_epochs_" + str(n_epochs) + "_in_" + str(input_dim) + "_h_" + str(hidden_dim) + "_nlabels_" + str(n_labels) + "_layers_" + str(local_layers)+ "_dp_" + str(dropout) + "_hd_" + str(heads) + "_preln_" + str(pre_linear) + "_res_" + str(res)+ "_ln_" + str(ln) + "_bn_" + str(bn)+ "_jk_" + str(jk)+ "_xres_" + str(x_res) + "_gnn_" + str(gnn) + "_halfhop_" + str(halfhop) + "_xyz_" + str(xyz_status)
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
datamodule = AnnDataGraphDataModule(data_dir=paths["data_root"], file_names=["VISp_nhood.h5ad"], batch_size=1, n_hops=2)
if concrete:
    model = GNN_concrete(input_dim, hidden_dim, n_labels, n_mask, lr, 1.0, 1.0, local_layers, dropout, heads, pre_linear, res, ln, bn, jk, x_res, gnn, halfhop, xyz_status)
else:
    model = GNN(input_dim, hidden_dim, n_labels, lr, 1.0, 1.0, local_layers, dropout, heads, pre_linear, res, ln, bn, jk, x_res, gnn, halfhop, xyz_status)
trainer = L.Trainer(limit_train_batches=1000, limit_val_batches=100, max_epochs=n_epochs, logger=tb_logger, callbacks=[checkpoint_callback])
trainer.fit(model=model, datamodule=datamodule)

if concrete:
    gene_dir = log_path + "/selected_genes.txt"
    with open(gene_dir, "w") as f:
        f.write(str(model.model.concrete_argmax()))
