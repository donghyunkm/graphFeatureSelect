"""4-hop (complete-receptive-field) loader for a **frozen gene panel** — no selector.

Feature-ablation arm: the 4-hop GFS model (`docs/20260820_deep_sample_shallow_loss.md`) trained
*without* the selection module (`fs_method="none"`) on an already-chosen gene panel, so it isolates
the value of the learned feature selection from the value of the 4-hop complete-field training.

This reuses everything from ``hemisphere_new_4hop.py`` — same 4-hop sampling, same ``batch.loss_mask``
(so ``trainmode=2`` works unchanged), same root-only val/test — and adds exactly one thing: it slices
the AnnData down to the frozen panel before building the PyG graph, so the model's input is the panel
itself (``gene_exp_ind = arange(len(genes))``, ``gene_ch = len(genes)``). No permutation, no upsampling.

See ``gfs/trainers/antelope_4hop_panel.py`` and ``docs/feature_ablation.md``.
"""

from scipy.sparse import issparse

from gfs.data.hemisphere_new_4hop import PyGAnnData, PyGAnnDataGraphDataModule


class Panel4hopAnnData(PyGAnnData):
    """``PyGAnnData`` (4-hop variant) restricted to a frozen gene panel.

    Args:
        genes: column indices of the frozen panel, in report order (e.g. ``[2, 250, 435, 480, 481]``),
            0-indexed into ``adata.var`` (AnnData default). ``None`` keeps all genes.
    """

    def __init__(self, *args, genes=None, **kwargs):
        super().__init__(*args, **kwargs)
        if genes is not None:
            genes = list(genes)
            n_total = self.adata_train.shape[1]
            assert max(genes) < n_total, f"gene index {max(genes)} out of range for {n_total} genes"
            assert len(set(genes)) == len(genes), "duplicate gene indices"
            names = self.adata_train.var_names[genes].tolist()
            # slice both splits to the panel; obsp (adjacency) and obs (labels/xyz) are untouched,
            # so adj_train/adj_valtest and the label encoder built in super().__init__ stay valid.
            self.adata_train = self.adata_train[:, genes].copy()
            self.adata_valtest = self.adata_valtest[:, genes].copy()
            self.data_issparse = issparse(self.adata_train.X)
            print(f"frozen 4hop panel: columns {genes} -> {names}")


class Panel4hopDataModule(PyGAnnDataGraphDataModule):
    """Same 4-hop loaders as ``antelope_4hop.py``, but serving a frozen panel."""

    def __init__(self, *args, genes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.genes = genes

    def setup(self, stage: str):
        self.dataset = Panel4hopAnnData(
            path_train=self.path_train,
            path_valtest=self.path_valtest,
            spatial_coords=self.spatial_coords,
            cell_type=self.cell_type,
            self_loops_only=self.self_loops_only,
            rand_seed=self.rand_seed,
            genes=self.genes,
        )
        self.data_train = self.dataset.get_pygdata_obj_train()
        self.data_valtest = self.dataset.get_pygdata_obj_valtest()
        n_genes = len(self.data_train.gene_exp_ind)
        print(
            f"4hop-panel: train {tuple(self.data_train.x.shape)} valtest {tuple(self.data_valtest.x.shape)} | "
            f"{n_genes} genes | n_hops {self.n_hops} loss_hops {self.loss_hops}"
        )
