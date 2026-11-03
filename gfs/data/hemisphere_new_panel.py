"""Graph loader for a **frozen gene panel** — with the graph still live.

This is the missing arm of the shuffle investigation. The tabular pipeline
(``gfs/data/panel_tabular.py``) removes the selector *and* the graph at once, so a low score
there cannot distinguish two hypotheses:

  a) **co-adaptation / annealing** — the shuffle GNN's 0.627 came from weights trained alongside a
     still-moving soft mask (tau 10 -> 0.01), i.e. a full-gene warm start plus per-patch Gumbel
     gene-dropout, and
  b) **noise-robustness** — training with random neighbours regularizes the node representations.
     Under ``add_self_loops=False`` the conv path is pure neighbour noise, and with ``res=true``
     each conv has a parallel ``Linear`` bypass, so this is structurally noise injection into a
     residual branch.

This module keeps the graph (and its shuffled, noisy neighbours) while removing the selector, so
it isolates (b) from (a):

| run | annealing | noisy conv | what it measures |
|---|---|---|---|
| shuffle GNN (`antelope_shuffle.py`) | yes | yes | 0.627 |
| **this, `shuffle_cells=True`** | **no** | **yes** | **noise-robustness alone** |
| this, `shuffle_cells=False` | no | no* | intact-graph frozen-panel control |
| `panel_tabular.py` (self-edges) | no | no | tabular reference (~0.57) |

\\* the intact graph is real signal, not noise -- that row is the "does the graph help at all,
given a frozen panel" control.

**Exposure needs no upsampling here.** This uses the same ``NeighborLoader`` as the graph arm, so
per-cell exposure is the graph's own ~64.84 loss terms/epoch by construction. Upsampling is only
needed when leaving the graph behind (see ``gfs/data/node_exposure.py``).

Everything else -- the shared permutation of ``gene_exp``/``celltype``, xyz and edges left
unpermuted, val/test masks -- is inherited unchanged from ``hemisphere_new_shuffle.py``. The only
additions are (1) slicing ``gene_exp`` down to a frozen panel and (2) a switch to skip the
permutation.
"""

import torch
from torch_geometric.data import Data as PyGData
from torch_geometric.loader.neighbor_loader import NeighborLoader

from gfs.data.hemisphere_new_shuffle import NeighborLoaderMod, PyGAnnData, PyGAnnDataGraphDataModule, seed_worker


class PanelGraphAnnData(PyGAnnData):
    """``PyGAnnData`` restricted to a frozen gene panel, with the cell permutation optional.

    Args:
        genes: column indices of the frozen panel, in the order the panel was reported (e.g.
            ``[2, 12, 83, 435, 480]``). ``None`` keeps all genes.
        shuffle_cells: if True, apply the shared ``gene_exp``/``celltype`` permutation exactly as
            ``hemisphere_new_shuffle.py`` does. If False, leave cells in place (intact-graph
            control).
    """

    def __init__(self, *args, genes=None, shuffle_cells=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.genes = None if genes is None else torch.as_tensor(list(genes), dtype=torch.long)
        self.shuffle_cells = shuffle_cells
        if self.genes is not None:
            n_total = self.adata_train.shape[1]
            assert int(self.genes.max()) < n_total, (
                f"gene index {int(self.genes.max())} out of range for {n_total} genes"
            )
            assert len(set(self.genes.tolist())) == self.genes.numel(), "duplicate gene indices"
            names = self.adata_train.var_names[self.genes.tolist()].tolist()
            print(f"frozen panel: columns {self.genes.tolist()} -> {names}")
        print(f"shuffle_cells={self.shuffle_cells}")

    def _build(self, adata, adj, tag, extra_fields=None):
        """Shared body of the train/valtest getters: read X, slice the panel, optionally permute."""
        gene_exp = torch.tensor(adata.X.toarray() if self.data_issparse else adata.X).float()
        if self.genes is not None:
            gene_exp = gene_exp[:, self.genes]

        celltype = torch.tensor(self.cell_type_labelencoder.transform(adata.obs[self.cell_type])).long()
        # xyz and edgelist are intentionally never permuted -- same as hemisphere_new_shuffle.py
        xyz = torch.tensor(adata.obs[self.spatial_coords].values).float()
        edgelist = self.convert_torch_sparse_coo(adj).long()
        num_cells = gene_exp.shape[0]

        if self.shuffle_cells:
            print(f"{tag} shuffle")
            generator = torch.Generator()
            generator.manual_seed(self.rand_seed)
            perm = torch.randperm(num_cells, generator=generator)
            # one shared permutation, so each cell keeps its own expression<->label pairing
            gene_exp = gene_exp[perm]
            celltype = celltype[perm]

        fields = dict(
            x=torch.cat([gene_exp, xyz], dim=1),
            edge_index=edgelist,
            celltype=celltype,
            xyz=xyz,
            num_nodes=num_cells,
            gene_exp_ind=torch.arange(gene_exp.shape[1]),
            xyz_ind=torch.arange(gene_exp.shape[1], gene_exp.shape[1] + xyz.shape[1]),
        )
        fields.update(extra_fields or {})
        return PyGData(**fields)

    def get_pygdata_obj_train(self):
        return self._build(
            self.adata_train,
            self.adj_train,
            "train",
            {"train_mask": torch.ones(self.adata_train.shape[0], dtype=torch.bool)},
        )

    def get_pygdata_obj_valtest(self):
        return self._build(
            self.adata_valtest,
            self.adj_valtest,
            "valtest",
            {
                "val_mask": torch.tensor(self.adata_valtest.obs["val_mask"].to_numpy(dtype=bool), dtype=torch.bool),
                "test_mask": torch.tensor(self.adata_valtest.obs["test_mask"].to_numpy(dtype=bool), dtype=torch.bool),
            },
        )


class PanelGraphDataModule(PyGAnnDataGraphDataModule):
    """Same loaders as the graph arm, but serving a frozen panel.

    ``num_workers`` is respected here (the parent hardcodes 8), since this module is used for
    cheap short runs as well as full ones.
    """

    def __init__(self, *args, genes=None, shuffle_cells=True, num_workers=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.genes = genes
        self.shuffle_cells = shuffle_cells
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.dataset = PanelGraphAnnData(
            path_train=self.path_train,
            path_valtest=self.path_valtest,
            spatial_coords=self.spatial_coords,
            cell_type=self.cell_type,
            self_loops_only=self.self_loops_only,
            rand_seed=self.rand_seed,
            genes=self.genes,
            shuffle_cells=self.shuffle_cells,
        )
        self.data_train = self.dataset.get_pygdata_obj_train()
        self.data_valtest = self.dataset.get_pygdata_obj_valtest()
        n_genes = len(self.data_train.gene_exp_ind)
        print(
            f"panel-graph: train {tuple(self.data_train.x.shape)} valtest {tuple(self.data_valtest.x.shape)} | "
            f"{n_genes} genes | edges train {self.data_train.edge_index.shape[1]}"
        )

    def _loader(self, data, input_nodes, shuffle):
        og = NeighborLoader(
            data,
            input_nodes=input_nodes,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
        )
        return NeighborLoaderMod(og, self.n_hops)

    def train_dataloader(self):
        return self._loader(self.data_train, self.data_train.train_mask, True)

    def val_dataloader(self):
        return self._loader(self.data_valtest, self.data_valtest.val_mask, False)

    def test_dataloader(self):
        return self._loader(self.data_valtest, self.data_valtest.test_mask, False)
