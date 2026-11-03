"""Shuffle control with the permutation restricted to *within* each evaluation split.

Same negative control as ``hemisphere_new_shuffle.py`` -- one shared permutation of
``gene_exp``/``celltype`` with ``xyz`` and the edgelist left intact, so each biological cell lands
on a random spatial node while keeping its own expression<->label pairing. The **only** difference
is the valtest permutation:

| | ``hemisphere_new_shuffle.py`` | this module |
|---|---|---|
| train permutation | full, over all 74,633 train cells | **inherited, byte-identical** |
| valtest permutation | full, over all 63,124 valtest cells | **block-diagonal**: val cells -> val nodes, test cells -> test nodes |
| ``val_mask`` / ``test_mask`` | never permuted | never permuted (unchanged) |
| ``xyz``, ``edge_index`` | never permuted | never permuted (unchanged) |

**Why.** The parent's full valtest permutation is a bijection over a set that val and test
*partition*, so it never duplicates or drops a cell -- val nodes get 31,562 distinct cells and test
nodes get exactly the complement. It is a re-partition, not a leak. But roughly half of each set
swaps sides, so "the val split" and "the test split" are not the same cell sets they are in every
other arm of the project. Restricting the permutation to each block keeps the *cell membership* of
val and test exactly as the h5ad defines it, while still destroying the spatial arrangement inside
each -- which is the thing the control is meant to ablate.

Because val and test are statistically interchangeable in this dataset (subclass distributions
agree to 4e-4 L1, xyz means to 3 decimals), this is not expected to move the metrics much. It makes
the comparison exact rather than approximate.

Train needs no special handling: ``train_mask`` is all-ones on the train graph, so the parent's
full permutation is already "within train". ``get_pygdata_obj_train`` is therefore inherited
unchanged, and so are all three dataloaders -- including the hardcoded ``num_workers=8`` and
generator seed 42 -- so patch composition matches the completed shuffle runs exactly.
"""

import torch
from torch_geometric.data import Data as PyGData

from gfs.data.hemisphere_new_shuffle import PyGAnnData
from gfs.data.hemisphere_new_shuffle import PyGAnnDataGraphDataModule as ShuffleDataModule


class WithinSplitShuffleAnnData(PyGAnnData):
    """``PyGAnnData`` whose valtest permutation is block-diagonal over val / test / neither."""

    def _within_split_perm(self, val_mask, test_mask, num_cells):
        """Build a permutation that maps each block's nodes onto that same block's cells.

        Returns ``perm`` with ``perm[i]`` = index of the cell placed at node ``i``, guaranteeing
        ``val_mask[perm[i]] == val_mask[i]`` and likewise for test.
        """
        generator = torch.Generator()
        generator.manual_seed(self.rand_seed)

        perm = torch.empty(num_cells, dtype=torch.long)
        blocks = {
            "val": torch.where(val_mask)[0],
            "test": torch.where(test_mask)[0],
            # defensive: nodes in neither split are permuted among themselves rather than left in
            # place. The current h5ad has none (val | test covers all 63,124), but a future split
            # with held-out nodes must not silently leak across blocks.
            "other": torch.where(~val_mask & ~test_mask)[0],
        }
        for name, idx in blocks.items():
            n = idx.numel()
            if n == 0:
                continue
            perm[idx] = idx[torch.randperm(n, generator=generator)]
            print(f"valtest shuffle within {name}: {n} cells")

        assert torch.equal(perm.sort().values, torch.arange(num_cells)), "perm is not a bijection"
        assert torch.equal(val_mask[perm], val_mask), "permutation crossed the val boundary"
        assert torch.equal(test_mask[perm], test_mask), "permutation crossed the test boundary"
        return perm

    def get_pygdata_obj_valtest(self):
        """As the parent, but permuting cells only within val and within test.

        ``xyz``, ``edge_index``, ``val_mask`` and ``test_mask`` are left untouched.
        """
        if self.data_issparse:
            gene_exp = torch.tensor(self.adata_valtest.X.toarray()).float()
        else:
            gene_exp = torch.tensor(self.adata_valtest.X).float()

        celltype = torch.tensor(self.cell_type_labelencoder.transform(self.adata_valtest.obs[self.cell_type])).long()

        # xyz and edgelist are intentionally not shuffled -- same as hemisphere_new_shuffle.py
        xyz = torch.tensor(self.adata_valtest.obs[self.spatial_coords].values).float()
        edgelist = self.convert_torch_sparse_coo(self.adj_valtest).long()

        # masks are read from the h5ad and never permuted, so val/test stay on the same nodes
        val_mask = torch.tensor(self.adata_valtest.obs["val_mask"].to_numpy(dtype=bool), dtype=torch.bool)
        test_mask = torch.tensor(self.adata_valtest.obs["test_mask"].to_numpy(dtype=bool), dtype=torch.bool)

        num_cells = gene_exp.shape[0]
        perm = self._within_split_perm(val_mask, test_mask, num_cells)

        # one shared permutation, so each cell keeps its own expression<->label pairing
        gene_exp = gene_exp[perm]
        celltype = celltype[perm]

        x = torch.cat([gene_exp, xyz], dim=1)

        return PyGData(
            x=x,
            edge_index=edgelist,
            celltype=celltype,
            xyz=xyz,
            num_nodes=num_cells,
            gene_exp_ind=torch.arange(gene_exp.shape[1]),
            xyz_ind=torch.arange(gene_exp.shape[1], gene_exp.shape[1] + xyz.shape[1]),
            val_mask=val_mask,
            test_mask=test_mask,
        )


class PyGAnnDataGraphDataModule(ShuffleDataModule):
    """Identical to the shuffle datamodule except for the valtest permutation.

    Keeps the parent's ``train_dataloader`` / ``val_dataloader`` / ``test_dataloader`` verbatim
    (``num_workers=8``, generator seed 42), so patch composition is unchanged from the completed
    shuffle runs.
    """

    def setup(self, stage: str):
        self.dataset = WithinSplitShuffleAnnData(
            path_train=self.path_train,
            path_valtest=self.path_valtest,
            spatial_coords=self.spatial_coords,
            cell_type=self.cell_type,
            self_loops_only=self.self_loops_only,
            rand_seed=self.rand_seed,
        )
        self.data_train = self.dataset.get_pygdata_obj_train()
        self.data_valtest = self.dataset.get_pygdata_obj_valtest()
