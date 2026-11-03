"""gfs — graph-based gene-panel feature selection for spatial transcriptomics.

Train a GNN to classify cell subclasses from gene expression + a spatial neighbor graph, with a
learnable feature-selection layer, to pick a small (~10-20 gene) panel. Subpackages: ``models``
(GNN + selectors), ``data`` (data modules), ``trainers`` (Hydra+Lightning entrypoints),
``analysis`` (metric/significance recomputation)."""
