# GFSNet

### Problem
Neuroscience experiments that measure cellular activity or morphology in the brain can now be followed by a profiling step to determine the transcriptomic types of the recorded cells. Due to experimental constraints, this profiling step can only reliably measure the expression of a small number of genes (~10–20). Choosing which genes to include in this panel is therefore critical: a well-chosen set enables accurate cell type assignment.

Existing computational methods select gene panels using reference single-cell RNA-seq atlases, treating each cell's expression profile independently. They do not account for the anatomical context in which the profiling is performed. Namely, the expression profiles of neighboring cells. This is a missed opportunity, because the profiling step captures gene expression not only for the cells of interest but for all surrounding cells as well.

### Approach
We leverage recent spatial transcriptomic atlases of the brain to develop a graph neural network (GNN) that selects gene panels while accounting for anatomical context. The model considers:

1. Expression profiles of the transcriptomic types of interest. 
2. Neighborhood context - expression profiles of nearby cells captured during profiling.
3. (optionally) spatial location of cell bodies within the brain.

By formulating the problem over a spatial graph, our approach naturally incorporates information from neighboring cells and is robust to segmentation noise, a known issue in current spatial atlases. We also provide theoretical guarantees that the graph-based formulation performs at least as well as context-free (tabular) selection methods. 
