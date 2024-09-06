Neuroscientists routinely perform experiments to measure cellular activity or morphology in the brain.
It is now possible to perform a profiling step on the same brain tissue to determine transcriptomic types for cells involved in such experiments. 
While fluorescence labeling can identify the cells of interest and provide anatomical location of the cell bodies, a measurement of gene expression is required for transcriptomic cell type assignment.
Due to experimental constraints, the expression of only a small ($\sim$10-20) set of genes can be reliably measured in the profiling step.
While many computational methods exist to select such gene sets based on reference single-cell RNAseq atlases, they do not consider anatomical context.
Here we leverage recent spatial atlases of the brain to develop a graph neural network that selects optimal gene sets.
Our model accounts not only for expression profiles of the transcriptomic types of interest, but also anatomical context that includes expression profiles of near by cells and location of cell bodies in standardized co-ordinate space.
