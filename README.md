### Overview

 - Neuroscientists routinely perform experiments to measure cellular activity or morphology in the brain.
 - It is now possible to perform a profiling step on the same brain tissue to determine transcriptomic types for cells involved in such experiments. 
 - While fluorescence labeling can identify the cells of interest and provide anatomical location of the cell bodies, a measurement of gene expression is required for transcriptomic cell type assignment.
 - Due to experimental constraints, the expression of only a small (e.g. 10-20) number of genes can be measured in the profiling step.
 - Here we develop a feature selection approach that uses anatomical context available in spatial transcriptomics atlases to improve gene panel selection for such profiling experiments. 


### Environment

 - Use conda to create the virtual enviroment.
 - Install `pytorch`, `pyg`, and `captum`. 
 - Finally use `pip install -e .` to install this package.