### Overview

 - Neuroscientists routinely perform experiments to measure cellular activity or morphology in the brain.
 - It is now possible to perform a profiling step on the same brain tissue to determine transcriptomic types for cells involved in such experiments. 
 - While fluorescence labeling can identify the cells of interest and provide anatomical location of the cell bodies, a measurement of gene expression is required for transcriptomic cell type assignment.
 - Due to experimental constraints, the expression of only a small (e.g. 10-20) number of genes can be measured in the profiling step.
 - Here we develop a feature selection approach that uses anatomical context available in spatial transcriptomics atlases to improve gene panel selection for such profiling experiments. 


### Environment

```bash
conda create -n gfs python=3.12
conda activate gfs
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install torch_geometric
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu124/pyg_lib-0.4.0%2Bpt25cu124-cp312-cp312-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu124/torch_cluster-1.6.3%2Bpt25cu124-cp312-cp312-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu124/torch_scatter-2.1.2%2Bpt25cu124-cp312-cp312-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu124/torch_sparse-0.6.18%2Bpt25cu124-cp312-cp312-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu124/torch_spline_conv-1.2.2%2Bpt25cu124-cp312-cp312-linux_x86_64.whl
 
pip install lightning scikit-learn jupyterlab tensorboard hydra-core
pip install rich tqdm seaborn
pip install -e .[dev]
```
