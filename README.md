# GFSNet

GFSNet learns optimal gene panels for spatial transcriptomics using graph neural networks with differentiable feature selection. The system helps neuroscientists select which genes to measure (typically 10-50) in post-hoc profiling experiments.

The broad goal is to go over different variants of feature selection layers and graph neural network implementations for the classification task.

The main entry points for the current scope are:
`trainers/antelope.py`
`trainers/antelope_stg.py`

# Notes on graph construction

Use this to construct the KNN graph on which the GNN operates. A is the adjacency matrix.
The data is actually only 2d (per section). Once the user tells you what the within-section co-ordinates are:

```python
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix

k = 20
xy = np.stack((data.obs['x_section'], data.obs['y_section']), axis=1)
A = kneighbors_graph(xy, k, mode='connectivity', include_self=False)
A = csr_matrix(A)
data.obsp['spatial_connectivities'] = A
```