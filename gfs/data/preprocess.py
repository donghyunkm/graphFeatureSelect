"""One-off preprocessing script: reads ``data/raw.h5ad``, keeps subclasses with >=5 cells, builds a
20-nearest-neighbor spatial connectivity graph from the (x_section, y_section) coordinates, stores it
in ``adata.obsp['spatial_connectivities']``, and writes ``data/processed.h5ad``."""

import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph

path = "./data/raw.h5ad"
data = ad.read_h5ad(path)


# filter >=5 for specified cell type label

celltype_label = "subclass"
label_counts = data.obs[celltype_label].value_counts()
labels_to_keep = label_counts[label_counts >= 5].index
data = data[data.obs["subclass"].isin(labels_to_keep)].copy()


xy = np.stack((data.obs["x_section"], data.obs["y_section"]), axis=1)
A = kneighbors_graph(xy, 20, mode="connectivity", include_self=False)
A = csr_matrix(A)

data.obsp["spatial_connectivities"] = A

data.write_h5ad("./data/processed.h5ad")
