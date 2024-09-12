import anndata as ad
import numpy as np
def get_adata():
    adata = ad.read_h5ad('../data/VISp.h5ad')
    adata.obsm["ccf"] = np.concatenate(
        (
            np.expand_dims(np.array(adata.obs["x_ccf"]), axis=1),
            np.expand_dims(np.array(adata.obs["y_ccf"]), axis=1),
            np.expand_dims(np.array(adata.obs["z_ccf"]), axis=1),
        ),
        axis=1,
    )
    adata.var.set_index("gene_symbol", inplace=True, drop=False)

    return adata