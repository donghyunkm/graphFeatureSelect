from .base import FeatureSelector
from .gumbel import GumbelFeatureSelector
from .scgist import ScGistFeatureSelector
from .stg import STGFeatureSelector


def get_feature_selector(fs_method: str, n_genes: int, n_select: int, **kwargs) -> FeatureSelector:
    if fs_method == "gumbel":
        return GumbelFeatureSelector(n_genes=n_genes, n_select=n_select)
    elif fs_method == "scgist":
        return ScGistFeatureSelector(
            n_genes=n_genes, n_select=n_select, l1=kwargs.get("l1", 0.1)
        )
    elif fs_method == "stg":
        return STGFeatureSelector(
            n_genes=n_genes, n_select=n_select, sigma=kwargs.get("sigma", 0.5)
        )
    else:
        raise ValueError(f"Unknown feature selection method: {fs_method}")
