from gfs.models.feature_selection.gumbel import GumbelFeatureSelector, ScGistFeatureSelector
from gfs.models.feature_selection.stg import STGFeatureSelector


def get_feature_selector(fs_method, **kwargs):
    if fs_method == "persist":
        return GumbelFeatureSelector(
            n_select=kwargs["n_select"],
            gene_ch=kwargs["gene_ch"],
        )
    elif fs_method == "scGist":
        return ScGistFeatureSelector(
            gene_ch=kwargs["gene_ch"],
            n_select=kwargs["n_select"],
        )
    elif fs_method == "stg":
        return STGFeatureSelector(
            input_dim=kwargs["gene_ch"],
            sigma=kwargs["sigma"],
        )
    else:
        raise ValueError(f"Unknown feature selection method: {fs_method}")
