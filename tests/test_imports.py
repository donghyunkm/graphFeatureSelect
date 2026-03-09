"""Test that all key modules are importable."""


def test_imports():
    from gfs.models.lit_module import LitGnnFs
    from gfs.models.backbone import GnnFs
    from gfs.models.components import MLP, FeatureRegularizer
    from gfs.models.feature_selection import get_feature_selector
    from gfs.models.feature_selection.gumbel import GumbelFeatureSelector, ScGistFeatureSelector
    from gfs.models.feature_selection.stg import STGFeatureSelector
    from gfs.models.transforms import HalfHop
    from gfs.data.hemisphere import PyGAnnDataGraphDataModule, PyGAnnData
    from gfs.utils import get_datetime, get_paths
    from gfs.trainers.train import main
    print("All imports OK")


if __name__ == "__main__":
    test_imports()
