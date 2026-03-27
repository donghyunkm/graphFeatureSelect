"""Test that all key modules are importable."""


def test_imports():
    from gfs.data.datamodule import HemisphereDataModule
    from gfs.data.dataset import PyGAnnData
    from gfs.models.backbone import GNNBackbone
    from gfs.models.extras import HalfHop
    from gfs.models.feature_selection import get_feature_selector
    from gfs.models.feature_selection.gumbel import GumbelFeatureSelector
    from gfs.models.feature_selection.scgist import ScGistFeatureSelector
    from gfs.models.feature_selection.stg import STGFeatureSelector
    from gfs.models.lit_module import LitGnnFs
    from gfs.trainers.train import main

    print("All imports OK")


if __name__ == "__main__":
    test_imports()
