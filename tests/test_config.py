"""Test that Hydra configs compose correctly for both model variants."""

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path


def test_antelope_config():
    config_dir = str(Path(__file__).parent.parent / "src" / "gfs" / "conf")
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="config", overrides=["model=antelope"])
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["model"]["fs_method"] == "persist"
        assert resolved["model"]["gene_ch"] == 500
        assert resolved["model"]["out_ch"] == 158
        print("antelope config OK")


def test_antelope_stg_config():
    config_dir = str(Path(__file__).parent.parent / "src" / "gfs" / "conf")
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="config", overrides=["model=antelope_stg"])
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["model"]["fs_method"] == "stg"
        assert resolved["model"]["sigma"] == 0.5
        print("antelope_stg config OK")


if __name__ == "__main__":
    test_antelope_config()
    test_antelope_stg_config()
    print("All config tests passed")
