"""Test that GnnFs constructs with each feature selector and forward pass runs."""

import torch

from gfs.models.backbone import GnnFs


def _make_dummy_inputs(n_nodes=32, gene_ch=500, spatial_ch=3):
    x = torch.randn(n_nodes, gene_ch)
    edge_index = torch.randint(0, n_nodes, (2, n_nodes * 4))
    xyz = torch.randn(n_nodes, spatial_ch)
    subgraph_id = torch.zeros(n_nodes, dtype=torch.long)
    return x, edge_index, xyz, subgraph_id


def _base_kwargs():
    return dict(
        gene_ch=500,
        spatial_ch=3,
        hid_ch=32,
        out_ch=10,
        n_select=5,
        local_layers=2,
        dropout=0.1,
        heads=1,
        pre_linear=True,
        res=True,
        ln=True,
        bn=False,
        jk=True,
        x_res=True,
        gnn="gat",
        xyz_status=True,
        focal_loss=False,
    )


def test_persist():
    model = GnnFs(**_base_kwargs(), fs_method="persist", sigma=1.0, lam=0.1)
    x, edge_index, xyz, subgraph_id = _make_dummy_inputs()
    model.train()
    out = model(x, edge_index, xyz, subgraph_id, tau=1.0, hard_=False)
    assert out.shape == (32, 10)
    model.eval()
    out = model(x, edge_index, xyz, subgraph_id, tau=1.0, hard_=True)
    assert out.shape == (32, 10)
    print("persist forward OK")


def test_scgist():
    model = GnnFs(**_base_kwargs(), fs_method="scGist", sigma=1.0, lam=0.1)
    x, edge_index, xyz, subgraph_id = _make_dummy_inputs()
    model.train()
    out = model(x, edge_index, xyz, subgraph_id, tau=1.0, hard_=False)
    assert out.shape == (32, 10)
    print("scGist forward OK")


def test_stg():
    model = GnnFs(**_base_kwargs(), fs_method="stg", sigma=0.5, lam=0.1)
    x, edge_index, xyz, subgraph_id = _make_dummy_inputs()
    model.train()
    out = model(x, edge_index, xyz, subgraph_id, tau=1.0, hard_=False)
    assert out.shape == (32, 10)
    reg = model.feature_selector.regularization_loss()
    assert reg.shape == ()
    print("stg forward OK")


def test_gnn_variants():
    for gnn_type in ["gat", "sage", "gcn"]:
        kwargs = _base_kwargs()
        kwargs["gnn"] = gnn_type
        model = GnnFs(**kwargs, fs_method="persist", sigma=1.0, lam=0.1)
        x, edge_index, xyz, subgraph_id = _make_dummy_inputs()
        model.eval()
        out = model(x, edge_index, xyz, subgraph_id, tau=1.0, hard_=True)
        assert out.shape == (32, 10)
        print(f"{gnn_type} forward OK")


if __name__ == "__main__":
    test_persist()
    test_scgist()
    test_stg()
    test_gnn_variants()
    print("All model assembly tests passed")
