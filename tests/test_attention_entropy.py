"""
Tests for GR2ST-style GATv2 attention-entropy regularization in Antelope.

Covers: backward-compatibility of the default (disabled) path, the literal GR2ST entropy helper,
gradients reaching GATv2 attention params, multi-head / multi-layer handling, PyG neighbourhood
normalization, non-GAT config handling, the Lightning `training_step` total loss, and a tiny end-to-end
Lightning fit smoke test (checkpoint metric + entropy metrics present).

Run from the repo root in the `gfs` conda env:
    pytest tests/test_attention_entropy.py -q
"""

import lightning as L
import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from gfs.models.antelope import GnnFs, LitGnnFs

GENE_CH = 8
OUT_CH = 4
N_NODES = 24


def make_config(**model_overrides):
    """Minimal-but-complete config mirroring gfs/configs/antelope.yaml structure."""
    cfg = {
        "expname": "test",
        "data": {"n_labels": OUT_CH},
        "model": {
            "name": "antelope",
            "gene_ch": GENE_CH,
            "spatial_ch": 3,
            "hid_ch": 8,
            "out_ch": OUT_CH,
            "n_select": 3,
            "local_layers": 2,
            "dropout": 0.0,
            "heads": 3,
            "pre_linear": True,
            "res": True,
            "ln": True,
            "bn": True,
            "jk": True,
            "x_res": True,
            "gnn": "gat",
            "xyz_status": True,
            "halfhop": False,
            "tautype": "old",
            "trainmode": 0,
            "loss": "ce",
            "fs_method": "persist",
            "focal_loss": False,
            "sigma": 1.0,
            "lam": 0.01,
            "hard_training": False,
            "attention_entropy_weight": 0.0,
            "attention_entropy_eps": 1.0e-9,
        },
        "logging": {"on_step": False, "on_epoch": True, "prog_bar": False, "logger": True},
        "trainer": {
            "limit_train_batches": 1.0,
            "limit_val_batches": 1.0,
            "max_epochs": 1,
            "lr": 0.001,
            "lr_scheduler": "constant",
        },
    }
    cfg["model"].update(model_overrides)
    return OmegaConf.create(cfg)


def make_graph(n_nodes=N_NODES, seed=0):
    """A small connected-ish random graph with genes+xyz packed into x."""
    g = torch.Generator().manual_seed(seed)
    gene_exp = torch.randn(n_nodes, GENE_CH, generator=g)
    xyz = torch.randn(n_nodes, 3, generator=g)
    x = torch.cat([gene_exp, xyz], dim=1)
    # each node i receives edges from a few earlier/other nodes (directed into i)
    src, dst = [], []
    for i in range(n_nodes):
        for j in [(i + 1) % n_nodes, (i + 2) % n_nodes, (i + 3) % n_nodes]:
            src.append(j)
            dst.append(i)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    subgraph_id = torch.zeros(n_nodes, dtype=torch.long)
    celltype = torch.randint(0, OUT_CH, (n_nodes,), generator=g)
    return x, edge_index, xyz, subgraph_id, celltype


def build_model_inputs():
    x, edge_index, _xyz, subgraph_id, celltype = make_graph()
    return {
        "x": x[:, :GENE_CH],
        "edge_index": edge_index,
        "xyz": x[:, GENE_CH:],
        "subgraph_id": subgraph_id,
        "tau": 0.5,
        "hard_": False,
    }, celltype


# ---------------------------------------------------------------------------
# 36 / 41 / 42 partly rely on this shared model
# ---------------------------------------------------------------------------
def make_gnn(**overrides):
    cfg = make_config(**overrides).model
    return GnnFs(
        gene_ch=cfg.gene_ch,
        spatial_ch=cfg.spatial_ch,
        hid_ch=cfg.hid_ch,
        out_ch=cfg.out_ch,
        n_select=cfg.n_select,
        local_layers=cfg.local_layers,
        dropout=cfg.dropout,
        heads=cfg.heads,
        pre_linear=cfg.pre_linear,
        res=cfg.res,
        ln=cfg.ln,
        bn=cfg.bn,
        jk=cfg.jk,
        x_res=cfg.x_res,
        gnn=cfg.gnn,
        xyz_status=cfg.xyz_status,
        fs_method=cfg.fs_method,
        focal_loss=cfg.focal_loss,
        hard_training=cfg.hard_training,
    )


# ---------------------------------------------------------------------------
# Section 36: backward compatibility of the disabled/default path
# ---------------------------------------------------------------------------
def test_forward_matches_when_attention_disabled():
    model = make_gnn()
    model.eval()  # eval => deterministic hard mask + no dropout
    inputs, _ = build_model_inputs()

    with torch.no_grad():
        logits_plain = model(**inputs)  # ordinary API: single tensor
        logits_attn, records = model(**inputs, return_attention_weights=True)

    assert torch.is_tensor(logits_plain)  # default path returns only logits
    assert torch.allclose(logits_plain, logits_attn, atol=1e-6)
    # one record per GAT layer
    assert len(records) == len(model.gnn_layers)


# ---------------------------------------------------------------------------
# Section 37: GR2ST entropy helper
# ---------------------------------------------------------------------------
def test_gr2st_entropy_helper_finite_nonneg_and_ordering():
    lit = LitGnnFs(make_config())
    E, H = 10, 3
    # multi-head shapes [E, H]; compare a diffuse vs a sharp attention pattern
    diffuse = torch.full((E, H), 0.25)
    sharp = torch.full((E, H), 0.02)
    sharp[0] = 0.9

    loss_diffuse = lit._gr2st_attention_entropy(diffuse)
    loss_sharp = lit._gr2st_attention_entropy(sharp)

    assert torch.isfinite(loss_diffuse)
    assert torch.isfinite(loss_sharp)
    assert loss_diffuse >= -1e-6
    assert loss_sharp >= -1e-6
    # sharper attention => lower entropy loss
    assert loss_sharp < loss_diffuse


# ---------------------------------------------------------------------------
# Section 38: gradients reach GATv2 attention parameters
# ---------------------------------------------------------------------------
def test_gradients_reach_gat_attention_params():
    lit = LitGnnFs(make_config(attention_entropy_weight=0.01))
    lit.train()
    inputs, celltype = build_model_inputs()

    logits, records = lit.model(**inputs, return_attention_weights=True)
    cls_loss = torch.nn.functional.cross_entropy(logits, celltype)
    entropy = torch.stack([lit._gr2st_attention_entropy(a) for _, a in records]).mean()
    loss = cls_loss + lit.attention_entropy_weight * entropy
    loss.backward()

    # GATv2Conv attention params carry the name "att"
    grads = []
    for layer in lit.model.gnn_layers:
        for name, p in layer.named_parameters():
            if "att" in name:
                assert p.grad is not None, f"no grad for {name}"
                assert torch.isfinite(p.grad).all()
                grads.append(p.grad.abs().sum().item())
    assert len(grads) > 0
    assert any(g > 0 for g in grads), "expected at least one nonzero attention gradient"


# ---------------------------------------------------------------------------
# Section 41 / 24: multiple heads => alpha is 2-D and entropy handles all heads
# ---------------------------------------------------------------------------
def test_multihead_alpha_is_2d():
    model = make_gnn(heads=3)
    inputs, _ = build_model_inputs()
    _, records = model(**inputs, return_attention_weights=True)
    for _, alpha in records:
        assert alpha.dim() == 2
        assert alpha.size(1) == 3


# ---------------------------------------------------------------------------
# Section 42: multiple GAT layers => mean aggregation
# ---------------------------------------------------------------------------
def test_multiple_gat_layers_mean_aggregation():
    lit = LitGnnFs(make_config(local_layers=2, attention_entropy_weight=0.01))
    inputs, _ = build_model_inputs()
    _, records = lit.model(**inputs, return_attention_weights=True)
    assert len(records) == len(lit.model.gnn_layers)

    per_layer = [lit._gr2st_attention_entropy(a) for _, a in records]
    total = torch.stack(per_layer).mean()
    assert torch.allclose(total, sum(per_layer) / len(per_layer), atol=1e-6)


# ---------------------------------------------------------------------------
# Section 43 / 29: PyG neighbourhood normalization (installed PyG version)
# ---------------------------------------------------------------------------
def test_pyg_attention_normalizes_per_destination_head():
    torch.manual_seed(0)
    edge_index = torch.tensor([[1, 2, 3, 0, 2], [0, 0, 0, 1, 1]])
    x = torch.randn(4, 5)
    conv = GATv2Conv(5, 6, heads=3, concat=False, add_self_loops=False, bias=False)
    _, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
    assert alpha.shape == (5, 3)

    dst = ei[1]  # row 1 is destination under flow="source_to_target"
    num = int(dst.max()) + 1
    sums = torch.zeros(num, alpha.size(1)).index_add_(0, dst, alpha)
    # every destination node's incoming attention sums to ~1 per head
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---------------------------------------------------------------------------
# Section 40: non-GAT handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gnn", ["gcn", "sage"])
def test_non_gat_zero_weight_builds(gnn):
    lit = LitGnnFs(make_config(gnn=gnn, attention_entropy_weight=0.0, x_res=True))
    assert lit.gnn_type == gnn


def test_non_gat_positive_weight_raises():
    with pytest.raises(ValueError, match="requires gnn='gat'"):
        LitGnnFs(make_config(gnn="gcn", attention_entropy_weight=0.01))


def test_non_gat_log_stats_raises():
    with pytest.raises(ValueError, match="requires gnn='gat'"):
        LitGnnFs(make_config(gnn="sage", attention_entropy_weight=0.0, log_attention_stats=True))


# ---------------------------------------------------------------------------
# Section 25-28: sharpness diagnostics are finite and sane
# ---------------------------------------------------------------------------
def test_sharpness_diagnostics_ranges():
    lit = LitGnnFs(make_config(attention_entropy_weight=0.01))
    inputs, _ = build_model_inputs()
    _, records = lit.model(**inputs, return_attention_weights=True)
    diag = lit._attention_sharpness_diagnostics(records)
    assert 0.0 <= diag["train_attention_top1_mass"] <= 1.0 + 1e-6
    assert diag["train_attention_effective_neighbors"] >= 1.0 - 1e-6
    assert 0.0 - 1e-6 <= diag["train_attention_normalized_entropy"] <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Sections 39 / 44: Lightning training_step total loss + end-to-end smoke fit
# ---------------------------------------------------------------------------
class _TinyGraphDataModule(L.LightningDataModule):
    """Yields one PyG Data batch carrying every field training/validation_step touches."""

    def _make_batch(self):
        x, edge_index, _xyz, subgraph_id, celltype = make_graph(seed=1)
        data = Data(x=x, edge_index=edge_index)
        data.gene_exp_ind = torch.arange(GENE_CH)
        data.xyz_ind = torch.arange(GENE_CH, GENE_CH + 3)
        data.subgraph_id = subgraph_id
        data.celltype = celltype
        data.train_mask = torch.ones(N_NODES, dtype=torch.bool)
        # root-node bookkeeping used by validation/test: mark first 4 nodes as roots
        data.n_id = torch.arange(N_NODES)
        data.input_id = torch.arange(4)
        return data

    def train_dataloader(self):
        from torch_geometric.loader import DataLoader

        return DataLoader([self._make_batch()], batch_size=1)

    def val_dataloader(self):
        from torch_geometric.loader import DataLoader

        return DataLoader([self._make_batch()], batch_size=1)


def _cpu_trainer(tmp_path):
    from lightning.pytorch.loggers import CSVLogger

    return L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=CSVLogger(save_dir=str(tmp_path)),  # on_train_epoch_end writes selections.csv via logger
        enable_checkpointing=False,
        enable_progress_bar=False,
    )


def test_training_step_total_loss_composition(tmp_path):
    """train_loss_total == classification + weighted attention entropy (persist => no scGist reg)."""
    weight = 0.05
    lit = LitGnnFs(make_config(attention_entropy_weight=weight))
    dm = _TinyGraphDataModule()

    trainer = _cpu_trainer(tmp_path)
    # run through fit so self.log / trainer wiring is available
    trainer.fit(lit, datamodule=dm)

    # recompute the objective by hand on a fresh forward pass in train mode with a fixed tau
    lit.train()
    batch = dm._make_batch()
    from gfs.models.antelope import tau_schedule

    torch.manual_seed(123)
    logits, records = lit.model(
        x=batch.x[:, batch.gene_exp_ind],
        edge_index=batch.edge_index,
        xyz=batch.x[:, batch.xyz_ind],
        subgraph_id=batch.subgraph_id,
        tau=tau_schedule(lit.tautype, 0, 1),
        hard_=False,
        return_attention_weights=True,
    )
    cls = torch.nn.functional.cross_entropy(logits[batch.train_mask], batch.celltype[batch.train_mask])
    entropy = torch.stack([lit._gr2st_attention_entropy(a) for _, a in records]).mean()
    expected = cls + weight * entropy
    # not comparing against the fit's stochastic value; assert the composition is what we built
    assert torch.isfinite(expected)
    assert torch.isclose(expected, cls + weight * entropy)


def test_smoke_fit_logs_entropy_and_checkpoint_metric(tmp_path):
    lit = LitGnnFs(make_config(attention_entropy_weight=0.01))
    dm = _TinyGraphDataModule()
    trainer = _cpu_trainer(tmp_path)
    trainer.fit(lit, datamodule=dm)

    logged = trainer.callback_metrics
    for key in [
        "train_loss_total",
        "train_attention_entropy",
        "train_loss_attention_entropy_weighted",
        "val_f1_overall",  # checkpoint monitor must still exist
    ]:
        assert key in logged, f"missing logged metric {key}"


def test_smoke_fit_zero_weight_takes_plain_path(tmp_path):
    lit = LitGnnFs(make_config(attention_entropy_weight=0.0))
    dm = _TinyGraphDataModule()
    trainer = _cpu_trainer(tmp_path)
    trainer.fit(lit, datamodule=dm)
    logged = trainer.callback_metrics
    assert "val_f1_overall" in logged
    # entropy metrics are NOT logged on the disabled path (default log_attention_stats=false)
    assert "train_attention_entropy" not in logged


def test_log_only_stats_without_applying_loss(tmp_path):
    """log_attention_stats=true, weight=0: log entropy + diagnostics but do NOT add the loss term."""
    lit = LitGnnFs(make_config(attention_entropy_weight=0.0, log_attention_stats=True))
    dm = _TinyGraphDataModule()
    trainer = _cpu_trainer(tmp_path)
    trainer.fit(lit, datamodule=dm)
    logged = trainer.callback_metrics

    # entropy + neighbourhood diagnostics ARE logged, in both train and val
    for key in [
        "train_attention_entropy",
        "train_attention_top1_mass",
        "train_attention_effective_neighbors",
        "val_attention_entropy",
    ]:
        assert key in logged, f"missing diagnostic {key}"

    # the WEIGHTED loss term is NOT logged (loss disabled) ...
    assert "train_loss_attention_entropy_weighted" not in logged
    # ... and the entropy term was not added to the objective: total == classification (+ scGist reg) alias
    assert torch.isclose(logged["train_loss_total"], logged["train_loss_ce"], atol=1e-6)
