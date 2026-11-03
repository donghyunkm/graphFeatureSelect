# CLAUDE.md

Durable guide for this repo. **Volatile detail — architecture internals, current experiment state,
active bugs, config conventions — lives in `progress.md`** (see its Reference appendix). Deep write-ups
live in `docs/`; research history in `MEMORY.md`. **Read `progress.md` before substantial work.**

## Project
Graph-based gene-panel feature selection for spatial transcriptomics: train a GNN to classify cell
subclasses from gene expression + a spatial neighbor graph, with a learnable feature-selection layer, to
pick a small (~10–20 gene) panel. Current research question and status: see `progress.md`.

## Tech stack
PyTorch + PyTorch Geometric (GATv2/SAGE/GCN), PyTorch Lightning, Hydra configs, TensorBoard/CSV logging.
Data: AnnData `.h5ad` (mouse isocortex). Python 3.12, package installed editable as `gfs`. Slurm (A100) for training.

## Repo layout
- `gfs/` — package: `models/`, `data/` (data modules), `trainers/` (Hydra+Lightning entrypoints), `configs/` (`antelope*.yaml`).
- `scripts/` — Slurm sbatch launchers (`main*.sh`); one script ≈ one experiment arm.
- `docs/` — focused technical write-ups (experiments, analyses, decisions).
- `data/` — `.h5ad` files + `checkpoints/`, `logs/`, `exposure/`; `config.toml` sets `data_root`.
- `outs/` — Slurm stdout/stderr.

## Environment (required)
Always run in the `gfs` conda env:
```bash
source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs
```
`sbatch`/`squeue` are NOT on the assistant's non-interactive PATH — the user runs Slurm commands in their own terminal.

## Commands
```bash
# Setup (full CUDA/PyG wheel list in README.md):
pip install -e .[dev]

# Lint / format (ruff, line-length 120; config in pyproject.toml):
ruff check . && ruff format .
pre-commit run --all-files

# Train (run from scripts/ — sbatch scripts use ../ relative paths):
cd /home/donghyun.kim/graphFeatureSelect/scripts && sbatch main.sh
# Or run a trainer directly, overriding config via Hydra:
python -u ../gfs/trainers/antelope.py --config-name antelope_abundant data.prefix="myexp" model.n_select=5 data.rand_seed=0

# Monitor Slurm (user's terminal only):
squeue -u donghyun.kim
tail -f /home/donghyun.kim/graphFeatureSelect/outs/out-comms-<jobid>.out
```
No automated test suite. "Validating a change" = smoke-test that the affected loader/model builds and a
few steps run, then recompute metrics from `test_pred.pt` — never quote F1 off a plot.

## Development guidelines
- Inspect before implementing — the idea usually already exists in a sibling file.
- Reuse existing abstractions: new experiment = new config + sbatch script reusing an existing trainer;
  add a new trainer/data module only when behavior genuinely differs.
- Keep changes focused and minimal; no unrelated refactors or reformatting.
- Don't silently change load-bearing interfaces (config keys, `fs_method` semantics, the `x` layout, the
  shuffle-control contract) — flag them.
- Validate before claiming; clearly separate completed & verified / untested / proposed work.

## Documentation
- Detailed write-ups (experiments, analyses, decisions, investigations) go in `docs/`, one topic per file;
  update an existing doc rather than spawning overlapping ones. See `docs/README.md` for the index.
  Naming: dated experiment run → `YYYYMMDD_topic.md`; durable investigation → topic name. `docs/old-claude.md`
  is read-only history.
- `progress.md` = concise chronological log (date, what was done, outcomes, `docs/` links, running jobs +
  Slurm IDs, blockers, next steps) plus a Reference appendix for durable-but-volatile detail. Deep detail goes in `docs/`.
- `MEMORY.md` = research history (prior findings, retractions); skim before concluding on baselines/shuffle/panel.

## End-of-session procedure
1. Review the session diff. 2. Run available checks (ruff; smoke-test affected loaders/models).
3. Update relevant `docs/*.md`. 4. Update `progress.md` (concise). 5. Record unfinished work + next steps.
6. Record running jobs + identifiers (Slurm IDs, log dirs). 7. Confirm docs match the repo's actual state.

Session-specific detail belongs in `progress.md`/`docs/`, never in `CLAUDE.md`.
