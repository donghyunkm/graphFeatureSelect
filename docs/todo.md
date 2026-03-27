# TODO: Rewrite Progress

## Phase 1: Data pipeline

- [x] Explore raw data format (metadata + sp-self h5ad files)
- [x] Identify dev section (section 600, ID 1199651094, balanced hemispheres)
- [x] Create dev dataset script (`notebooks/02_create_dev_dataset.py`)
- [x] Generate dev h5ad files with spatial graphs (`data/dev/`)
- [x] Decide on normalization strategy → already log1p(CPM), use as-is
- [x] Rewrite PyGAnnData: load normalized h5ad → PyG Data (`data/dataset.py`)
  - [x] Store gene_exp and xyz as separate Data attributes (not concatenated)
  - [x] Remove cell types with < min_count cells
  - [x] Encode labels with LabelEncoder
  - [x] Stratified train/val split within a single hemisphere file
  - [x] Support external label_encoder for test hemisphere
- [x] Rewrite DataModule (`data/datamodule.py`)
  - [x] Support separate train and test h5ad files (hemisphere split)
  - [x] Extract common dataloader helper (DRY)
  - [x] Wire up test_path config for inductive test set
- [x] Tests: 10 tests in `test_data_pipeline.py` — shapes, dtypes, no leakage, filtering, stratification

## Phase 2: Feature selection layers

- [x] FeatureSelector ABC base class (`feature_selection/base.py`)
- [x] Gumbel: ported from current, already correct for eval
- [x] STG: per-subgraph noise, hard top-k eval mask, no more _apply override
- [x] scGist: hard top-k eval mask, inlined regularization
- [x] Updated factory in `__init__.py` with new interface
- [x] Tests: 58 tests in `test_feature_selection.py` — mask shapes, binary at eval, per-subgraph variation, gradients

## Phase 3: GNN backbone

- [x] Extract GNN layer factory (`_build_gnn_layer(in_ch, out_ch, type)`)
- [x] Separate backbone from feature selection and task head
- [x] Replace Python-loop XYZ normalization with scatter ops
- [x] Tests: 11 tests in `test_backbone.py` — forward pass, GNN types, options, XYZ centering

## Phase 4: Task heads

- [x] Classification head: linear → logits (`heads.py`)
- [x] Reconstruction head: MLP → predicted expression, MSE loss (`heads.py`)
- [x] Tests: 2 tests in `test_backbone.py` — shape verification

## Phase 5: Lightning module

- [x] Wire feature selection + backbone + task head(s)
- [x] Fix tau_schedule (was referencing undefined variable)
- [x] Extract logging options helper (`_log_metrics`)
- [x] Remove empty no-op methods
- [x] Ensure hard masks at val/test, seed-node-only metrics
- [x] Two-phase init: `setup_model()` called after DataModule provides dimensions
- [x] Tests: 6 tests in `test_end_to_end.py` — creation, training step, loss reduction, hard masks, all selectors, reconstruction

## Phase 6: Hydra config

- [x] Split model config into backbone/, feature_selection/, task/ groups
- [x] Add global flags: n_select, trainmode, halfhop, lam
- [ ] Tests: config assembly, overrides, composition

## Phase 7: Entry point + cleanup

- [x] Minimal training entry point (`trainers/train.py`)
- [x] End-to-end test: train on dev data for a few epochs
- [x] Update CLAUDE.md with new structure
- [x] Update docs/ and refactor/ with current state
- [ ] Archive old src/gfs/ code (or remove dead code: stg/, get_sampler.py, samplers/)

## Functional feature selection tests

- [x] Shared test harness: `GatedMLP`, `train_gated_mlp`, `eval_accuracy` (`tests/featselect/conftest.py`)
- [x] Synthetic `toy_data` fixture: `make_classification` (5000 samples, 100 features, 10 informative, 10 classes)
- [x] Feature recovery tests: Gumbel/STG/scGist recover >= 5/10 informative features (`test_feature_recovery.py`)
- [x] Baseline tests: MLP accuracy sanity + learned mask beats random mask (`test_baseline.py`)
- [x] Tau/sigma behavior tests: temperature controls sharpness, gates bounded [0,1] (`test_tau_behavior.py`)
- [x] pytest `slow` marker for training-heavy tests (`pyproject.toml`)

## Future work

- [ ] Per-subgraph mask assignment via NeighborLoaderMod or equivalent (currently defaults to single subgraph)
- [ ] HalfHop transform integration
- [ ] Focal loss option in classification head
- [ ] Multi-task training (joint classification + reconstruction)
- [ ] Scale to full dataset (multiple sections)
