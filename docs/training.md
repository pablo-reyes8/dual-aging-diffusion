# Training Pipeline

The main training wrapper is:

```text
src/training/train_aging_model.py::train_global_local_face_aging
```

The high-level CLI entrypoint is:

```text
scripts/train_cli.py
```

The default auditable config is:

```text
configs/training/default_train.yaml
```

## Training Flow

The training pipeline performs the following steps:

1. load local and global dataloaders;
2. load global and local diffusion bundles;
3. inject LoRA/DoRA adapters;
4. build optimizers for adapter parameters;
5. optionally load ScoreNet;
6. build local and global loss objects;
7. call the global-local training wrapper;
8. alternate branch training according to `train_order`;
9. save latest, best, and inference checkpoints.

## Branch Scheduling

The wrapper supports training local and global branches in the same run while keeping their epoch counts independent.

Key hyperparameters:

```yaml
num_epochs: 5
local_num_epochs:
global_num_epochs:
train_order: [local, global]
train_local: true
train_global: true
```

If `local_num_epochs` or `global_num_epochs` is omitted, it inherits `num_epochs`.

`train_order` controls which branch is run first inside each epoch. The default is:

```text
local -> global
```

## Gradient Accumulation

The branches can use different accumulation and clipping settings:

```yaml
local_grad_accum_steps: 4
global_grad_accum_steps: 4
local_grad_clip: 1.0
global_grad_clip: 1.0
```

This is important because the global branch uses `512x512` images and has higher memory cost than the local branch.

## Local Mode Sampling

The local training loop samples among local loss modes:

```yaml
local_p_full: 0.50
local_p_score: 0.35
local_p_zone: 0.15
local_enable_full: true
local_enable_score: true
local_enable_zone: true
```

This prevents every batch from carrying every local objective at the same time.

## Global Mode Sampling

The global training loop samples between diffusion and semantic modes:

```yaml
global_p_diff: 0.55
global_p_semantic: 0.45
global_enable_diff: true
global_enable_semantic: true
global_semantic_components: [age, delta_age, id]
```

This is the main memory control for the global branch.

## Target Age Range

The wrapper can sample target ages within:

```yaml
min_target_age: 18
max_target_age: 90
```

This range is used when building global target prompts.

## Learning Rate Schedulers

Schedulers can be built automatically if missing from the bundles:

```yaml
build_schedulers_if_missing: true
local_warmup_ratio: 0.05
global_warmup_ratio: 0.05
local_min_lr: 0.000001
global_min_lr: 0.000001
min_warmup_steps: 10
max_warmup_steps:
```

The scheduler logic lives in:

```text
src/training/scheduler_warmup.py
```

## Checkpoints

The wrapper can save:

```yaml
save_latest: true
save_best: true
save_inference_copy: true
local_monitor_key: loss/total
global_monitor_key: loss/total
```

Checkpoint helpers live in:

```text
src/training/chekpoints.py
```

The inference copy is the most important artifact for downstream inference because it stores adapter weights and metadata needed by the inference CLI.

## Memory Controls

Recommended defaults:

```yaml
enable_gradient_checkpointing_flag: true
offload_after_each_branch: true
print_memory: true
```

These options exist because the project trains two diffusion branches but usually only one branch needs to be active on GPU at a time.

## Running Training

Dry-run the config without loading models:

```bash
python -m scripts.train_cli --config configs/training/default_train.yaml --dry-run --print-config
```

Start training:

```bash
python -m scripts.train_cli --config configs/training/default_train.yaml
```

## Tests

The tests are smoke tests. They validate imports, shapes, prompt utilities, checkpoint helpers, adapter injection, deterministic fusion, and CLI dry-runs. They do not download diffusion models or run training.

```bash
pytest -q
```

