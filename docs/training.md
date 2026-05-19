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

## Optional Local Fused Loss

The local branch can optionally add a deterministic fused loss after the base crop-level local loss. This is disabled by default and uses a second aligned dataloader grouped by image/person.

The base local loader remains random and crop-level. The fused loader returns:

```text
full_pixel_values: [B, 3, 512, 512]
pixel_values:      [B, K, 3, 256, 256]
boxes:             [B, K, 4]
masks:             [B, K, 1, 256, 256]
target_scores:     [B, K]
valid_mask:        [B, K]
```

The fused loss is active only when:

```text
use_fused_loss is true
current epoch >= fused_loss_epoch
global local-step counter satisfies fused_loss_every_n_steps
```

Default flags:

```yaml
use_fused_loss: false
fused_loss_epoch: 15
fused_loss_every_n_steps: 1
lambda_fuse_score: 0.03
lambda_fuse_seam: 0.01
```

The fused path runs after the base local backward pass, so the base crop graph and fused graph are not kept alive at the same time. ScoreNet remains frozen, but the ScoreNet forward is not wrapped in `torch.no_grad()` because gradients must flow through the fused crop input back to the local branch.

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
save_epoch_checkpoints: true
save_inference_copy: true
local_monitor_key: loss/total
global_monitor_key: loss/total
```

Checkpoint helpers live in:

```text
src/training/chekpoints.py
```

The inference copy is the most important artifact for downstream inference because it stores adapter weights and metadata needed by the inference CLI.

Checkpoint outputs are organized per branch:

```text
checkpoint_root/
  local/
    latest/
    best/
    epoch_001/
    epoch_002/
  global/
    latest/
    best/
    epoch_001/
    epoch_002/
```

`latest/` remains the resume-friendly checkpoint. `best/` tracks the monitor metric. `epoch_XXX/` stores immutable snapshots for every completed branch epoch when `save_epoch_checkpoints=true`.

## Memory Controls

Recommended defaults:

```yaml
enable_gradient_checkpointing_flag: true
offload_after_each_branch: true
print_memory: true
```

These options exist because the project trains two diffusion branches but usually only one branch needs to be active on GPU at a time.

## Non-Finite Loss Diagnostics

The training loops check both losses and optimizer steps for non-finite values. These checks are always printed when triggered, even if inner verbose logging is disabled.

For a non-finite local loss, the warning includes:

```text
branch=local
batch_idx=...
global_step=...
loss_mode=full | zone | score
pixel_values min/max/mean/nonfinite
score and score_raw ranges
target score ranges
Components: loss_full/loss_zone/loss_score/noise_pred_...
```

For a non-finite global loss, the warning includes the same batch context plus the active global mode:

```text
loss_mode=diff | semantic
Components: loss_diff/loss_age/loss_delta_age/loss_id/...
```

## Monitoring Sampling Local Recycling

Monitoring fusion can optionally refine each local crop with more than one local img2img pass before deterministic fusion. This is inference-only sampling behavior; it does not change training gradients or the local loss.

Default behavior is unchanged:

```yaml
sample_local_recycle_passes: 1
```

Use two passes to enable local recycling:

```python
result = train_global_local_face_aging(
    ...,
    sample_local_recycle_passes=2,
    sample_local_recycle_strength=0.12,  # optional; defaults to sample_local_strength
)
```

If `sample_local_recycle_strength`, `sample_local_recycle_guidance_scale`, or `sample_local_recycle_num_inference_steps` are omitted, the recycling pass reuses the normal local sampling values.

The most important pattern to recognize is adapter corruption after an optimizer step:

```text
noise_pred_full: shape=(B, 4, H, W) nonfinite=all values
noise_pred_zone: shape=(B, 4, H, W) nonfinite=all values
noise_pred_target: shape=(B, 4, H, W) nonfinite=all values
```

If this appears across `full`, `zone`, and `score`, the issue is not the dataset, targets, prompts, or ScoreNet. It means the UNet forward itself is producing non-finite predictions. In this project the main known cause is training LoRA/DoRA adapter weights stored in `float16`.

The code prevents this by keeping trainable adapter parameters in `float32` while allowing AMP for forward compute. See:

```text
docs/models_and_adapters.md#adapter-dtype-stability
```

If non-finite adapter updates occurred before this protection was active, restart the kernel/runtime, rebuild bundles from scratch, and do not resume from the contaminated checkpoint.

Expected healthy local summary:

```text
micro ~= number of loader batches
optim_epoch ~= ceil(micro / grad_accum_steps)
skipped=0
```

Small stochastic variation in `loss/total` is normal because the loop samples different loss modes per batch.

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
