# Configuration and CLIs

The repository exposes high-level command-line entrypoints under:

```text
scripts/
```

The corresponding default configs live under:

```text
configs/
```

The goal is to make data inspection, training, and inference reproducible without hiding the important hyperparameters inside notebooks.

## Data CLI

Entrypoint:

```bash
python -m scripts.data_cli --config configs/data/local_data.yaml --branch local
```

Config:

```text
configs/data/local_data.yaml
configs/data/global_data.yaml
```

Key fields:

```yaml
branch: local
batch_size: 8
num_workers: 0
n_batches: 20
audit: true
```

When `audit: true`, the CLI calls the detailed loader audit functions. When `audit: false`, it only builds the dataloaders.

## Training CLI

Entrypoint:

```bash
python -m scripts.train_cli --config configs/training/default_train.yaml
```

Dry-run:

```bash
python -m scripts.train_cli --config configs/training/default_train.yaml --dry-run --print-config
```

The training CLI orchestrates:

- local/global dataloaders;
- diffusion model bundles;
- LoRA/DoRA adapters;
- optimizers;
- optional ScoreNet;
- local/global loss objects;
- global-local training wrapper;
- checkpoints.

## Inference CLI

Entrypoint:

```bash
python -m scripts.inference_cli \
  --config configs/inference/default_inference.yaml \
  --image path/to/person.png \
  --global-prompt "a portrait photo of an elderly person" \
  --local-spec path/to/local_crops.json \
  --global-checkpoint path/to/global_best_inference.pt \
  --local-checkpoint path/to/local_best_inference.pt \
  --output-dir outputs/inference/example
```

The inference CLI assumes trained adapter checkpoints already exist.

## Config Philosophy

Configs should be explicit enough to audit an experiment after it runs. The important choices should be visible:

- model IDs;
- VAE IDs;
- adapter type/rank/alpha/dropout;
- optimizer settings;
- loss weights;
- branch scheduling;
- gradient accumulation;
- checkpoint behavior;
- inference strengths and fusion parameters.

## Recommended Workflow

1. Audit local data:

```bash
python -m scripts.data_cli --config configs/data/local_data.yaml --branch local
```

2. Dry-run training:

```bash
python -m scripts.train_cli --config configs/training/default_train.yaml --dry-run --print-config
```

3. Train:

```bash
python -m scripts.train_cli --config configs/training/default_train.yaml
```

4. Dry-run inference:

```bash
python -m scripts.inference_cli \
  --config configs/inference/default_inference.yaml \
  --image path/to/person.png \
  --global-prompt "a portrait photo of an elderly person" \
  --local-spec configs/inference/local_spec.example.json \
  --dry-run
```

5. Run inference with checkpoints:

```bash
python -m scripts.inference_cli \
  --config configs/inference/default_inference.yaml \
  --image path/to/person.png \
  --global-prompt "a portrait photo of an elderly person" \
  --local-spec path/to/local_crops.json \
  --global-checkpoint path/to/global_best_inference.pt \
  --local-checkpoint path/to/local_best_inference.pt \
  --output-dir outputs/inference/example
```

