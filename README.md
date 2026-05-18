# Diffusion Aging

Research code for a dual-scale face aging pipeline with:

- a **global branch** for full-face age coherence,
- a **local branch** for region-level aging details,
- deterministic global-local fusion for training-time monitoring and inference experiments.

The project is still evolving, but the current structure is organized around reusable `data`, `src/training`, and `src/inference` modules.

## Repository Layout

```text
data/                 Dataset builders and loader analysis scripts
src/diffusion_pipeline LoRA/DoRA diffusion model utilities
src/inference/         Deterministic fusion and optional refiner bundle
src/loss/              Local/global loss modules
src/score_net/         Local aging score network
src/training/          Training loops, schedulers, checkpoints, sampling helpers
scripts/               High-level CLIs for data, training, and inference
configs/               YAML/JSON configs for data, training, and inference runs
tests/                 Pytest smoke tests for shapes, imports, and core contracts
planning/              Methodology and implementation notes
```

## Setup

CPU/dev setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-dev.txt
```

GPU/diffusion setup depends on the CUDA version available in your machine. Install the matching PyTorch build first, then:

```bash
pip install -r requirements.txt
```

## Tests

Default tests are smoke tests. They verify dimensions, prompt contracts, checkpoint helpers, data builders, and deterministic fusion without downloading diffusion models or running training.

```bash
pytest -q
```

## Data

Local branch fixtures are expected under `data/data_subset` and `data/results_labeling`. Global branch paths are kept Colab/Drive-oriented unless overridden by the caller.

Prepare and audit the local dataloader:

```bash
python -m scripts.data_cli --config configs/data/local_data.yaml --branch local
```

The global data config is intended for Colab/Drive-mounted runs:

```bash
python -m scripts.data_cli --config configs/data/global_data.yaml --branch global
```

## Training CLI

The training CLI is a high-level orchestration entrypoint. It loads data, diffusion bundles, adapters, optional ScoreNet, losses, checkpoint managers, and then calls the global-local training wrapper.

Validate the config without loading models:

```bash
python -m scripts.train_cli --config configs/training/default_train.yaml --dry-run --print-config
```

Run training after configuring model IDs, data paths, loss options, and checkpoint output:

```bash
python -m scripts.train_cli --config configs/training/default_train.yaml
```

## Inference CLI

The inference CLI assumes trained adapter `.pt` checkpoints exist. The user provides the full image, a global prompt, checkpoint paths, and a JSON local crop spec.

Dry-run config validation:

```bash
python -m scripts.inference_cli \
  --config configs/inference/default_inference.yaml \
  --image path/to/person.png \
  --global-prompt "a portrait photo of an elderly person" \
  --local-spec configs/inference/local_spec.example.json \
  --dry-run
```

Actual inference:

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

## Docker

Build and run the CPU smoke-test image:

```bash
docker build -t diffusion-aging .
docker run --rm diffusion-aging
```

## License

MIT.
