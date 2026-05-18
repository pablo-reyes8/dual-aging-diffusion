# Diffusion Aging

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-pytest-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Dual-scale face aging with latent diffusion models. The project combines a full-face global aging branch, a region-level local aging branch, and deterministic residual fusion to preserve identity while adding both coarse and fine-grained aging cues.

The repository is organized as research code with an engineering layer around it: reusable modules, auditable YAML configs, high-level CLIs, smoke tests, Docker support, and documentation.

## Contents

- [What This Project Does](#what-this-project-does)
- [Repository Structure](#repository-structure)
- [Documentation](#documentation)
- [Quick Start](#quick-start)
- [Data](#data)
- [Training](#training)
- [Inference](#inference)
- [Testing](#testing)
- [Docker](#docker)
- [License](#license)

## What This Project Does

Face aging is treated as a dual-scale problem.

The **global branch** works on full-face images and learns broad age-related changes: apparent age, facial structure, hair/skin appearance, and low-frequency consistency. The **local branch** works on facial region crops and learns localized signs of aging such as forehead lines, crow's feet, under-eye wrinkles, nasolabial folds, and marionette lines.

The final image is assembled through deterministic fusion. The original image remains the structural anchor, the global branch contributes a controlled low-frequency residual, and the local branch contributes region-specific details.

## Repository Structure

```text
diffusion_aging/
|-- configs/                 YAML/JSON configs for data, training, and inference
|-- data/                    Dataset builders, local/global dataloaders, loader audits
|-- docs/                    Project documentation in Markdown
|-- models/                  Local model artifacts and checkpoints
|-- notebooks/               Research notebooks and pipeline experiments
|-- planning/                Methodology notes and design checkpoints
|-- scripts/                 High-level CLIs for data, training, and inference
|-- src/
|   |-- diffusion_pipeline/  Diffusion bundle loading plus LoRA/DoRA adapters
|   |-- inference/           Deterministic global-local fusion
|   |-- loss/                Local and global aging losses
|   |-- score_net/           Local aging score network
|   |-- training/            Training wrapper, epochs, schedulers, checkpoints
|   `-- utils/               Shared utilities
|-- tests/                   Pytest smoke tests and module contracts
|-- Dockerfile
|-- pyproject.toml
`-- requirements*.txt
```

## Documentation

Start here:

- [Project Overview](docs/overview.md)
- [Data Pipeline](docs/data.md)
- [Models and Adapters](docs/models_and_adapters.md)
- [Losses](docs/losses.md)
- [Training Pipeline](docs/training.md)
- [Inference Pipeline](docs/inference.md)
- [Configuration and CLIs](docs/configuration_and_clis.md)
- [Testing and DevOps](docs/testing_and_devops.md)

## Quick Start

Create an environment and install the development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-dev.txt
```

For GPU training, install the PyTorch build that matches your CUDA runtime first, then install the project dependencies:

```bash
pip install -r requirements.txt
```

## Data

The local branch can be audited from repository fixtures. The global branch keeps Drive/Colab-oriented paths by default because the full global dataset is external.

Audit the local dataloader:

```bash
python -m scripts.data_cli --config configs/data/local_data.yaml --branch local
```

Audit the global dataloader in an environment where the configured global paths exist:

```bash
python -m scripts.data_cli --config configs/data/global_data.yaml --branch global
```

## Training

Training is configured through `configs/training/default_train.yaml` and launched through the high-level training CLI. The CLI loads data, diffusion bundles, adapters, optional ScoreNet, losses, schedulers, and the global-local training wrapper.

Validate the config without loading diffusion models:

```bash
python -m scripts.train_cli --config configs/training/default_train.yaml --dry-run --print-config
```

Start training:

```bash
python -m scripts.train_cli --config configs/training/default_train.yaml
```

## Inference

Inference assumes trained adapter `.pt` checkpoints are available. The user provides an input image, a global prompt, checkpoint paths, and a JSON file describing the local crops.

Dry-run:

```bash
python -m scripts.inference_cli \
  --config configs/inference/default_inference.yaml \
  --image path/to/person.png \
  --global-prompt "a portrait photo of an elderly person" \
  --local-spec configs/inference/local_spec.example.json \
  --dry-run
```

Run inference:

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

## Testing

The test suite validates module contracts without downloading diffusion models or running training.

```bash
pytest -q
```

## Docker

Build and run the CPU smoke-test image:

```bash
docker build -t diffusion-aging .
docker run --rm diffusion-aging
```

## License

MIT. See [LICENSE](LICENSE).
