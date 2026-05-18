<p align="center">
  <img src="assets\header_image.png" width="1000"/>
</p>

# Diffusion Aging

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Status](https://img.shields.io/badge/status-active_research-orange)
![Tests](https://img.shields.io/badge/tests-pytest-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Dual-scale face aging with latent diffusion models. The project combines a full-face global aging branch, a region-level local aging branch, and deterministic residual fusion to preserve identity while adding both coarse and fine-grained aging cues.

This is research code with an engineering layer around it: reusable data modules, explicit configs, high-level CLIs, DataOps manifests, smoke tests, Docker support, and technical documentation.

> [!NOTE]
> This repository is not affiliated with FFHQ, NVIDIA, Stability AI, or the authors of any upstream diffusion model. FFHQ-derived data and pretrained model checkpoints remain governed by their own licenses and terms. The project is intended for research and controlled experimentation, not identity verification, demographic decision-making, or biometric deployment.

## Index

- [Why This Repo Exists](#why-this-repo-exists)
- [System Overview](#system-overview)
- [Implementation Coverage](#implementation-coverage)
- [Repository Layout](#repository-layout)
- [Documentation](#documentation)
- [Installation](#installation)
- [DataOps and Governance](#dataops-and-governance)
- [Command Line Tools](#command-line-tools)
- [Training Workflow](#training-workflow)
- [Inference Workflow](#inference-workflow)
- [Testing and CI](#testing-and-ci)
- [Docker](#docker)
- [Scope and Limitations](#scope-and-limitations)
- [License](#license)

## Why This Repo Exists

Face aging is not only a global semantic edit and not only a local wrinkle synthesis problem.

Global diffusion edits can produce plausible older faces, but they may change identity, expression, hair, face geometry, or demographic presentation. Local edits preserve structure better, but they do not fully capture broad apparent-age changes such as volume, hair, and global skin tone.

This repository studies a dual-scale design:

1. **Global branch:** learns full-face aging direction at `512x512`.
2. **Local branch:** learns region-specific aging details at `256x256`.
3. **Residual fusion:** uses the original image as the structural anchor, adds only controlled low-frequency global residuals, then inserts local aged crops with feathering and color matching.

The goal is a modular research pipeline where data, losses, adapters, checkpoints, and inference composition can be inspected independently.

## System Overview

```text
input face image
    |
    |-- global branch
    |     full-face latent diffusion img2img
    |     target: coarse apparent-age consistency
    |
    |-- local branch
    |     facial-region latent diffusion img2img
    |     target: wrinkles, folds, texture, local aging scores
    |
    `-- deterministic fusion
          x_coarse = x_orig + alpha * lowpass(x_global - x_orig)
          x_blend  = insert local aged crops into x_coarse
          x_final  = deterministic output or optional refiner output
```

## Implementation Coverage

| Area | Status |
| :--- | :--- |
| Local dataloader | Repository-local ZIP/JSON assets, local crop extraction, score-aware sampler |
| Global dataloader | FFHQ-derived full-face loader with Drive/Colab defaults and CSV metadata |
| DataOps | Dataset version config, schemas, governance docs, local quality manifest pipeline |
| Diffusion bundles | Global/local Stable Diffusion bundle loading with VAE and train/infer schedulers |
| Adapters | LoRA and DoRA injection into UNet attention projections |
| Local loss | LDLA-style full/zone/score/cycle components with ScoreNet support |
| Global loss | Diffusion, age, delta-age, identity, and optional perceptual components |
| Training | Global-local wrapper with branch scheduling, gradient accumulation, checkpointing, memory offload |
| Inference | Adapter checkpoint restore, global/local img2img, deterministic fusion |
| CLIs | Data audit, high-level training orchestration, high-level inference orchestration |
| Tests | CPU-safe smoke and contract tests; no diffusion downloads or full training in tests |

## Repository Layout

```text
diffusion_aging/
|-- configs/                       YAML/JSON configs for data, training, and inference
|   |-- data/
|   |-- inference/
|   `-- training/
|
|-- data/                          Data layer and DataOps assets
|   |-- configs/                   Dataset version definitions and quality thresholds
|   |-- data_subset/               Local fixture image ZIP
|   |-- ffhq_predictions/          Global attribute CSVs
|   |-- governance/                Dataset source, usage, and sensitive-attribute policy
|   |-- manifests/                 Generated quality/version manifests
|   |-- preprocessing/             Image quality audit pipeline
|   |-- results_labeling/          Local annotation ZIP/JSON files
|   `-- schemas/                   JSON schemas for annotations and manifests
|
|-- docs/                          Technical documentation
|-- models/                        Local model artifacts and checkpoints
|-- notebooks/                     Research notebooks and pipeline experiments
|-- planning/                      Methodology notes and design checkpoints
|
|-- scripts/                       Operational CLIs
|   |-- data_cli.py                Build/audit local and global dataloaders
|   |-- train_cli.py               High-level training entrypoint
|   `-- inference_cli.py           High-level deterministic inference entrypoint
|
|-- src/
|   |-- diffusion_pipeline/        Diffusion loading plus LoRA/DoRA adapters
|   |-- inference/                 Deterministic global-local fusion
|   |-- loss/                      Local/global aging losses
|   |-- score_net/                 Local aging score network
|   |-- training/                  Training loops, schedulers, checkpoints, sampling helpers
|   `-- utils/                     Shared utilities
|
|-- tests/                         Pytest smoke tests and module contracts
|-- Dockerfile
|-- pyproject.toml
`-- requirements*.txt
```

## Documentation

The technical documentation lives in [`docs/`](docs/). Recommended entry points:

- [Project Overview](docs/overview.md)
- [Data Pipeline](docs/data.md)
- [Models and Adapters](docs/models_and_adapters.md)
- [Losses](docs/losses.md)
- [Training Pipeline](docs/training.md)
- [Inference Pipeline](docs/inference.md)
- [Configuration and CLIs](docs/configuration_and_clis.md)
- [Testing and DevOps](docs/testing_and_devops.md)

Data-specific documentation:

- [DataOps Overview](data/README.md)
- [Data Governance](data/governance/DATA_GOVERNANCE.md)
- [Preprocessing and Quality Audit](data/preprocessing/README.md)

## Installation

For CPU development and tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-dev.txt
```

For GPU training, install the PyTorch build matching your CUDA runtime first, then:

```bash
pip install -r requirements.txt
```

If `GlobalLossAuxBundle(use_identity=True)` is enabled, install FaceNet separately without dependencies:

```bash
pip install --no-deps facenet-pytorch
```

This is intentional. `facenet-pytorch` declares PyTorch dependencies, and installing it normally can make pip try to download or replace `torch`/`torchvision`. The project only needs the FaceNet network code; it should reuse the PyTorch build already installed in the environment.

## DataOps and Governance

The local branch can be audited from the repository fixtures. The global branch expects external FFHQ-derived ZIP files and metadata CSVs.

Build the local quality manifest:

```bash
python -m data.preprocessing.quality_audit \
  --dataset-version data/configs/dataset_versions.yaml \
  --version local_subset_v1 \
  --output-dir data/manifests
```

This records image hashes, dimensions, quality metrics, annotation coverage, crop validity, and preprocessing flags. The generated manifest is intended to answer reproducibility questions such as:

- Which files were used?
- Did any image change?
- Which annotations are linked to each image?
- Are there obvious quality risks before training?
- Which images are blurry, overexposed, underexposed, noisy, too small, or suspiciously compressed?

Dataset governance is documented in [`data/governance/DATA_GOVERNANCE.md`](data/governance/DATA_GOVERNANCE.md). In short: FFHQ-derived data is used for research; demographic labels are noisy conditioning metadata; face data should be treated as sensitive.

## Command Line Tools

Audit local data:

```bash
python -m scripts.data_cli --config configs/data/local_data.yaml --branch local
```

Audit global data where Drive/external paths exist:

```bash
python -m scripts.data_cli --config configs/data/global_data.yaml --branch global
```

Validate training config without loading diffusion models:

```bash
python -m scripts.train_cli --config configs/training/default_train.yaml --dry-run --print-config
```

Validate inference arguments without loading checkpoints:

```bash
python -m scripts.inference_cli \
  --config configs/inference/default_inference.yaml \
  --image path/to/person.png \
  --global-prompt "a portrait photo of an elderly person" \
  --local-spec configs/inference/local_spec.example.json \
  --dry-run
```

## Training Workflow

Training is configured through [`configs/training/default_train.yaml`](configs/training/default_train.yaml). The high-level training CLI loads:

1. local and global dataloaders;
2. global and local diffusion bundles;
3. LoRA/DoRA adapters and adapter optimizers;
4. optional ScoreNet for local score loss;
5. local and global losses;
6. the global-local training wrapper.

Start training:

```bash
python -m scripts.train_cli --config configs/training/default_train.yaml
```

The notebook workflow can use the same modules directly without YAML. See [`docs/training.md`](docs/training.md) for the main training arguments and memory controls.

## Inference Workflow

Inference assumes trained adapter `.pt` checkpoints exist. The user provides an input image, a global prompt, checkpoint paths, and a JSON file describing local crops.

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

The deterministic fusion path writes intermediate images and the final fused output.

## Testing and CI

The test suite validates contracts without downloading diffusion models or running training:

```bash
pytest -q
```

Current local validation:

```text
26 passed
```

CI is defined in [`.github/workflows/tests.yml`](.github/workflows/tests.yml). It runs when code, configs, scripts, tests, dependency files, or workflow files change.

## Docker

Build and run the CPU smoke-test image:

```bash
docker build -t diffusion-aging .
docker run --rm diffusion-aging
```

## Scope and Limitations

This project does not currently provide published quantitative aging results, ablation tables, or pretrained release checkpoints for the full global-local system. The focus is the architecture, data pipeline, training mechanics, and reproducible research scaffolding.

The global branch depends on external FFHQ-derived data paths. The repository-local data is enough for local dataloader audits and smoke tests, but not enough to claim full model quality.

## License

MIT. See [LICENSE](LICENSE).
