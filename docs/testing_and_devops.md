# Testing and DevOps

The repository includes a lightweight engineering layer around the research code.

## Tests

Tests are written for `pytest` and live in:

```text
tests/
```

They are smoke and contract tests. They validate:

- local dataset extraction and sample contracts;
- global dataset sample contracts;
- adapter injection;
- ScoreNet shape behavior;
- training utilities;
- checkpoint helpers;
- deterministic fusion;
- CLI dry-runs and config structure.

The tests intentionally do not:

- download diffusion models;
- run full training;
- require external Drive datasets;
- require GPU.

Run:

```bash
pytest -q
```

## CI

GitHub Actions configuration lives in:

```text
.github/workflows/tests.yml
```

The workflow runs tests when relevant project files change:

- `src/**`
- `data/**`
- `tests/**`
- `scripts/**`
- `configs/**`
- dependency files
- workflow files

README-only changes are not expected to trigger the full test workflow.

## Docker

Two Dockerfiles are provided under `docker/`: one CPU-only image for local
verification and one CUDA-enabled image for training servers. The root
`Dockerfile` remains as a backwards-compatible alias for the CPU workflow.

CPU build and tests:

```bash
docker build -f docker/Dockerfile.cpu -t diffusion-aging:cpu .
docker run --rm --network none diffusion-aging:cpu
```

CUDA training image:

```bash
docker build -f docker/Dockerfile.cuda -t diffusion-aging:cuda .
docker run --rm --gpus all --network none diffusion-aging:cuda
```

The CUDA image's default command is a dry-run. Start a real job explicitly with
`python -m scripts.train_cli --config ...`; mount `data/`, `models/`, and
`training_checkpoints/` from the server. It also contains JupyterLab for
`notebooks/train_model_yaml.ipynb`. Full training and model downloads are not
part of the CPU test suite.

## Dependency Files

Runtime dependencies:

```text
requirements.txt
```

Identity-loss dependency:

```bash
pip install --no-deps facenet-pytorch
```

Install FaceNet this way after PyTorch is already installed. A normal `pip install facenet-pytorch` may try to resolve and reinstall PyTorch packages.

Development/test dependencies:

```text
requirements-dev.txt
```

Project metadata:

```text
pyproject.toml
```

## Repository Hygiene

Large generated artifacts should not be committed unless they are intentional fixtures. Model checkpoints, training outputs, and generated inference outputs should generally remain outside version control.

The local fixture ZIPs under `data/` are currently part of the project workflow because the local branch can be audited from the repository.
