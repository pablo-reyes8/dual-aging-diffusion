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

The Dockerfile is intended for CPU smoke testing and reproducible development checks.

Build:

```bash
docker build -t diffusion-aging .
```

Run tests:

```bash
docker run --rm diffusion-aging
```

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
