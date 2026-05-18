# Contributing

This repository is research code for a dual-scale face aging pipeline. Contributions should keep the project reproducible, inspectable, and safe to run in CPU-only CI.

## Development Setup

For CPU-only CI parity:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-dev.txt
```

For GPU work, install the PyTorch build that matches your CUDA runtime first, then install project dependencies:

```bash
pip install -r requirements.txt
```

If identity loss is enabled, install FaceNet without dependencies after PyTorch is already installed:

```bash
pip install --no-deps facenet-pytorch
```

Do not use a plain `pip install facenet-pytorch` in CUDA environments unless you intentionally want pip to resolve PyTorch dependencies.

## Pull Request Scope

Prefer small PRs that touch one area:

- data/dataloaders/DataOps;
- diffusion bundles/adapters;
- losses;
- training loop/checkpoints/sampling;
- inference/fusion;
- docs/tests/devops.

Avoid unrelated formatting churn, notebook output churn, and broad refactors unless the PR is explicitly about cleanup.

## Tests

Run the default smoke suite before opening a PR:

```bash
pytest -q
```

Focused tests are encouraged:

```bash
pytest tests/data -q
pytest tests/training -q
pytest tests/inference -q
```

Default tests must not download diffusion models, train models, require a GPU, or depend on external Drive paths. Tests requiring model weights should be marked with `requires_models` and kept out of the default CI path.

## Data and Model Artifacts

Do not commit:

- private face images;
- generated outputs;
- training checkpoints;
- Hugging Face caches;
- API keys or storage credentials;
- local experiment logs.

Small manifests, schemas, and fixture metadata are acceptable when they improve reproducibility.

## Code Style

- Prefer existing project patterns over new abstractions.
- Keep data paths configurable.
- Avoid import-time downloads, extraction, model loading, or GPU allocation.
- Keep training behavior unchanged unless the PR explicitly changes it.
- Make optional heavy behavior opt-in through config or function arguments.

## Dependency Changes

Dependency PRs should explain:

- why the dependency is needed;
- whether it can reinstall or conflict with PyTorch/CUDA;
- whether it belongs in runtime, optional extras, or development dependencies.

PyTorch and torchvision are intentionally installed from an explicit CPU/CUDA wheel index. Do not let dependency updates silently rewrite that behavior.

## Documentation

Update README/docs/config comments when changing:

- user-facing commands;
- config keys;
- data path expectations;
- loader output contracts;
- training or inference behavior.
