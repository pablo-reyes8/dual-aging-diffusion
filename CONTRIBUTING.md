# Contributing

This project is research code for a dual-scale face aging pipeline. Keep changes scoped and reproducible.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

For CPU-only CI parity, install PyTorch from the CPU wheel index before installing the rest:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-dev.txt
```

## Tests

Run smoke tests before opening a PR:

```bash
pytest -q
```

The default tests must not download diffusion models, train models, or require a GPU. Tests that need model weights should be marked with `requires_models` and kept out of the default CI path.

## Style

- Prefer small, focused modules.
- Keep data paths configurable and avoid import-time extraction/downloads.
- Do not commit generated checkpoints, large datasets, or local experiment outputs.
