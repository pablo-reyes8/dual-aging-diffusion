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

## Docker

Build and run the CPU smoke-test image:

```bash
docker build -t diffusion-aging .
docker run --rm diffusion-aging
```

## License

MIT.
