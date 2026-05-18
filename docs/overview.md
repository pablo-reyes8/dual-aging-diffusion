# Project Overview

This repository implements a dual-scale face aging pipeline built around latent diffusion models. The system separates face aging into two complementary branches:

- a global branch that edits the full face and captures coarse age-related appearance changes;
- a local branch that edits region-level facial crops and captures wrinkles, folds, and texture details;
- a deterministic fusion stage that combines both outputs while keeping the original image as the structural anchor.

The project is designed as research code with production-oriented structure: reusable data modules, explicit configuration files, command-line entrypoints, tests, Docker support, and documentation for the training and inference lifecycle.

## Core Idea

Face aging has both low-frequency and high-frequency components.

The global branch learns broad facial age cues such as apparent age, volume, hair color, skin tone, and global consistency. It operates on full-face images, typically at `512x512`.

The local branch learns zone-specific aging details such as forehead lines, crow's feet, under-eye wrinkles, nasolabial folds, glabellar lines, upper-lip texture, nasal bridge changes, and marionette lines. It operates on local crops, typically at `256x256`.

The final image is not produced by blindly trusting the global diffusion output. Instead, the original image remains the identity and geometry anchor. The global output contributes a controlled low-frequency residual, and the local outputs contribute high-frequency region details.

## Pipeline Summary

```text
input face image
    |
    |--> global diffusion branch
    |       full-face img2img aging
    |       output: x_global
    |
    |--> local diffusion branch
    |       region crops + local prompts/scores
    |       output: aged local crops
    |
    |--> deterministic fusion
            x_coarse = x_orig + lowpass(x_global - x_orig)
            x_blend  = insert aged crops into x_coarse
            x_final  = optional refiner or deterministic output
```

## Why Not Use Only One Diffusion Model?

A full-face diffusion model can create plausible aged faces, but it can also change identity, facial geometry, hair, expression, or demographic attributes. This is especially risky when training data is not longitudinal, because the model does not observe the same person aging over time.

A local-only model can preserve identity better, but it has limited access to global age semantics and can produce region edits that do not match the overall apparent age.

The dual-scale approach makes this tradeoff explicit:

- global branch: age direction and coarse consistency;
- local branch: detailed localized aging;
- original image: structure and identity anchor;
- fusion: controlled integration.

## Main Components

| Area | Main files | Purpose |
|---|---|---|
| Data | `data/create_data.py`, `data/local_path_dataset.py`, `data/global_path_datasets.py` | Build local/global datasets and dataloaders |
| Loader audits | `data/analyze_loaders.py`, `data/local_utils.py`, `data/global_utils.py` | Inspect sample distributions, scores, prompts, and batch shapes |
| Diffusion models | `src/diffusion_pipeline/load_diffusion_models.py` | Load SD models, VAEs, schedulers, and training bundles |
| Adapters | `src/diffusion_pipeline/LoRa.py`, `src/diffusion_pipeline/DoRa.py` | Inject LoRA/DoRA adapters into attention projections |
| Local loss | `src/loss/local_loss.py` | LDLA-style local branch objective |
| Global loss | `src/loss/global_loss.py`, `src/loss/global_aux_bundle.py` | Diffusion + age + identity + perceptual objectives |
| Training | `src/training/train_aging_model.py` | Global/local training wrapper with memory-aware branch scheduling |
| Inference | `src/inference/global_local_fusion.py` | Deterministic global-local fusion |
| CLIs | `scripts/*.py` | High-level data, training, and inference entrypoints |
| Configs | `configs/**/*.yaml` | Auditable run configuration |

## Recommended Reading Order

1. [Data Pipeline](data.md)
2. [Models and Adapters](models_and_adapters.md)
3. [Losses](losses.md)
4. [Training Pipeline](training.md)
5. [Inference Pipeline](inference.md)
6. [Configuration and CLIs](configuration_and_clis.md)

