# Models and Adapters

The project uses latent diffusion models as the generative backbone. The trainable parameters are adapter modules injected into diffusion UNet attention projections.

## Diffusion Bundles

The model loading utilities live in:

```text
src/diffusion_pipeline/load_diffusion_models.py
```

The loader builds two independent bundles:

- global bundle: full-face aging branch;
- local bundle: region crop aging branch.

Each bundle contains the modules required for training and inference:

```python
{
    "vae": AutoencoderKL,
    "unet": UNet2DConditionModel,
    "tokenizer": CLIPTokenizer,
    "text_encoder": CLIPTextModel,
    "scheduler_train": training_scheduler,
    "scheduler_infer": inference_scheduler,
}
```

The default model IDs in the configs are:

```yaml
global_model_id: SG161222/Realistic_Vision_V6.0_B1_noVAE
global_vae_id: stabilityai/sd-vae-ft-mse
local_model_id: runwayml/stable-diffusion-v1-5
local_vae_id:
```

## Why Two Diffusion Branches?

The global and local tasks have different resolutions and objectives.

The global branch operates on `512x512` full-face images and optimizes age direction, identity preservation, and broad semantic consistency.

The local branch operates on `256x256` facial crops and optimizes region-level aging detail. It receives local aging scores and zone prompts.

Keeping the branches separate prevents the local model from needing to learn full-face identity and prevents the global model from being solely responsible for small wrinkle-level details.

## LoRA

LoRA adapters are implemented in:

```text
src/diffusion_pipeline/LoRa.py
```

LoRA freezes the base linear layer and adds a low-rank residual update:

```text
W_eff = W_base + scale * B @ A
```

This reduces trainable parameters while preserving the pretrained diffusion model as the base.

Typical global adapter config:

```yaml
adapter_type: lora
rank: 8
alpha: 8
dropout: 0.0
target_suffixes: [to_q, to_k, to_v, to_out.0]
```

## DoRA

DoRA adapters are implemented in:

```text
src/diffusion_pipeline/DoRa.py
```

DoRA separates direction and magnitude in the adapted weight update. In this project it is useful for the local branch, where small local texture changes can benefit from stronger control over the adaptation.

Typical local adapter config:

```yaml
adapter_type: dora
rank: 16
alpha: 16
dropout: 0.05
target_suffixes: [to_q, to_k, to_v, to_out.0]
```

## Injection Targets

Adapters are injected into attention projection layers whose names end with one of:

```text
to_q
to_k
to_v
to_out.0
```

This targets the cross/self-attention projections in the UNet without modifying the entire architecture.

## Training Setup

The high-level setup function is:

```python
build_mixed_lora_dora_training_setup(...)
```

It:

1. freezes base diffusion parameters;
2. injects adapters into global and local UNets;
3. creates optimizers for trainable adapter parameters;
4. returns mixed bundles ready for the training wrapper.

## Checkpointing

Checkpoint helpers live in:

```text
src/training/chekpoints.py
```

The training wrapper can save:

- latest checkpoints;
- best checkpoints based on monitored loss;
- inference copies that contain adapter weights and metadata.

The inference CLI expects adapter `.pt` checkpoints and restores them into freshly loaded diffusion bundles.

