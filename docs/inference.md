# Inference Pipeline

Inference combines trained global and local adapters with deterministic fusion.

The main CLI is:

```text
scripts/inference_cli.py
```

The main fusion function is:

```text
src/inference/global_local_fusion.py::fuse_global_local_outputs
```

For notebooks that already build `sampling_objects` with `build_single_person_sampling_loader`, use:

```text
src/inference/inference_wrapper.py::run_sampling_objects_inference
```

## Inputs

Inference expects:

- an input face image;
- a global aging prompt;
- a global adapter checkpoint `.pt`;
- a local adapter checkpoint `.pt`;
- a local crop specification JSON;
- an output directory.

The crop specification can be a list or an object with a `crops` key.

Example:

```json
{
  "crops": [
    {
      "zone_name": "forehead",
      "bbox": [120, 80, 390, 190],
      "prompt": "a close-up of the forehead region with visible forehead wrinkles and an aging score of 80%"
    }
  ]
}
```

Each crop can optionally include:

- `crop_path`;
- `mask_path`;
- `negative_prompt`;
- `strength`;
- `guidance_scale`;
- `num_inference_steps`;
- `seed`.

If `crop_path` is omitted, the crop is extracted from the input image using `bbox`.

## Global Generation

The global branch runs img2img on the full image using the global prompt.

Default generation settings:

```yaml
global_strength: 0.30
global_guidance_scale: 5.0
global_num_inference_steps: 35
```

The global output should be treated as an age-direction signal, not as the final image.

## Local Generation

The local branch supports two interchangeable inference strategies with the
same trained DoRA weights:

- `img2img`: historical baseline; adds newly sampled noise to the crop latent;
- `ddim_inversion`: inverts the observed crop with its source condition and
  denoises the same inverted latent with the target condition.

DDIM inversion is an inference operator. It does not add a loss, trainable
parameter, or mandatory checkpoint migration. It constructs a latent/noise
trajectory compatible with the observed crop; it should not be described as
recovering the "true noise" that originally generated a photograph.

Default generation settings:

```yaml
local_strength: 0.20
local_guidance_scale: 0.8
local_num_inference_steps: 40
```

Each crop can override these defaults.

The historical default remains backward-compatible:

```yaml
generation:
  local_generation_method: img2img
```

The experimental baseline is [`configs/inference/ddim_inversion.yaml`](../configs/inference/ddim_inversion.yaml):

```yaml
generation:
  local_generation_method: ddim_inversion
  local_inversion:
    enabled: true
    method: ddim
    num_steps: 40
    strength: 0.45
    inversion_guidance_scale: 1.0
    edit_guidance_scale: null
    return_source_reconstruction: true
    cache_enabled: true
    post_edit_img2img_passes: 0
    fallback_to_img2img: true
```

Source conditioning is resolved in this order: an explicit/metadata source
score, a loaded `bundle["score_net"]`, and finally the score-free `zone_prompt`.
The target prompt is never used with its target score as the description of the
observed source crop. Local JSON specs may provide `source_score`,
`source_prompt`, `zone_prompt`, and `target_score` explicitly.

`strength` selects the fraction of the deterministic inverse trajectory. A
small value anchors the edit more strongly; a large value gives the target
condition more freedom. With inversion active, local recycling is disabled by
default. Any extra img2img pass must be requested explicitly with
`post_edit_img2img_passes`.

## Deterministic Fusion

The fusion stage combines:

- `x_orig`: original image;
- `x_global`: full-face aged image;
- `local_outputs`: generated aged crops.

The deterministic path does not require a refiner model. It computes a low-frequency global residual:

```text
x_coarse = x_orig + alpha * lowpass(x_global - x_orig)
```

Then it inserts local aged crops with mask feathering and optional color matching:

```text
x_blend = insert_local_outputs(x_coarse, local_outputs)
x_final = x_blend
```

Default fusion settings:

```yaml
residual_alpha: 0.35
residual_sigma: 9.0
use_face_mask: true
face_mask_blur_sigma: 3.0
local_insert_alpha: 1.0
local_mask_blur_sigma: 5.0
color_match: true
color_match_strength: 0.75
```

## Optional Refiner

The fusion function can accept a `fusion_bundle`. When no bundle is passed, inference is deterministic.

The current CLI uses deterministic fusion by default. This keeps inference predictable and avoids requiring an additional model.

The notebook wrapper can build and use the refiner by setting `config["refiner"]["enabled"] = True`.

Example:

```python
from src.inference.inference_wrapper import run_sampling_objects_inference

result = run_sampling_objects_inference(
    sampling_objects=sampling_objects,
    mixed_global_bundle=mixed_global_bundle,
    mixed_local_bundle=mixed_local_bundle,
    checkpoint_paths={
        "global": "training_checkpoints/run/global/best/best_adapter_inference.pt",
        "local": "training_checkpoints/run/local/best/best_adapter_inference.pt",
    },
    config={
        "generation": {
            "global_strength": 0.30,
            "global_guidance_scale": 5.0,
            "global_num_inference_steps": 35,
            "local_strength": 0.20,
            "local_guidance_scale": 0.8,
            "local_num_inference_steps": 40,
            "local_recycle_passes": 2,
            "seed": 77,
        },
        "refiner": {
            "enabled": True,
            "strength": 0.055,
            "guidance_scale": 1.5,
            "num_inference_steps": 12,
        },
    },
    output_dir="outputs/inference/09501",
)
```

If the bundles already have adapters injected, the wrapper only restores weights. If they are plain base bundles, it injects the adapter architecture from checkpoint metadata before restoring weights.

The refiner also has an independent experimental DDIM inversion flag. It is
OFF by default. This route uses the SDXL pipeline's own `encode_prompt` and
micro-conditioning helpers so both text encoders, pooled embeddings and
`time_ids` reach the UNet. If the loaded SDXL pipeline/scheduler is
incompatible, `fallback_to_img2img: true` emits a warning and runs the
historical low-strength refiner.

```yaml
refiner:
  enabled: true
  inversion:
    enabled: true
    method: ddim
    num_steps: 20
    strength: 0.15
    inversion_guidance_scale: 1.0
    fallback_to_img2img: true
```

This refiner mode is experimental until its source round-trip is validated on
the exact SDXL checkpoint/version used on the GPU host. Null-text Inversion is
not implemented.

## GPU Ablation After Training

The ablation runner loads an existing local checkpoint once and compares the
historical path against DDIM inversion on identical crops, prompts and seeds.
It does not train anything:

```bash
python -m scripts.ablate_ddim_inversion \
  --config configs/inference/ddim_inversion.yaml \
  --image path/to/person.png \
  --local-spec path/to/local_crops.json \
  --local-checkpoint path/to/local_best_inference.pt \
  --score-net-checkpoint path/to/scorenet_best.pt \
  --strengths 0.25,0.35,0.45,0.55 \
  --output-dir outputs/ddim_ablation
```

Add `--compute-lpips` when the optional `lpips` package is installed. For every
zone the runner saves the source, historical img2img output, source
reconstruction, each inverted edit, a comparison grid, and `metrics.json` with
MSE/PSNR, optional LPIPS, ScoreNet target error, latency and peak allocated CUDA
memory. These measurements are the evidence needed to decide whether inversion
actually improves preservation without weakening score control.

## Running Inference

Dry-run:

```bash
python -m scripts.inference_cli \
  --config configs/inference/default_inference.yaml \
  --image path/to/person.png \
  --global-prompt "a portrait photo of an elderly person" \
  --local-spec configs/inference/local_spec.example.json \
  --dry-run
```

Actual run:

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

The CLI writes intermediate and final outputs into the selected output directory.
