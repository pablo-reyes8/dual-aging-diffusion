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

The local branch runs img2img independently for each crop.

Default generation settings:

```yaml
local_strength: 0.20
local_guidance_scale: 0.8
local_num_inference_steps: 40
```

Each crop can override these defaults.

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
