from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch

from src.training.training_display_helpers import print_section
from src.training.training_memory_helpers import (
    hard_cuda_cleanup,
    move_bundle_modules_only_to_device,
    offload_bundle_modules_only,
)

try:
    from src.inference.image_tensor_utils import image_to_tensor01, tensor01_to_pil
except ImportError:
    image_to_tensor01 = None
    tensor01_to_pil = None

try:
    from src.inference.global_local_fusion import fuse_global_local_outputs
except ImportError:
    fuse_global_local_outputs = None

# ============================================================
# Sampling batch parsing helpers
# ============================================================

def _first_batch(loader):
    if loader is None:
        return None

    if isinstance(loader, dict):
        return loader

    return next(iter(loader))


def _get_first_existing(batch: Dict[str, Any], keys: Sequence[str], default=None):
    for k in keys:
        if k in batch:
            return batch[k]
    return default


def _unwrap_singleton(x):
    """
    Dataloader batches often wrap one item in a list/tuple.
    For sampling loaders we assume batch size 1.
    """
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]

    if torch.is_tensor(x) and x.ndim >= 4 and x.shape[0] == 1:
        return x

    return x


def parse_sampling_global_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected flexible global sampling batch.

    Supported keys:
        image / x_orig / pixel_values / pil_image
        global_prompt / prompt / target_prompt
        face_mask / mask
        id / filename / image_id
    """
    if batch is None:
        raise ValueError("sampling_loader_global returned None.")

    x_orig = _get_first_existing(
        batch,
        ["image", "x_orig", "pixel_values", "pil_image"],
    )

    global_prompt = _get_first_existing(
        batch,
        ["global_prompt", "target_prompt", "prompt"],
    )

    face_mask = _get_first_existing(
        batch,
        ["face_mask", "mask", "full_face_mask"],
        default=None,
    )

    sample_id = _get_first_existing(
        batch,
        ["id", "image_id", "filename", "name"],
        default="sample",
    )

    x_orig = _unwrap_singleton(x_orig)
    global_prompt = _unwrap_singleton(global_prompt)
    face_mask = _unwrap_singleton(face_mask)
    sample_id = _unwrap_singleton(sample_id)

    if isinstance(global_prompt, (list, tuple)):
        global_prompt = global_prompt[0]

    if isinstance(sample_id, (list, tuple)):
        sample_id = sample_id[0]

    if x_orig is None:
        raise KeyError(
            "Global sampling batch must contain one of: "
            "'image', 'x_orig', 'pixel_values', 'pil_image'."
        )

    if global_prompt is None:
        raise KeyError(
            "Global sampling batch must contain one of: "
            "'global_prompt', 'target_prompt', 'prompt'."
        )

    return {
        "x_orig": x_orig,
        "global_prompt": str(global_prompt),
        "face_mask": face_mask,
        "sample_id": str(sample_id),
    }


def parse_sampling_local_batch(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expected flexible local sampling batch.

    Best format:
        {
            "zones": [
                {
                    "crop": ...,
                    "prompt": ...,
                    "bbox": (x1,y1,x2,y2),
                    "mask": ...
                },
                ...
            ]
        }

    Alternative vectorized batch format:
        {
            "crop" or "image" or "pixel_values": list/tensor,
            "prompt" or "target_prompt": list[str],
            "bbox": list[tuple],
            "mask": optional list/tensor,
            "zone_name": optional list[str]
        }
    """
    if batch is None:
        raise ValueError("sampling_loader_local returned None.")

    if "zones" in batch:
        zones = batch["zones"]

        if isinstance(zones, (list, tuple)):
            return list(zones)

        raise TypeError("batch['zones'] must be a list/tuple of zone dicts.")

    crops = _get_first_existing(batch, ["crop", "image", "pixel_values", "x_crop"])
    prompts = _get_first_existing(batch, ["target_prompt", "prompt", "local_prompt"])
    bboxes = _get_first_existing(batch, ["bbox", "bboxes", "bbox_xyxy"])
    masks = _get_first_existing(batch, ["mask", "masks", "local_mask"], default=None)
    names = _get_first_existing(batch, ["zone_name", "zone_names", "region"], default=None)

    if crops is None:
        raise KeyError("Local sampling batch must contain 'zones' or crop/image/pixel_values.")

    if prompts is None:
        raise KeyError("Local sampling batch must contain 'prompt' or 'target_prompt'.")

    if bboxes is None:
        raise KeyError("Local sampling batch must contain 'bbox'/'bboxes'/'bbox_xyxy'.")

    # Normalize to list-like.
    if torch.is_tensor(crops):
        if crops.ndim == 4:
            crop_list = [crops[i:i+1] for i in range(crops.shape[0])]
        else:
            crop_list = [crops]
    else:
        crop_list = list(crops) if isinstance(crops, (list, tuple)) else [crops]

    prompt_list = list(prompts) if isinstance(prompts, (list, tuple)) else [prompts]
    bbox_list = list(bboxes) if isinstance(bboxes, (list, tuple)) else [bboxes]

    if masks is None:
        mask_list = [None] * len(crop_list)
    elif torch.is_tensor(masks):
        if masks.ndim == 4:
            mask_list = [masks[i:i+1] for i in range(masks.shape[0])]
        else:
            mask_list = [masks]
    else:
        mask_list = list(masks) if isinstance(masks, (list, tuple)) else [masks]

    if names is None:
        name_list = [f"zone_{i}" for i in range(len(crop_list))]
    else:
        name_list = list(names) if isinstance(names, (list, tuple)) else [names]

    zones = []

    for i in range(len(crop_list)):
        zones.append({
            "zone_name": str(name_list[i]) if i < len(name_list) else f"zone_{i}",
            "crop": crop_list[i],
            "prompt": str(prompt_list[i]) if i < len(prompt_list) else str(prompt_list[-1]),
            "bbox": bbox_list[i] if i < len(bbox_list) else bbox_list[-1],
            "mask": mask_list[i] if i < len(mask_list) else None,
        })

    return zones


# ============================================================
# Generic sampling generation helpers
# ============================================================

def _get_img2img_pipe_from_bundle_safe(bundle: Dict[str, Any], bundle_name: str):
    """
    Tries to get a diffusers img2img pipeline from bundle.

    If your bundles do not expose a pipeline, pass:
        sample_global_forward_fn
        sample_local_forward_fn
    to the wrapper.
    """
    for key in ["pipe", "pipeline", "img2img_pipeline"]:
        if key in bundle and bundle[key] is not None:
            return bundle[key]

    raise KeyError(
        f"{bundle_name} does not contain 'pipe', 'pipeline', or 'img2img_pipeline'. "
        "Pass a custom sampling forward function to the wrapper."
    )


def _call_img2img_pipe_safe(
    pipe,
    image,
    prompt: str,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
    negative_prompt: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Minimal diffusers img2img call.
    Assumes image can be PIL or tensor convertible by your existing utilities.
    """
    if torch.is_tensor(image):
        if image_to_tensor01 is None or tensor01_to_pil is None:
            raise ImportError("image_to_tensor01 and tensor01_to_pil must be available from src.inference.image_tensor_utils for tensor sampling inputs.")
        image_input = tensor01_to_pil(image_to_tensor01(image))
    else:
        image_input = image

    kwargs = {
        "prompt": prompt,
        "image": image_input,
        "strength": float(strength),
        "guidance_scale": float(guidance_scale),
        "num_inference_steps": int(num_inference_steps),
        "generator": generator,
    }

    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt

    out = pipe(**kwargs)

    if hasattr(out, "images"):
        return out.images[0]

    if isinstance(out, (list, tuple)):
        return out[0]

    return out


def _bundle_has_tensor_img2img_components(bundle: Dict[str, Any]) -> bool:
    required = ["vae", "unet", "tokenizer", "text_encoder"]
    has_core = all(k in bundle and bundle[k] is not None for k in required)
    has_scheduler = (
        bundle.get("scheduler_infer", None) is not None
        or bundle.get("scheduler_train", None) is not None
    )
    return bool(has_core and has_scheduler)


def _image_to_minus1_1_tensor(image, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if image_to_tensor01 is None:
        raise ImportError("image_to_tensor01 must be available from src.inference.image_tensor_utils for tensor bundle sampling.")

    x01 = image_to_tensor01(image, device=device, dtype=dtype)
    return (x01 * 2.0 - 1.0).clamp(-1.0, 1.0)


@torch.inference_mode()
def _img2img_tensor_bundle_safe(
    *,
    bundle: Dict[str, Any],
    image,
    prompt: str,
    device: torch.device,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
    negative_prompt: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Minimal deterministic img2img path for the project's training bundles.

    It uses the already-loaded VAE, UNet, tokenizer, text encoder and
    scheduler_infer/scheduler_train, so monitoring sampling does not require
    a diffusers pipeline object inside the bundle.
    """
    if not _bundle_has_tensor_img2img_components(bundle):
        missing = [
            key for key in ["vae", "unet", "tokenizer", "text_encoder", "scheduler_infer"]
            if bundle.get(key, None) is None
        ]
        raise KeyError(f"Bundle does not expose pipeline or tensor img2img components. Missing: {missing}")

    vae = bundle["vae"]
    unet = bundle["unet"]
    tokenizer = bundle["tokenizer"]
    text_encoder = bundle["text_encoder"]
    scheduler = bundle.get("scheduler_infer", None) or bundle["scheduler_train"]

    unet_dtype = next(unet.parameters()).dtype
    vae_dtype = next(vae.parameters()).dtype
    text_dtype = next(text_encoder.parameters()).dtype

    num_inference_steps = max(1, int(num_inference_steps))
    strength = float(strength)

    image_tensor = _image_to_minus1_1_tensor(
        image,
        device=device,
        dtype=vae_dtype,
    )

    if strength <= 0:
        return image_tensor.detach()

    text_inputs = tokenizer(
        [negative_prompt or "", str(prompt)],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = getattr(text_inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    text_out = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    text_embeds = text_out.last_hidden_state.to(device=device, dtype=text_dtype)

    latents = vae.encode(image_tensor).latent_dist.mean
    latents = latents * vae.config.scaling_factor
    latents = latents.to(device=device, dtype=unet_dtype)

    try:
        scheduler.set_timesteps(num_inference_steps, device=device)
    except TypeError:
        scheduler.set_timesteps(num_inference_steps)

    timesteps_all = scheduler.timesteps.to(device)
    init_timestep = min(max(int(num_inference_steps * strength), 1), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = timesteps_all[t_start:]

    if timesteps.numel() == 0:
        return image_tensor.detach()

    if generator is not None and getattr(generator, "device", None) != device:
        local_generator = torch.Generator(device=device)
        local_generator.manual_seed(int(generator.initial_seed()))
        generator = local_generator

    noise = torch.randn(
        latents.shape,
        generator=generator,
        device=device,
        dtype=unet_dtype,
    )
    first_timestep = timesteps[0].repeat(latents.shape[0])
    latents = scheduler.add_noise(latents, noise, first_timestep).to(dtype=unet_dtype)

    for t in timesteps:
        latent_model_input = torch.cat([latents, latents], dim=0)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        noise_pred = unet(
            latent_model_input.to(dtype=unet_dtype),
            t,
            encoder_hidden_states=text_embeds.to(dtype=unet_dtype),
            return_dict=True,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + float(guidance_scale) * (
            noise_pred_text - noise_pred_uncond
        )

        latents = scheduler.step(
            noise_pred,
            t,
            latents,
            return_dict=True,
        ).prev_sample

    decoded = vae.decode(
        (latents / vae.config.scaling_factor).to(dtype=vae_dtype),
        return_dict=True,
    ).sample

    return decoded.clamp(-1.0, 1.0).detach().float()


def default_sample_global_forward(
    *,
    mixed_global_bundle: Dict[str, Any],
    x_orig,
    global_prompt: str,
    device: torch.device,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
    negative_prompt: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Default global sampling forward.

    Uses a diffusers pipeline when the bundle exposes one. Otherwise it falls
    back to the project's tensor bundle path.
    """
    pipe = None
    for key in ["pipe", "pipeline", "img2img_pipeline"]:
        if key in mixed_global_bundle and mixed_global_bundle[key] is not None:
            pipe = mixed_global_bundle[key]
            break

    if pipe is None:
        return _img2img_tensor_bundle_safe(
            bundle=mixed_global_bundle,
            image=x_orig,
            prompt=global_prompt,
            device=device,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            generator=generator,
        )

    if hasattr(pipe, "to"):
        pipe.to(device)

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    return _call_img2img_pipe_safe(
        pipe=pipe,
        image=x_orig,
        prompt=global_prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )


def default_sample_local_forward(
    *,
    mixed_local_bundle: Dict[str, Any],
    zones: List[Dict[str, Any]],
    device: torch.device,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
    negative_prompt: Optional[str] = None,
    recycle_passes: int = 1,
    recycle_strength: Optional[float] = None,
    recycle_guidance_scale: Optional[float] = None,
    recycle_num_inference_steps: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> List[Dict[str, Any]]:
    """
    Default local crop sampling forward.

    Returns local_outputs compatible with fuse_global_local_outputs:
        [
            {
                "zone_name": ...,
                "aged_crop": ...,
                "bbox": ...,
                "mask": ...
            }
        ]
    """
    pipe = None
    for key in ["pipe", "pipeline", "img2img_pipeline"]:
        if key in mixed_local_bundle and mixed_local_bundle[key] is not None:
            pipe = mixed_local_bundle[key]
            break

    if pipe is not None and hasattr(pipe, "to"):
        pipe.to(device)

    if pipe is not None and hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    local_outputs = []
    recycle_passes = max(1, int(recycle_passes))

    for zone in zones:
        crop = zone["crop"]
        prompt = zone["prompt"]
        zone_negative_prompt = zone.get("negative_prompt", negative_prompt)
        zone_strength = float(zone.get("strength", strength))
        zone_guidance_scale = float(zone.get("guidance_scale", guidance_scale))
        zone_num_inference_steps = int(zone.get("num_inference_steps", num_inference_steps))

        recycle_zone_strength = float(
            zone.get(
                "recycle_strength",
                zone_strength if recycle_strength is None else recycle_strength,
            )
        )
        recycle_zone_guidance_scale = float(
            zone.get(
                "recycle_guidance_scale",
                zone_guidance_scale if recycle_guidance_scale is None else recycle_guidance_scale,
            )
        )
        recycle_zone_num_inference_steps = int(
            zone.get(
                "recycle_num_inference_steps",
                zone_num_inference_steps if recycle_num_inference_steps is None else recycle_num_inference_steps,
            )
        )

        if pipe is None:
            aged_crop = _img2img_tensor_bundle_safe(
                bundle=mixed_local_bundle,
                image=crop,
                prompt=prompt,
                negative_prompt=zone_negative_prompt,
                device=device,
                strength=zone_strength,
                guidance_scale=zone_guidance_scale,
                num_inference_steps=zone_num_inference_steps,
                generator=generator,
            )
        else:
            aged_crop = _call_img2img_pipe_safe(
                pipe=pipe,
                image=crop,
                prompt=prompt,
                negative_prompt=zone_negative_prompt,
                strength=zone_strength,
                guidance_scale=zone_guidance_scale,
                num_inference_steps=zone_num_inference_steps,
                generator=generator,
            )

        for _ in range(recycle_passes - 1):
            if pipe is None:
                aged_crop = _img2img_tensor_bundle_safe(
                    bundle=mixed_local_bundle,
                    image=aged_crop,
                    prompt=prompt,
                    negative_prompt=zone_negative_prompt,
                    device=device,
                    strength=recycle_zone_strength,
                    guidance_scale=recycle_zone_guidance_scale,
                    num_inference_steps=recycle_zone_num_inference_steps,
                    generator=generator,
                )
            else:
                aged_crop = _call_img2img_pipe_safe(
                    pipe=pipe,
                    image=aged_crop,
                    prompt=prompt,
                    negative_prompt=zone_negative_prompt,
                    strength=recycle_zone_strength,
                    guidance_scale=recycle_zone_guidance_scale,
                    num_inference_steps=recycle_zone_num_inference_steps,
                    generator=generator,
                )

        local_outputs.append({
            "zone_name": zone.get("zone_name", None),
            "box_index": zone.get("box_index", None),
            "aged_crop": aged_crop,
            "bbox": zone["bbox"],
            "mask": zone.get("mask", None),
            "prompt": prompt,
        })

    return local_outputs


def _call_sample_forward_with_supported_kwargs(fn: Callable, kwargs: Dict[str, Any]):
    """
    Calls custom sampling hooks without forcing new optional kwargs on older hooks.
    """
    signature = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return fn(**kwargs)

    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return fn(**filtered_kwargs)


# ============================================================
# Saving monitoring images
# ============================================================

def save_monitoring_fusion_outputs(
    fusion_out: Dict[str, Any],
    output_dir: Union[str, Path],
    run_name: str,
    epoch: int,
    sample_id: str = "sample",
    save_grid: bool = True,
) -> Dict[str, Path]:
    """
    Saves x_orig, x_global, x_coarse, x_blend, x_final, optional x_refined,
    and residual/debug visualizations.

    Requires fuse_global_local_outputs(..., return_pil=True).
    """
    output_dir = Path(output_dir) / f"epoch_{int(epoch) + 1:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_id = str(sample_id).replace("/", "_").replace("\\", "_").replace(" ", "_")
    prefix = f"{run_name}_e{epoch:03d}_{safe_id}"

    pil_pack = fusion_out.get("pil", None)

    if pil_pack is None:
        raise ValueError("fusion_out must contain key 'pil'. Use return_pil=True.")

    paths = {}

    keys = [
        "x_orig",
        "x_global",
        "x_coarse",
        "x_blend",
        "x_final",
        "x_refined",
        "residual_raw_vis",
        "residual_low_vis",
        "local_union_mask_vis",
        "alpha_map_vis",
    ]

    for key in keys:
        if key in pil_pack and pil_pack[key] is not None:
            path = output_dir / f"{prefix}_{key}.png"
            pil_pack[key].save(path)
            paths[key] = path

    if save_grid:
        try:
            from PIL import Image, ImageDraw

            grid_keys = ["x_orig", "x_global", "x_coarse", "x_blend", "x_final"]
            if "x_refined" in pil_pack:
                grid_keys.append("x_refined")
            images = [pil_pack[k].convert("RGB") for k in grid_keys if k in pil_pack and pil_pack[k] is not None]

            if len(images) > 0:
                w, h = images[0].size
                label_h = 28
                grid = Image.new("RGB", (w * len(images), h + label_h), "white")
                draw = ImageDraw.Draw(grid)

                for i, (k, img) in enumerate(zip(grid_keys, images)):
                    if img.size != (w, h):
                        img = img.resize((w, h))
                    grid.paste(img, (i * w, label_h))
                    draw.text((i * w + 8, 8), k, fill=(0, 0, 0))

                grid_path = output_dir / f"{prefix}_grid.png"
                grid.save(grid_path)
                paths["grid"] = grid_path

        except Exception as e:
            print(f"└─ [WARN] Could not save monitoring grid: {e}")

    return paths


# ============================================================
# Reconstruction sampling callback
# ============================================================

def run_deterministic_training_reconstruction_sample(
    *,
    mixed_global_bundle: Dict[str, Any],
    mixed_local_bundle: Dict[str, Any],

    sampling_loader_global,
    sampling_loader_local,

    device: torch.device,
    run_name: str,
    epoch: int,
    output_dir: Union[str, Path],

    # Optional custom forward functions.
    sample_global_forward_fn: Optional[Callable] = None,
    sample_local_forward_fn: Optional[Callable] = None,

    # Global sampling params.
    sample_global_strength: float = 0.30,
    sample_global_guidance_scale: float = 5.0,
    sample_global_num_inference_steps: int = 35,
    sample_global_negative_prompt: Optional[str] = None,

    # Local sampling params.
    sample_local_strength: float = 0.20,
    sample_local_guidance_scale: float = 0.8,
    sample_local_num_inference_steps: int = 40,
    sample_local_negative_prompt: Optional[str] = None,
    sample_local_recycle_passes: int = 1,
    sample_local_recycle_strength: Optional[float] = None,
    sample_local_recycle_guidance_scale: Optional[float] = None,
    sample_local_recycle_num_inference_steps: Optional[int] = None,

    # Deterministic fusion params.
    residual_alpha: float = 0.35,
    residual_sigma: float = 9.0,
    use_face_mask: bool = True,
    face_mask_blur_sigma: float = 3.0,
    local_insert_alpha: float = 1.0,
    local_mask_blur_sigma: float = 5.0,
    color_match: bool = True,
    color_match_strength: float = 0.75,

    seed: Optional[int] = None,
    save_grid: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs one deterministic reconstruction sample for monitoring.

    Important:
        - Uses both trained branches in eval mode.
        - Uses torch.inference_mode().
        - Does NOT use ScoreNet.
        - Does NOT use refiner/fusion_bundle.
        - Does NOT move optimizer states to GPU.
    """
    if sample_global_forward_fn is None:
        sample_global_forward_fn = default_sample_global_forward

    if sample_local_forward_fn is None:
        sample_local_forward_fn = default_sample_local_forward

    global_batch = parse_sampling_global_batch(_first_batch(sampling_loader_global))
    local_zones = parse_sampling_local_batch(_first_batch(sampling_loader_local))

    sample_id = global_batch["sample_id"]
    x_orig = global_batch["x_orig"]
    global_prompt = global_batch["global_prompt"]
    face_mask = global_batch["face_mask"]

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + int(epoch))

    if verbose:
        print_section(f"Deterministic reconstruction sample | epoch={epoch}")
        print("Sample id      :", sample_id)
        print("Global prompt  :", global_prompt)
        print("Local zones    :", len(local_zones))
        print("Local recycling:", max(1, int(sample_local_recycle_passes)), "pass(es)")
        print("Fusion         : deterministic only")
        print("Refiner        : disabled")

    with torch.inference_mode():
        # --------------------------------------------------------
        # 1. Global branch forward.
        # --------------------------------------------------------
        move_bundle_modules_only_to_device(mixed_global_bundle, device, eval_mode=True)

        x_global = sample_global_forward_fn(
            mixed_global_bundle=mixed_global_bundle,
            x_orig=x_orig,
            global_prompt=global_prompt,
            device=device,
            strength=sample_global_strength,
            guidance_scale=sample_global_guidance_scale,
            num_inference_steps=sample_global_num_inference_steps,
            negative_prompt=sample_global_negative_prompt,
            generator=generator,
        )

        offload_bundle_modules_only(mixed_global_bundle, label="global sampling")

        # --------------------------------------------------------
        # 2. Local branch forward.
        # --------------------------------------------------------
        move_bundle_modules_only_to_device(mixed_local_bundle, device, eval_mode=True)

        local_outputs = _call_sample_forward_with_supported_kwargs(
            sample_local_forward_fn,
            {
                "mixed_local_bundle": mixed_local_bundle,
                "zones": local_zones,
                "device": device,
                "strength": sample_local_strength,
                "guidance_scale": sample_local_guidance_scale,
                "num_inference_steps": sample_local_num_inference_steps,
                "negative_prompt": sample_local_negative_prompt,
                "recycle_passes": sample_local_recycle_passes,
                "recycle_strength": sample_local_recycle_strength,
                "recycle_guidance_scale": sample_local_recycle_guidance_scale,
                "recycle_num_inference_steps": sample_local_recycle_num_inference_steps,
                "generator": generator,
            },
        )

        offload_bundle_modules_only(mixed_local_bundle, label="local sampling")

        # --------------------------------------------------------
        # 3. Deterministic fusion.
        # --------------------------------------------------------
        if fuse_global_local_outputs is None:
            raise ImportError("fuse_global_local_outputs must be available from src.inference.global_local_fusion for monitoring fusion.")

        fusion_out = fuse_global_local_outputs(
            x_orig=x_orig,
            x_global=x_global,
            local_outputs=local_outputs,
            face_mask=face_mask,

            # No refiner during training monitoring.
            fusion_bundle=None,

            residual_alpha=residual_alpha,
            residual_sigma=residual_sigma,
            use_face_mask=use_face_mask,
            face_mask_blur_sigma=face_mask_blur_sigma,

            local_insert_alpha=local_insert_alpha,
            local_mask_blur_sigma=local_mask_blur_sigma,

            color_match=color_match,
            color_match_strength=color_match_strength,

            device=device,
            seed=seed,
            return_pil=True,
            verbose=False,
        )

    paths = save_monitoring_fusion_outputs(
        fusion_out=fusion_out,
        output_dir=output_dir,
        run_name=run_name,
        epoch=epoch,
        sample_id=sample_id,
        save_grid=save_grid,
    )

    hard_cuda_cleanup(label=f"after reconstruction sample epoch={epoch}", reset_peak=True)

    if verbose:
        print("Saved monitoring images:")
        for k, p in paths.items():
            print(f"└─ [{k}] {p}")

    return {
        "epoch": int(epoch),
        "sample_id": sample_id,
        "paths": paths,
        "fusion_out": fusion_out,
    }
