from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch

from src.inference.image_tensor_utils import image_to_tensor01


class InversionUnavailableError(RuntimeError):
    """Raised when the requested inversion strategy cannot run safely."""


@dataclass(frozen=True)
class InversionConfig:
    enabled: bool = False
    method: str = "ddim"
    num_steps: int = 40
    strength: float = 0.45
    inversion_guidance_scale: float = 1.0
    edit_guidance_scale: Optional[float] = None
    source_score_mode: str = "auto"
    source_prompt_fallback: str = "zone"
    negative_prompt_during_inversion: bool = False
    return_source_reconstruction: bool = False
    cache_enabled: bool = True
    post_edit_img2img_passes: int = 0
    fallback_to_img2img: bool = True

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "InversionConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value

        known = cls.__dataclass_fields__
        config = cls(**{key: val for key, val in dict(value).items() if key in known})
        config.validate()
        return config

    def validate(self) -> None:
        if self.method != "ddim":
            raise ValueError(
                f"Unsupported inversion method={self.method!r}. The current baseline supports only 'ddim'."
            )
        if int(self.num_steps) < 1:
            raise ValueError("inversion.num_steps must be >= 1.")
        if not 0.0 <= float(self.strength) <= 1.0:
            raise ValueError("inversion.strength must be in [0, 1].")
        if float(self.inversion_guidance_scale) < 0.0:
            raise ValueError("inversion.inversion_guidance_scale must be >= 0.")
        if self.edit_guidance_scale is not None and float(self.edit_guidance_scale) < 0.0:
            raise ValueError("inversion.edit_guidance_scale must be >= 0 or null.")
        if int(self.post_edit_img2img_passes) < 0:
            raise ValueError("inversion.post_edit_img2img_passes must be >= 0.")
        if str(self.source_score_mode).lower().strip() not in {
            "auto",
            "metadata",
            "scorenet",
            "zone",
        }:
            raise ValueError(
                "inversion.source_score_mode must be one of: auto, metadata, scorenet, zone."
            )
        if self.source_prompt_fallback != "zone":
            raise ValueError("The DDIM baseline supports source_prompt_fallback='zone' only.")


@dataclass
class InversionEditResult:
    image: torch.Tensor
    reconstruction: Optional[torch.Tensor] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    inverted_latent: Optional[torch.Tensor] = None


def _module_dtype(module: torch.nn.Module, fallback: torch.dtype = torch.float32) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except (StopIteration, AttributeError):
        return fallback


def _module_device(module: torch.nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(module.parameters()).device
    except (StopIteration, AttributeError):
        return fallback


def _scheduler_config_value(scheduler: Any, key: str, default: Any = None) -> Any:
    config = getattr(scheduler, "config", None)
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _set_timesteps(scheduler: Any, num_steps: int, device: torch.device) -> None:
    try:
        scheduler.set_timesteps(num_steps, device=device)
    except TypeError:
        scheduler.set_timesteps(num_steps)
        if torch.is_tensor(scheduler.timesteps):
            scheduler.timesteps = scheduler.timesteps.to(device)


def _scheduler_step_sample(output: Any) -> torch.Tensor:
    if hasattr(output, "prev_sample"):
        return output.prev_sample
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError("Scheduler step output must expose 'prev_sample' or be tuple-like.")


def _normalise_prompt_batch(prompt: Union[str, Sequence[str]], batch_size: int) -> list[str]:
    if isinstance(prompt, str):
        return [prompt] * batch_size
    prompts = [str(item) for item in prompt]
    if len(prompts) != batch_size:
        raise ValueError(f"Expected {batch_size} prompts, received {len(prompts)}.")
    return prompts


class DDIMInversionEditor:
    """Deterministic partial DDIM inversion using an already-loaded SD bundle.

    The editor deliberately holds references to the bundle's VAE, UNet and text
    encoder. It never reloads a model, so an injected LoRA/DoRA remains active in
    both the inverse and target-denoising trajectories.
    """

    def __init__(
        self,
        *,
        vae: torch.nn.Module,
        unet: torch.nn.Module,
        tokenizer: Any,
        text_encoder: torch.nn.Module,
        scheduler: Any,
        device: Union[str, torch.device],
        config: Optional[Union[InversionConfig, Mapping[str, Any]]] = None,
        model_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        adapter_type: Optional[str] = None,
        scheduler_classes: Optional[Tuple[type, type]] = None,
    ) -> None:
        self.vae = vae
        self.unet = unet
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.base_scheduler = scheduler
        self.device = torch.device(device)
        self.config = (
            config if isinstance(config, InversionConfig) else InversionConfig.from_mapping(config)
        )
        self.config.validate()
        self.model_id = model_id
        self.checkpoint_id = checkpoint_id
        self.adapter_type = adapter_type
        self._scheduler_classes = scheduler_classes
        self._cache: Dict[str, torch.Tensor] = {}

    @classmethod
    def from_bundle(
        cls,
        bundle: Mapping[str, Any],
        *,
        device: Union[str, torch.device],
        config: Optional[Union[InversionConfig, Mapping[str, Any]]] = None,
        scheduler_classes: Optional[Tuple[type, type]] = None,
    ) -> "DDIMInversionEditor":
        required = ("vae", "unet", "tokenizer", "text_encoder")
        missing = [key for key in required if bundle.get(key) is None]
        scheduler = bundle.get("scheduler_infer") or bundle.get("scheduler_train")
        if scheduler is None:
            missing.append("scheduler_infer/scheduler_train")
        if missing:
            raise KeyError(f"Bundle cannot run DDIM inversion. Missing: {missing}")

        editor = cls(
            vae=bundle["vae"],
            unet=bundle["unet"],
            tokenizer=bundle["tokenizer"],
            text_encoder=bundle["text_encoder"],
            scheduler=scheduler,
            device=device,
            config=config,
            model_id=bundle.get("model_id") or bundle.get("name"),
            checkpoint_id=bundle.get("inference_checkpoint_id"),
            adapter_type=bundle.get("adapter_type"),
            scheduler_classes=scheduler_classes,
        )
        if editor.unet is not bundle["unet"]:
            raise AssertionError("DDIM inversion must reuse the bundle UNet with its active adapters.")
        return editor

    def clear_cache(self) -> None:
        self._cache.clear()

    def _make_schedulers(self) -> Tuple[Any, Any]:
        if self._scheduler_classes is None:
            try:
                from diffusers import DDIMInverseScheduler, DDIMScheduler
            except (ImportError, AttributeError) as exc:
                raise InversionUnavailableError(
                    "DDIM inversion requires a diffusers version exposing "
                    "DDIMInverseScheduler and DDIMScheduler."
                ) from exc
            forward_cls, inverse_cls = DDIMScheduler, DDIMInverseScheduler
        else:
            forward_cls, inverse_cls = self._scheduler_classes

        try:
            forward = forward_cls.from_config(self.base_scheduler.config)
            inverse = inverse_cls.from_config(self.base_scheduler.config)
        except Exception as exc:
            raise InversionUnavailableError(
                "Could not create compatible DDIM forward/inverse schedulers from the bundle scheduler."
            ) from exc

        _set_timesteps(forward, int(self.config.num_steps), self.device)
        _set_timesteps(inverse, int(self.config.num_steps), self.device)
        return forward, inverse

    def _partial_timesteps(self, forward: Any, inverse: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if float(self.config.strength) <= 0.0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty

        effective_steps = min(
            int(self.config.num_steps),
            max(1, int(int(self.config.num_steps) * float(self.config.strength))),
        )
        inverse_steps = torch.as_tensor(inverse.timesteps, device=self.device)[:effective_steps]
        forward_steps = torch.as_tensor(forward.timesteps, device=self.device)[-effective_steps:]

        if inverse_steps.numel() != forward_steps.numel() or not torch.equal(
            inverse_steps, torch.flip(forward_steps, dims=[0])
        ):
            raise InversionUnavailableError(
                "DDIM forward and inverse timestep schedules are not symmetric for this scheduler config."
            )
        return inverse_steps, forward_steps

    def encode_image(self, image: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        vae_dtype = _module_dtype(self.vae)
        image_01 = image_to_tensor01(image, device=self.device, dtype=vae_dtype)
        image_m11 = (image_01 * 2.0 - 1.0).clamp(-1.0, 1.0)
        latent_dist = self.vae.encode(image_m11).latent_dist
        latents = latent_dist.mean * float(self.vae.config.scaling_factor)
        return latents.to(device=self.device, dtype=_module_dtype(self.unet)), image_m11

    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        vae_dtype = _module_dtype(self.vae)
        decoded = self.vae.decode(
            (latents / float(self.vae.config.scaling_factor)).to(dtype=vae_dtype),
            return_dict=True,
        ).sample
        return decoded.clamp(-1.0, 1.0).detach().float()

    def encode_prompt(
        self,
        prompt: Union[str, Sequence[str]],
        *,
        negative_prompt: Optional[Union[str, Sequence[str]]] = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        cond = _normalise_prompt_batch(prompt, batch_size)
        uncond = _normalise_prompt_batch(negative_prompt or "", batch_size)
        text_inputs = self.tokenizer(
            uncond + cond,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = getattr(text_inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        return hidden.to(device=self.device, dtype=_module_dtype(self.unet))

    def _predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds: torch.Tensor,
        guidance_scale: float,
        scheduler: Any,
    ) -> torch.Tensor:
        model_input = torch.cat([latents, latents], dim=0)
        model_input = scheduler.scale_model_input(model_input, timestep)
        noise = self.unet(
            model_input.to(dtype=_module_dtype(self.unet)),
            timestep,
            encoder_hidden_states=text_embeds,
            return_dict=True,
        ).sample
        noise_uncond, noise_cond = noise.chunk(2)
        return noise_uncond + float(guidance_scale) * (noise_cond - noise_uncond)

    def _cache_key(
        self,
        latents: torch.Tensor,
        source_prompt: Union[str, Sequence[str]],
        source_negative_prompt: Optional[Union[str, Sequence[str]]],
        cache_key: Optional[str],
    ) -> str:
        digest = hashlib.sha256()
        if cache_key is None:
            tensor = latents.detach().float().cpu().contiguous()
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(str(cache_key).encode("utf-8"))
        digest.update(str(source_prompt).encode("utf-8"))
        digest.update(str(source_negative_prompt).encode("utf-8"))
        digest.update(str(self.model_id).encode("utf-8"))
        digest.update(str(self.checkpoint_id).encode("utf-8"))
        digest.update(str(self.adapter_type).encode("utf-8"))
        digest.update(repr(getattr(self.base_scheduler, "config", None)).encode("utf-8"))
        digest.update(str(self.config.num_steps).encode("ascii"))
        digest.update(str(self.config.strength).encode("ascii"))
        digest.update(str(self.config.inversion_guidance_scale).encode("ascii"))
        return digest.hexdigest()

    def invert(
        self,
        latents: torch.Tensor,
        *,
        source_prompt: Union[str, Sequence[str]],
        negative_prompt: Optional[Union[str, Sequence[str]]] = None,
        cache_key: Optional[str] = None,
        schedulers: Optional[Tuple[Any, Any]] = None,
    ) -> Tuple[torch.Tensor, Any, torch.Tensor, bool]:
        forward, inverse = schedulers or self._make_schedulers()
        inverse_steps, forward_steps = self._partial_timesteps(forward, inverse)
        if inverse_steps.numel() == 0:
            return latents.detach().clone(), forward, forward_steps, False

        resolved_key = self._cache_key(latents, source_prompt, negative_prompt, cache_key)
        if self.config.cache_enabled and resolved_key in self._cache:
            return self._cache[resolved_key].clone(), forward, forward_steps, True

        embeds = self.encode_prompt(
            source_prompt,
            negative_prompt=negative_prompt,
            batch_size=latents.shape[0],
        )
        inverted = latents.clone()
        for timestep in inverse_steps:
            noise = self._predict_noise(
                inverted,
                timestep,
                embeds,
                self.config.inversion_guidance_scale,
                inverse,
            )
            inverted = _scheduler_step_sample(
                inverse.step(noise, timestep, inverted, return_dict=True)
            )

        inverted = inverted.detach()
        if self.config.cache_enabled:
            self._cache[resolved_key] = inverted.clone()
        return inverted, forward, forward_steps, False

    def denoise(
        self,
        inverted_latent: torch.Tensor,
        *,
        prompt: Union[str, Sequence[str]],
        negative_prompt: Optional[Union[str, Sequence[str]]],
        guidance_scale: float,
        scheduler: Any,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        embeds = self.encode_prompt(
            prompt,
            negative_prompt=negative_prompt,
            batch_size=inverted_latent.shape[0],
        )
        latents = inverted_latent.clone()
        for timestep in timesteps:
            noise = self._predict_noise(
                latents,
                timestep,
                embeds,
                guidance_scale,
                scheduler,
            )
            latents = _scheduler_step_sample(
                scheduler.step(noise, timestep, latents, return_dict=True)
            )
        return latents.detach()

    @staticmethod
    def reconstruction_metrics(source: torch.Tensor, reconstruction: torch.Tensor) -> Dict[str, float]:
        mse = float(torch.mean((source.float() - reconstruction.float()) ** 2).item())
        psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(4.0 / mse))
        return {"mse": mse, "psnr": psnr}

    @torch.inference_mode()
    def edit(
        self,
        image: Any,
        *,
        source_prompt: Union[str, Sequence[str]],
        target_prompt: Union[str, Sequence[str]],
        negative_prompt: Optional[Union[str, Sequence[str]]] = None,
        source_negative_prompt: Optional[Union[str, Sequence[str]]] = None,
        edit_guidance_scale: Optional[float] = None,
        cache_key: Optional[str] = None,
        return_inverted_latent: bool = False,
        diagnostics: Optional[Mapping[str, Any]] = None,
    ) -> InversionEditResult:
        latents, source_image = self.encode_image(image)
        source_negative = (
            source_negative_prompt if self.config.negative_prompt_during_inversion else ""
        )
        forward, inverse = self._make_schedulers()
        inverted, forward, forward_steps, cache_hit = self.invert(
            latents,
            source_prompt=source_prompt,
            negative_prompt=source_negative,
            cache_key=cache_key,
            schedulers=(forward, inverse),
        )
        guidance = (
            float(edit_guidance_scale)
            if edit_guidance_scale is not None
            else float(self.config.edit_guidance_scale)
            if self.config.edit_guidance_scale is not None
            else 1.0
        )
        target_latents = self.denoise(
            inverted,
            prompt=target_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance,
            scheduler=forward,
            timesteps=forward_steps,
        )
        edited = self.decode_latent(target_latents)

        reconstruction = None
        rec_metrics: Dict[str, float] = {}
        if self.config.return_source_reconstruction:
            rec_latents = self.denoise(
                inverted,
                prompt=source_prompt,
                negative_prompt=source_negative,
                guidance_scale=float(self.config.inversion_guidance_scale),
                scheduler=forward,
                timesteps=forward_steps,
            )
            reconstruction = self.decode_latent(rec_latents)
            rec_metrics = self.reconstruction_metrics(source_image.float(), reconstruction)

        metadata = dict(diagnostics or {})
        metadata.update(
            {
                "method": "ddim",
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "num_steps": int(self.config.num_steps),
                "effective_steps": int(forward_steps.numel()),
                "strength": float(self.config.strength),
                "inversion_guidance_scale": float(self.config.inversion_guidance_scale),
                "edit_guidance_scale": guidance,
                "scheduler_class": type(forward).__name__,
                "inverse_scheduler_class": type(inverse).__name__,
                "prediction_type": _scheduler_config_value(forward, "prediction_type"),
                "latent_shape": tuple(latents.shape),
                "dtype": str(latents.dtype),
                "device": str(latents.device),
                "cache_hit": cache_hit,
                "adapter_type": self.adapter_type,
                "reconstruction_metrics": rec_metrics or None,
            }
        )
        return InversionEditResult(
            image=edited,
            reconstruction=reconstruction,
            diagnostics=metadata,
            inverted_latent=inverted if return_inverted_latent else None,
        )


def evaluate_score_net(score_net: Optional[torch.nn.Module], image_m11: torch.Tensor) -> Optional[float]:
    if score_net is None:
        return None
    try:
        device = _module_device(score_net, image_m11.device)
        with torch.inference_mode():
            score = score_net(image_m11.to(device=device, dtype=torch.float32)).view(-1)[0]
        return float(score.detach().float().cpu().item())
    except Exception as exc:
        warnings.warn(f"ScoreNet diagnostic failed: {exc}", RuntimeWarning, stacklevel=2)
        return None


def _sdxl_encode_latent(pipe: Any, image: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    image_01 = image_to_tensor01(image, device=device, dtype=torch.float32)
    image_m11 = (image_01 * 2.0 - 1.0).clamp(-1.0, 1.0)
    vae = pipe.vae
    original_dtype = _module_dtype(vae)
    force_upcast = bool(getattr(vae.config, "force_upcast", False))
    if force_upcast:
        vae.to(dtype=torch.float32)
    encoded = vae.encode(image_m11.to(device=device, dtype=_module_dtype(vae))).latent_dist.mean
    if force_upcast:
        vae.to(dtype=original_dtype)

    unet_dtype = _module_dtype(pipe.unet)
    encoded = encoded.to(device=device, dtype=unet_dtype)
    latents_mean = getattr(vae.config, "latents_mean", None)
    latents_std = getattr(vae.config, "latents_std", None)
    if latents_mean is not None and latents_std is not None:
        mean = torch.as_tensor(latents_mean, device=device, dtype=unet_dtype).view(1, 4, 1, 1)
        std = torch.as_tensor(latents_std, device=device, dtype=unet_dtype).view(1, 4, 1, 1)
        encoded = (encoded - mean) * float(vae.config.scaling_factor) / std
    else:
        encoded = encoded * float(vae.config.scaling_factor)
    return encoded, image_01


def _sdxl_decode_latent(pipe: Any, latents: torch.Tensor) -> torch.Tensor:
    vae = pipe.vae
    original_dtype = _module_dtype(vae)
    force_upcast = bool(getattr(vae.config, "force_upcast", False))
    latents_mean = getattr(vae.config, "latents_mean", None)
    latents_std = getattr(vae.config, "latents_std", None)
    vae_latents = latents
    if latents_mean is not None and latents_std is not None:
        mean = torch.as_tensor(latents_mean, device=latents.device, dtype=latents.dtype).view(1, 4, 1, 1)
        std = torch.as_tensor(latents_std, device=latents.device, dtype=latents.dtype).view(1, 4, 1, 1)
        vae_latents = vae_latents * std / float(vae.config.scaling_factor) + mean
    else:
        vae_latents = vae_latents / float(vae.config.scaling_factor)
    if force_upcast:
        vae.to(dtype=torch.float32)
    decoded = vae.decode(vae_latents.to(dtype=_module_dtype(vae)), return_dict=True).sample
    if force_upcast:
        vae.to(dtype=original_dtype)
    return (decoded.float().clamp(-1.0, 1.0) + 1.0) / 2.0


def _sdxl_conditioning(
    pipe: Any,
    *,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    size_hw: Tuple[int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Use the pipeline's own dual-encoder and micro-conditioning helpers."""
    encoded = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
    )
    if not isinstance(encoded, (tuple, list)) or len(encoded) < 4:
        raise InversionUnavailableError(
            "The SDXL refiner encode_prompt contract did not return dual prompt/pooled embeddings."
        )
    prompt_embeds, negative_prompt_embeds, pooled, negative_pooled = encoded[:4]
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled = torch.cat([negative_pooled, pooled], dim=0)

    projection_dim = getattr(getattr(pipe, "text_encoder_2", None), "config", None)
    projection_dim = getattr(projection_dim, "projection_dim", pooled.shape[-1])
    height, width = size_hw
    add_time_ids, add_neg_time_ids = pipe._get_add_time_ids(
        original_size=(height, width),
        crops_coords_top_left=(0, 0),
        target_size=(height, width),
        aesthetic_score=6.0,
        negative_aesthetic_score=2.5,
        negative_original_size=(height, width),
        negative_crops_coords_top_left=(0, 0),
        negative_target_size=(height, width),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=projection_dim,
    )
    time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0).to(device)
    return prompt_embeds.to(device), {
        "text_embeds": pooled.to(device),
        "time_ids": time_ids,
    }


def _sdxl_predict_noise(
    pipe: Any,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    added_cond_kwargs: Dict[str, torch.Tensor],
    guidance_scale: float,
    scheduler: Any,
) -> torch.Tensor:
    model_input = torch.cat([latents, latents], dim=0)
    model_input = scheduler.scale_model_input(model_input, timestep)
    output = pipe.unet(
        model_input.to(dtype=_module_dtype(pipe.unet)),
        timestep,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=True,
    )
    noise = output.sample if hasattr(output, "sample") else output[0]
    uncond, cond = noise.chunk(2)
    return uncond + float(guidance_scale) * (cond - uncond)


@torch.inference_mode()
def edit_sdxl_refiner_with_ddim_inversion(
    *,
    pipe: Any,
    image: Any,
    source_prompt: str,
    target_prompt: str,
    negative_prompt: str,
    device: Union[str, torch.device],
    config: Union[InversionConfig, Mapping[str, Any]],
    edit_guidance_scale: float,
) -> InversionEditResult:
    """Experimental SDXL refiner inversion with complete XL conditioning.

    This path intentionally calls ``pipe.encode_prompt`` and
    ``pipe._get_add_time_ids`` so both text encoders, pooled embeddings and
    SDXL time ids match the loaded pipeline. Callers should retain img2img as a
    fallback because different/custom SDXL pipelines may expose a different
    internal contract.
    """
    config = config if isinstance(config, InversionConfig) else InversionConfig.from_mapping(config)
    config.validate()
    device = torch.device(device)
    try:
        from diffusers import DDIMInverseScheduler, DDIMScheduler
    except (ImportError, AttributeError) as exc:
        raise InversionUnavailableError(
            "SDXL DDIM inversion requires DDIMInverseScheduler in diffusers."
        ) from exc

    forward = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    _set_timesteps(forward, int(config.num_steps), device)
    _set_timesteps(inverse, int(config.num_steps), device)
    effective_steps = 0 if config.strength <= 0 else max(
        1, int(int(config.num_steps) * float(config.strength))
    )
    if effective_steps:
        inverse_steps = torch.as_tensor(inverse.timesteps, device=device)[:effective_steps]
        forward_steps = torch.as_tensor(forward.timesteps, device=device)[-effective_steps:]
    else:
        inverse_steps = torch.empty(0, device=device, dtype=torch.long)
        forward_steps = torch.empty(0, device=device, dtype=torch.long)
    if effective_steps and not torch.equal(inverse_steps, torch.flip(forward_steps, dims=[0])):
        raise InversionUnavailableError("SDXL DDIM forward/inverse timesteps are not symmetric.")

    latents, source_image_01 = _sdxl_encode_latent(pipe, image, device)
    size_hw = tuple(source_image_01.shape[-2:])
    source_negative = negative_prompt if config.negative_prompt_during_inversion else ""
    source_embeds, source_added = _sdxl_conditioning(
        pipe,
        prompt=source_prompt,
        negative_prompt=source_negative,
        guidance_scale=config.inversion_guidance_scale,
        size_hw=size_hw,
        device=device,
    )
    inverted = latents.clone()
    for timestep in inverse_steps:
        noise = _sdxl_predict_noise(
            pipe,
            inverted,
            timestep,
            source_embeds,
            source_added,
            config.inversion_guidance_scale,
            inverse,
        )
        inverted = _scheduler_step_sample(
            inverse.step(noise, timestep, inverted, return_dict=True)
        )

    target_embeds, target_added = _sdxl_conditioning(
        pipe,
        prompt=target_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=edit_guidance_scale,
        size_hw=size_hw,
        device=device,
    )
    edited_latents = inverted.clone()
    for timestep in forward_steps:
        noise = _sdxl_predict_noise(
            pipe,
            edited_latents,
            timestep,
            target_embeds,
            target_added,
            edit_guidance_scale,
            forward,
        )
        edited_latents = _scheduler_step_sample(
            forward.step(noise, timestep, edited_latents, return_dict=True)
        )
    edited = _sdxl_decode_latent(pipe, edited_latents).clamp(0.0, 1.0)

    reconstruction = None
    metrics = None
    if config.return_source_reconstruction:
        rec_latents = inverted.clone()
        for timestep in forward_steps:
            noise = _sdxl_predict_noise(
                pipe,
                rec_latents,
                timestep,
                source_embeds,
                source_added,
                config.inversion_guidance_scale,
                forward,
            )
            rec_latents = _scheduler_step_sample(
                forward.step(noise, timestep, rec_latents, return_dict=True)
            )
        reconstruction = _sdxl_decode_latent(pipe, rec_latents).clamp(0.0, 1.0)
        mse = float(torch.mean((source_image_01.float() - reconstruction.float()) ** 2).item())
        metrics = {
            "mse": mse,
            "psnr": float("inf") if mse == 0 else float(10.0 * math.log10(1.0 / mse)),
        }

    return InversionEditResult(
        image=edited,
        reconstruction=reconstruction,
        diagnostics={
            "method": "ddim",
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "num_steps": int(config.num_steps),
            "effective_steps": int(effective_steps),
            "strength": float(config.strength),
            "inversion_guidance_scale": float(config.inversion_guidance_scale),
            "edit_guidance_scale": float(edit_guidance_scale),
            "scheduler_class": type(forward).__name__,
            "inverse_scheduler_class": type(inverse).__name__,
            "prediction_type": _scheduler_config_value(forward, "prediction_type"),
            "latent_shape": tuple(latents.shape),
            "dtype": str(latents.dtype),
            "device": str(latents.device),
            "reconstruction_metrics": metrics,
            "status": "experimental",
        },
    )
