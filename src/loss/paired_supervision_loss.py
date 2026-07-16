"""Optional supervised denoising on real longitudinal aging pairs.

This loss deliberately avoids L1/LPIPS between photographs from different
years: FG-NET and AgeDB pairs are identity/age matched, not pixel aligned.
The reliable signal is therefore exact-age conditioned denoising of both real
endpoints. A latent delta term is available only as an opt-in experiment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairedDiffusionSupervisionLoss(nn.Module):
    """Branch-agnostic diffusion loss for paired full faces or paired crops."""

    def __init__(
        self,
        bundle: Dict[str, Any],
        *,
        lambda_target_diff: float = 1.0,
        lambda_source_diff: float = 0.25,
        lambda_latent_delta: float = 0.0,
        use_min_snr: bool = True,
        min_snr_gamma: float = 5.0,
        timestep_min: int = 0,
        timestep_max: Optional[int] = None,
        device: Optional[str] = None,
        unet_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.bundle = bundle
        self.vae = bundle["vae"]
        self.unet = bundle["unet"]
        self.tokenizer = bundle["tokenizer"]
        self.text_encoder = bundle["text_encoder"]
        self.scheduler = bundle["scheduler_train"]

        self.device = torch.device(device or next(self.unet.parameters()).device)
        self.unet_dtype = unet_dtype or next(self.unet.parameters()).dtype
        self.lambda_target_diff = float(lambda_target_diff)
        self.lambda_source_diff = float(lambda_source_diff)
        self.lambda_latent_delta = float(lambda_latent_delta)
        self.use_min_snr = bool(use_min_snr)
        self.min_snr_gamma = float(min_snr_gamma)
        self.timestep_min = int(timestep_min)
        max_train_timestep = int(self.scheduler.config.num_train_timesteps) - 1
        self.timestep_max = max_train_timestep if timestep_max is None else int(timestep_max)
        if not 0 <= self.timestep_min <= self.timestep_max <= max_train_timestep:
            raise ValueError(
                f"Invalid timestep range [{self.timestep_min}, {self.timestep_max}] "
                f"for scheduler maximum {max_train_timestep}"
            )
        if min(self.lambda_target_diff, self.lambda_source_diff, self.lambda_latent_delta) < 0:
            raise ValueError("Paired loss weights must be non-negative")
        if self.lambda_target_diff + self.lambda_source_diff + self.lambda_latent_delta <= 0:
            raise ValueError("At least one paired loss weight must be positive")

        self.vae.eval()
        self.text_encoder.eval()
        for module in (self.vae, self.text_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return self.text_encoder(tokens.input_ids.to(self.device), return_dict=True).last_hidden_state

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(device=self.device, dtype=self.unet_dtype)
        posterior = self.vae.encode(images, return_dict=True).latent_dist
        return (posterior.mean * self.vae.config.scaling_factor).to(dtype=self.unet_dtype)

    def _predict_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        prompts: List[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noisy = self.scheduler.add_noise(latents, noise, timesteps).to(self.unet_dtype)
        hidden = self.encode_prompts(prompts).to(self.unet_dtype)
        prediction = self.unet(
            noisy,
            timesteps,
            encoder_hidden_states=hidden,
            return_dict=True,
        ).sample
        return noisy, prediction

    def _min_snr_weight(self, timesteps: torch.Tensor) -> torch.Tensor:
        alphas = self.scheduler.alphas_cumprod.to(self.device, dtype=torch.float32)
        alpha = alphas[timesteps].clamp(1e-8, 1.0 - 1e-8)
        snr = alpha / (1.0 - alpha)
        return (torch.clamp(snr, max=self.min_snr_gamma) / snr).view(-1)

    def _diffusion_loss(
        self,
        prediction: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        per_sample = (prediction.float() - noise.float()).pow(2).mean(dim=(1, 2, 3))
        if self.use_min_snr:
            per_sample = per_sample * self._min_snr_weight(timesteps)
        return per_sample.mean()

    def _predict_x0(
        self,
        noisy: torch.Tensor,
        prediction: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alphas = self.scheduler.alphas_cumprod.to(self.device, dtype=noisy.dtype)
        alpha = alphas[timesteps].view(-1, 1, 1, 1)
        return (noisy - torch.sqrt(1.0 - alpha) * prediction) / torch.sqrt(alpha).clamp_min(1e-8)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        required = {"target_pixel_values", "target_prompt"}
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"Paired batch is missing keys: {sorted(missing)}")

        target_images = batch["target_pixel_values"]
        target_prompts = list(batch["target_prompt"])
        target_latents = self.encode_images(target_images)
        batch_size = target_latents.shape[0]
        timesteps = torch.randint(
            self.timestep_min,
            self.timestep_max + 1,
            (batch_size,),
            device=self.device,
        ).long()
        # Sharing t/noise across endpoints makes optional pair comparisons less noisy.
        noise = torch.randn_like(target_latents)
        target_noisy, target_prediction = self._predict_noise(
            target_latents, noise, timesteps, target_prompts
        )
        loss_target = self._diffusion_loss(target_prediction, noise, timesteps)

        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_source = zero
        loss_delta = zero
        source_prediction = None
        source_noisy = None
        source_latents = None

        needs_source = self.lambda_source_diff > 0 or self.lambda_latent_delta > 0
        if needs_source:
            for key in ("source_pixel_values", "source_prompt"):
                if key not in batch:
                    raise KeyError(f"Paired source supervision requires batch[{key!r}]")
            source_latents = self.encode_images(batch["source_pixel_values"])
            source_noisy, source_prediction = self._predict_noise(
                source_latents, noise, timesteps, list(batch["source_prompt"])
            )
            loss_source = self._diffusion_loss(source_prediction, noise, timesteps)

        if self.lambda_latent_delta > 0:
            predicted_delta = self._predict_x0(
                target_noisy, target_prediction, timesteps
            ) - self._predict_x0(source_noisy, source_prediction, timesteps)
            real_delta = target_latents.float() - source_latents.float()
            loss_delta = (1.0 - F.cosine_similarity(
                predicted_delta.float().flatten(1),
                real_delta.flatten(1),
                dim=1,
                eps=1e-8,
            )).mean()

        total = (
            self.lambda_target_diff * loss_target
            + self.lambda_source_diff * loss_source
            + self.lambda_latent_delta * loss_delta
        )
        return {
            "loss": total,
            "loss_target_diff": loss_target.detach(),
            "loss_source_diff": loss_source.detach(),
            "loss_latent_delta": loss_delta.detach(),
            "timestep_mean": timesteps.float().mean().detach(),
            "age_gap_mean": (
                batch["age_gap"].float().mean().detach()
                if torch.is_tensor(batch.get("age_gap")) else zero
            ),
        }
