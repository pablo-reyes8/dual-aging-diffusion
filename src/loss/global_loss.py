# ============================================================
# GLOBAL AGING LOSS
#
# Memory-aware version.
#
# Main modes:
#   loss_mode="diff":
#       L_diff only.
#       1 UNet source forward.
#       No VAE decode.
#       No age/id/perceptual models.
#
#   loss_mode="semantic":
#       Semantic losses only.
#       1 UNet target forward.
#       1 VAE decode.
#       Age / delta-age / identity / perceptual can share the same generated image.
#
#   loss_mode="all":
#       L_diff + semantic losses.
#       2 UNet forwards.
#       Use only for debugging or large VRAM.
#
# Uses:
#   global_bundle:
#       - vae
#       - unet
#       - tokenizer
#       - text_encoder
#       - scheduler_train
#
#   global_loss_bundle:
#       - predict_age(...)
#       - identity_similarity(...)
#       - perceptual_distance(...) optional
# ============================================================

from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss.global_aux_bundle import *
from src.loss.ddim_generation import ddim_generate_images

class GlobalAgingLoss(nn.Module):
    def __init__(
        self,
        global_bundle: Dict[str, Any],
        global_loss_bundle: nn.Module,

        lambda_diff: float = 1.0,
        lambda_id: float = 0.5,
        lambda_age: float = 0.25,
        lambda_delta_age: float = 0.25,
        lambda_perc: float = 0.0,

        # Age losses are normalized to avoid huge gradients.
        # Example: 10 years error -> 0.10 loss if age_loss_scale=100.
        age_loss_scale: float = 100.0,

        # Timestep weighting for semantic losses.
        gamma_timestep: float = 1.0,

        # Full diffusion timestep range.
        timestep_min: int = 0,
        timestep_max: Optional[int] = None,

        # Semantic one-step x0 estimate timestep range.
        # Usually lower than full range for more stable decoded images.
        semantic_timestep_min: int = 20,
        semantic_timestep_max: int = 300,

        # Which semantic losses are active by default in semantic mode.
        # Forward can override this.
        default_semantic_components: Tuple[str, ...] = ("age", "delta_age", "id"),

        # ----------------------------------------------------------------
        # Semantic generation mode (Error 1 fix).
        #   "1_step_per_loss":
        #       Original LDLA-style single-step x0_hat estimate (cheap, blurry).
        #       When semantic_anchor_to_source=True (default), all semantic
        #       losses become RELATIVE: the source reference is reconstructed
        #       through the SAME single-step path (shared noise/timestep) so the
        #       VAE/one-step blur cancels out between source and target.
        #   "full_ddim":
        #       Short deterministic DDIM trajectory -> sharper edited image.
        #       The frozen auxiliaries see an in-domain (sharp) image, so the
        #       source reference stays the clean input. Use a low
        #       full_ddim_max_timestep (~120) so the edit is mild.
        # ----------------------------------------------------------------
        semantic_loss_mode: str = "1_step_per_loss",
        semantic_anchor_to_source: bool = True,
        full_ddim_num_steps: int = 10,
        full_ddim_max_timestep: int = 120,

        # Min-SNR-gamma reweighting of the diffusion (diff) loss.
        # Off by default to keep the baseline unchanged.
        use_min_snr: bool = False,
        min_snr_gamma: float = 5.0,

        device: Optional[str] = None,
        unet_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        valid_semantic_modes = {"1_step_per_loss", "full_ddim"}
        semantic_loss_mode = str(semantic_loss_mode).lower().strip()
        if semantic_loss_mode not in valid_semantic_modes:
            raise ValueError(
                f"Invalid semantic_loss_mode='{semantic_loss_mode}'. "
                f"Valid: {sorted(valid_semantic_modes)}"
            )

        self.semantic_loss_mode = semantic_loss_mode
        self.semantic_anchor_to_source = bool(semantic_anchor_to_source)
        self.full_ddim_num_steps = int(full_ddim_num_steps)
        self.full_ddim_max_timestep = int(full_ddim_max_timestep)
        self.use_min_snr = bool(use_min_snr)
        self.min_snr_gamma = float(min_snr_gamma)

        self.global_bundle = global_bundle
        self.global_loss_bundle = global_loss_bundle

        self.vae = global_bundle["vae"]
        self.unet = global_bundle["unet"]
        self.tokenizer = global_bundle["tokenizer"]
        self.text_encoder = global_bundle["text_encoder"]
        self.scheduler = global_bundle["scheduler_train"]

        if device is None:
            device = next(self.unet.parameters()).device
        self.device = torch.device(device)

        if unet_dtype is None:
            unet_dtype = next(self.unet.parameters()).dtype
        self.unet_dtype = unet_dtype

        self.lambda_diff = float(lambda_diff)
        self.lambda_id = float(lambda_id)
        self.lambda_age = float(lambda_age)
        self.lambda_delta_age = float(lambda_delta_age)
        self.lambda_perc = float(lambda_perc)

        self.age_loss_scale = float(age_loss_scale)
        self.gamma_timestep = float(gamma_timestep)

        self.timestep_min = int(timestep_min)

        if timestep_max is None:
            timestep_max = int(self.scheduler.config.num_train_timesteps) - 1

        self.timestep_max = int(timestep_max)

        self.semantic_timestep_min = int(semantic_timestep_min)
        self.semantic_timestep_max = int(semantic_timestep_max)

        if self.semantic_timestep_max > self.timestep_max:
            self.semantic_timestep_max = self.timestep_max

        self.default_semantic_components = tuple(default_semantic_components)

        valid_components = {"age", "delta_age", "id", "perc"}
        for comp in self.default_semantic_components:
            if comp not in valid_components:
                raise ValueError(
                    f"Invalid semantic component '{comp}'. "
                    f"Valid components: {sorted(valid_components)}"
                )

        # Freeze VAE and text encoder.
        self.vae.eval()
        self.text_encoder.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        # Aux bundle is frozen internally, but gradients to generated image
        # must remain enabled when calling generated-image forwards.
        self.global_loss_bundle.eval()

        # Availability checks.
        if self.lambda_age > 0.0 or self.lambda_delta_age > 0.0:
            if hasattr(self.global_loss_bundle, "has_age"):
                if not self.global_loss_bundle.has_age():
                    raise ValueError(
                        "lambda_age/lambda_delta_age > 0 but global_loss_bundle has no age estimator."
                    )

        if self.lambda_id > 0.0:
            if hasattr(self.global_loss_bundle, "has_identity"):
                if not self.global_loss_bundle.has_identity():
                    raise ValueError(
                        "lambda_id > 0 but global_loss_bundle has no identity encoder."
                    )

        if self.lambda_perc > 0.0:
            if hasattr(self.global_loss_bundle, "has_lpips"):
                if not self.global_loss_bundle.has_lpips():
                    raise ValueError(
                        "lambda_perc > 0 but global_loss_bundle has no LPIPS model. "
                        "Create GlobalLossAuxBundle(use_lpips=True) or set lambda_perc=0."
                    )

        print("\n========== GLOBAL AGING LOSS ==========")
        print("lambda_diff:", self.lambda_diff)
        print("lambda_id:", self.lambda_id)
        print("lambda_age:", self.lambda_age)
        print("lambda_delta_age:", self.lambda_delta_age)
        print("lambda_perc:", self.lambda_perc)
        print("age_loss_scale:", self.age_loss_scale)
        print("gamma_timestep:", self.gamma_timestep)
        print("timestep range:", self.timestep_min, self.timestep_max)
        print("semantic timestep range:", self.semantic_timestep_min, self.semantic_timestep_max)
        print("semantic_loss_mode:", self.semantic_loss_mode)
        print("semantic_anchor_to_source:", self.semantic_anchor_to_source)
        if self.semantic_loss_mode == "full_ddim":
            print("full_ddim_num_steps:", self.full_ddim_num_steps)
            print("full_ddim_max_timestep:", self.full_ddim_max_timestep)
        print("use_min_snr:", self.use_min_snr, "| min_snr_gamma:", self.min_snr_gamma)
        print("default_semantic_components:", self.default_semantic_components)
        print("device:", self.device)
        print("unet_dtype:", self.unet_dtype)

    # --------------------------------------------------------
    # Text conditioning
    # --------------------------------------------------------

    @torch.no_grad()
    def encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(self.device)

        attention_mask = None
        if hasattr(text_inputs, "attention_mask"):
            attention_mask = text_inputs.attention_mask.to(self.device)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        return outputs.last_hidden_state.to(
            device=self.device,
            dtype=self.unet_dtype,
        )

    # --------------------------------------------------------
    # VAE helpers
    # --------------------------------------------------------

    @torch.no_grad()
    def encode_images_to_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values:
            [B, 3, 512, 512] in [-1, 1]

        returns:
            latents [B, 4, 64, 64]
        """
        x = pixel_values.to(device=self.device, dtype=self.unet_dtype)

        latents = self.vae.encode(x).latent_dist.mean
        latents = latents * self.vae.config.scaling_factor

        return latents.detach().to(dtype=self.unet_dtype)

    def decode_latents_to_images(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents:
            [B, 4, 64, 64]

        returns:
            decoded images [B, 3, 512, 512] in approx [-1, 1]

        No torch.no_grad():
            semantic losses need gradients to flow from aux models
            -> decoded image -> latents -> UNet adapters.
        """
        latents_for_vae = latents / self.vae.config.scaling_factor
        latents_for_vae = latents_for_vae.to(
            device=self.device,
            dtype=self.unet_dtype,
        )

        decoded = self.vae.decode(
            latents_for_vae,
            return_dict=True,
        ).sample

        return decoded.clamp(-1.0, 1.0)

    # --------------------------------------------------------
    # Diffusion helpers
    # --------------------------------------------------------

    def sample_timesteps(
        self,
        batch_size: int,
        low: Optional[int] = None,
        high: Optional[int] = None,
    ) -> torch.Tensor:
        if low is None:
            low = self.timestep_min
        if high is None:
            high = self.timestep_max

        timesteps = torch.randint(
            low=int(low),
            high=int(high) + 1,
            size=(batch_size,),
            device=self.device,
        ).long()

        return timesteps

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        noisy_latents = self.scheduler.add_noise(
            latents,
            noise,
            timesteps,
        )
        return noisy_latents.to(dtype=self.unet_dtype)

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        noise_pred = self.unet(
            noisy_latents.to(dtype=self.unet_dtype),
            timesteps,
            encoder_hidden_states=encoder_hidden_states.to(dtype=self.unet_dtype),
            return_dict=True,
        ).sample

        return noise_pred

    def predict_x0_from_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        noise_pred: torch.Tensor) -> torch.Tensor:

        """
        DDPM epsilon parameterization:

            x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_pred)
                     / sqrt(alpha_bar_t)
        """
        alphas_cumprod = self.scheduler.alphas_cumprod.to(
            device=self.device,
            dtype=noisy_latents.dtype,
        )

        alpha_bar_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t).clamp_min(1e-8)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        x0_hat = (
            noisy_latents - sqrt_one_minus_alpha_bar_t * noise_pred
        ) / sqrt_alpha_bar_t

        return x0_hat

    def semantic_timestep_weight(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        w(t) = alpha_bar_t ** gamma_timestep

        Returns:
            [B]
        """
        alphas_cumprod = self.scheduler.alphas_cumprod.to(
            device=self.device,
            dtype=torch.float32,
        )

        alpha_bar_t = alphas_cumprod[timesteps].float()
        weights = alpha_bar_t.clamp(0.0, 1.0) ** self.gamma_timestep

        return weights.view(-1)

    # --------------------------------------------------------
    # Branches
    # --------------------------------------------------------

    def min_snr_weight(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Min-SNR-gamma weight for epsilon-prediction:

            SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
            w(t)   = min(SNR(t), gamma) / SNR(t)

        Returns [B].
        """
        alphas_cumprod = self.scheduler.alphas_cumprod.to(
            device=self.device,
            dtype=torch.float32,
        )
        ab = alphas_cumprod[timesteps].float().clamp(1e-8, 1.0 - 1e-8)
        snr = ab / (1.0 - ab)
        w = torch.clamp(snr, max=self.min_snr_gamma) / snr
        return w.view(-1)

    def diffusion_loss(
        self,
        z0: torch.Tensor,
        prompts: List[str]) -> Dict[str, torch.Tensor]:

        """
        L_diff with source/observed prompts.

        Optionally reweighted with Min-SNR-gamma (use_min_snr=True).
        """
        B = z0.shape[0]

        noise = torch.randn_like(z0)
        timesteps = self.sample_timesteps(B)
        zt = self.add_noise(z0, noise, timesteps)

        hidden = self.encode_prompts(prompts)

        noise_pred = self.predict_noise(
            noisy_latents=zt,
            timesteps=timesteps,
            encoder_hidden_states=hidden,
        )

        if self.use_min_snr:
            per_sample = (noise_pred.float() - noise.float()).pow(2).mean(dim=[1, 2, 3])
            weights = self.min_snr_weight(timesteps)
            loss_diff = (weights * per_sample).mean()
        else:
            loss_diff = F.mse_loss(
                noise_pred.float(),
                noise.float(),
                reduction="mean",
            )

        return {
            "loss_diff": loss_diff,
            "timesteps_diff": timesteps,
            "noise_pred_diff": noise_pred}

    def semantic_forward(
        self,
        z0: torch.Tensor,
        target_prompts: List[str],
        source_prompts: Optional[List[str]] = None,
        clean_source_images: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        """
        Target-conditioned generation for semantic losses, shared by:
            age / delta-age / identity / perceptual.

        Two modes (Error 1 fix):

          "1_step_per_loss":
            Single-step x0_hat estimate (blurry). If semantic_anchor_to_source
            is True and source_prompts are given, the SOURCE reference is
            reconstructed through the SAME single-step path with shared
            noise/timestep, so all semantic losses become RELATIVE and the
            one-step/VAE blur cancels between source and target.

          "full_ddim":
            Short deterministic DDIM trajectory -> sharper edited image. The
            generated image is in-domain for the frozen auxiliaries, so the
            source reference stays the clean input image.

        Returns a dict that always includes:
            x0_hat_images       : generated (edited) image, grad-carrying
            source_ref_images   : reference image in the SAME domain as the
                                  generated image (for id/delta/perc)
            semantic_weights    : per-sample timestep weights ([B])
        """
        B = z0.shape[0]

        if self.semantic_loss_mode == "full_ddim":
            # Sharp generated image; auxiliaries are in-domain so the clean
            # source is a valid reference.
            x0_hat_images = ddim_generate_images(
                self,
                z0=z0,
                prompts=target_prompts,
                num_steps=self.full_ddim_num_steps,
                max_timestep=self.full_ddim_max_timestep,
            )

            if clean_source_images is not None:
                source_ref_images = clean_source_images
            else:
                source_ref_images = self.decode_latents_to_images(z0).detach()

            weights = torch.ones(B, device=self.device, dtype=torch.float32)

            return {
                "timesteps_semantic": None,
                "semantic_weights": weights,
                "noise_pred_target": None,
                "x0_hat_latents": None,
                "x0_hat_images": x0_hat_images,
                "source_ref_images": source_ref_images,
            }

        # ---- "1_step_per_loss" ----
        noise = torch.randn_like(z0)

        timesteps = self.sample_timesteps(
            B,
            low=self.semantic_timestep_min,
            high=self.semantic_timestep_max,
        )

        zt = self.add_noise(z0, noise, timesteps)

        target_hidden = self.encode_prompts(target_prompts)

        noise_pred_target = self.predict_noise(
            noisy_latents=zt,
            timesteps=timesteps,
            encoder_hidden_states=target_hidden,
        )

        x0_hat_latents = self.predict_x0_from_noise(
            noisy_latents=zt,
            timesteps=timesteps,
            noise_pred=noise_pred_target,
        )

        x0_hat_images = self.decode_latents_to_images(x0_hat_latents)

        # Relative source reference: same noise + same timestep, source prompt.
        if self.semantic_anchor_to_source and source_prompts is not None:
            source_hidden = self.encode_prompts(source_prompts)
            with torch.no_grad():
                noise_pred_source = self.predict_noise(
                    noisy_latents=zt,
                    timesteps=timesteps,
                    encoder_hidden_states=source_hidden,
                )
                x0_hat_source_latents = self.predict_x0_from_noise(
                    noisy_latents=zt,
                    timesteps=timesteps,
                    noise_pred=noise_pred_source,
                )
                source_ref_images = self.decode_latents_to_images(
                    x0_hat_source_latents
                ).detach()
        elif clean_source_images is not None:
            source_ref_images = clean_source_images
        else:
            source_ref_images = self.decode_latents_to_images(z0).detach()

        weights = self.semantic_timestep_weight(timesteps)

        return {
            "timesteps_semantic": timesteps,
            "semantic_weights": weights,
            "noise_pred_target": noise_pred_target,
            "x0_hat_latents": x0_hat_latents,
            "x0_hat_images": x0_hat_images,
            "source_ref_images": source_ref_images,
        }

    # --------------------------------------------------------
    # Semantic losses
    # --------------------------------------------------------

    def compute_semantic_losses(
        self,
        source_images: torch.Tensor,
        generated_images: torch.Tensor,
        source_ages: torch.Tensor,
        target_ages: torch.Tensor,
        semantic_weights: torch.Tensor,
        semantic_components: Tuple[str, ...]) -> Dict[str, torch.Tensor]:

        """
        Computes selected semantic losses on the same generated image.

        semantic_components can contain:
            "age"
            "delta_age"
            "id"
            "perc"
        """

        valid_components = {"age", "delta_age", "id", "perc"}
        semantic_components = tuple(semantic_components)

        for comp in semantic_components:
            if comp not in valid_components:
                raise ValueError(
                    f"Invalid semantic component '{comp}'. "
                    f"Valid components: {sorted(valid_components)}"
                )

        B = source_images.shape[0]
        device = self.device

        source_ages = source_ages.to(device=device, dtype=torch.float32).view(-1)
        target_ages = target_ages.to(device=device, dtype=torch.float32).view(-1)
        semantic_weights = semantic_weights.to(device=device, dtype=torch.float32).view(-1)

        if source_ages.shape[0] != B:
            raise ValueError(
                f"source_ages shape {source_ages.shape} incompatible with B={B}"
            )

        if target_ages.shape[0] != B:
            raise ValueError(
                f"target_ages shape {target_ages.shape} incompatible with B={B}"
            )

        compute_age = ("age" in semantic_components) and (self.lambda_age > 0.0)
        compute_delta_age = ("delta_age" in semantic_components) and (self.lambda_delta_age > 0.0)
        compute_id = ("id" in semantic_components) and (self.lambda_id > 0.0)
        compute_perc = ("perc" in semantic_components) and (self.lambda_perc > 0.0)

        # Defaults.
        loss_id_per = torch.zeros(B, device=device)
        loss_age_per = torch.zeros(B, device=device)
        loss_delta_age_per = torch.zeros(B, device=device)
        loss_perc_per = torch.zeros(B, device=device)

        identity_similarity = None
        age_src_pred = None
        age_gen_pred = None
        lpips_score = None

        # ------------------------------
        # Identity loss
        # ------------------------------
        if compute_id:
            with torch.amp.autocast(device_type=device.type, enabled=False):
                identity_similarity = self.global_loss_bundle.identity_similarity(
                    source_images=source_images.float(),
                    generated_images=generated_images.float(),
                )

            loss_id_per = 1.0 - identity_similarity.float()

        # ------------------------------
        # Age / delta-age losses
        # ------------------------------
        if compute_age or compute_delta_age:
            # Generated prediction must keep grad to generated image.
            with torch.amp.autocast(device_type=device.type, enabled=False):
                age_gen_pred = self.global_loss_bundle.predict_age(
                    generated_images.float(),
                    grad_to_input=True,
                ).float().view(-1)

            if compute_age:
                loss_age_per = torch.abs(age_gen_pred - target_ages) / self.age_loss_scale

            if compute_delta_age:
                # Source prediction does not need grad.
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    age_src_pred = self.global_loss_bundle.predict_age(
                        source_images.float(),
                        grad_to_input=False,
                    ).float().view(-1)

                # Error 6 fix: anchor the desired delta to the SAME estimator
                # (ViT) source prediction, not to the CSV `source_ages`. Mixing
                # CSV ages with ViT predictions made the delta target biased by
                # the systematic offset between the two age models.
                age_src_anchor = age_src_pred.detach()
                target_delta = target_ages - age_src_anchor
                pred_delta = age_gen_pred - age_src_anchor

                loss_delta_age_per = (
                    torch.abs(pred_delta - target_delta) / self.age_loss_scale
                )

        # ------------------------------
        # Perceptual loss
        # ------------------------------
        if compute_perc:
            with torch.amp.autocast(device_type=device.type, enabled=False):
                lpips_score = self.global_loss_bundle.perceptual_distance(
                    source_images=source_images.float(),
                    generated_images=generated_images.float(),
                ).float().view(-1)

            loss_perc_per = lpips_score

        # ------------------------------
        # Apply timestep weighting
        # ------------------------------
        weighted_id = semantic_weights * loss_id_per
        weighted_age = semantic_weights * loss_age_per
        weighted_delta_age = semantic_weights * loss_delta_age_per
        weighted_perc = semantic_weights * loss_perc_per

        loss_id = weighted_id.mean()
        loss_age = weighted_age.mean()
        loss_delta_age = weighted_delta_age.mean()
        loss_perc = weighted_perc.mean()

        return {
            "loss_id": loss_id,
            "loss_age": loss_age,
            "loss_delta_age": loss_delta_age,
            "loss_perc": loss_perc,

            "loss_id_unweighted": loss_id_per.mean().detach(),
            "loss_age_unweighted": loss_age_per.mean().detach(),
            "loss_delta_age_unweighted": loss_delta_age_per.mean().detach(),
            "loss_perc_unweighted": loss_perc_per.mean().detach(),

            "identity_similarity": (
                identity_similarity.detach()
                if identity_similarity is not None
                else None
            ),
            "age_src_pred": (
                age_src_pred.detach()
                if age_src_pred is not None
                else None
            ),
            "age_gen_pred": (
                age_gen_pred.detach()
                if age_gen_pred is not None
                else None
            ),
            "lpips_score": (
                lpips_score.detach()
                if lpips_score is not None
                else None
            ),
        }

    # --------------------------------------------------------
    # Main forward
    # --------------------------------------------------------

    def forward(
        self,
        batch: Dict[str, Any],

        # Minimal mode set:
        #   "diff"     -> source denoising only
        #   "semantic" -> target semantic branch only
        #   "all"      -> diff + semantic, expensive
        loss_mode: str = "all",

        # Optional override for semantic components.
        # If None, uses self.default_semantic_components.
        semantic_components: Optional[Tuple[str, ...]] = None,

        # Source prompts for L_diff.
        # This is where CFG-style prompt dropout can be applied.
        source_prompts: Optional[List[str]] = None,

        # Target prompts and ages for semantic losses.
        target_prompts: Optional[List[str]] = None,
        target_ages: Optional[torch.Tensor] = None,

        # Optional override if batch key is not "age".
        source_ages: Optional[torch.Tensor] = None,

        return_generated_image: bool = False) -> Dict[str, Any]:


        loss_mode = str(loss_mode).lower().strip()

        valid_modes = {"diff", "semantic", "all"}

        if loss_mode not in valid_modes:
            raise ValueError(
                f"Invalid loss_mode='{loss_mode}'. "
                f"Valid modes are: {sorted(valid_modes)}")

        if semantic_components is None:
            semantic_components = self.default_semantic_components
        else:
            semantic_components = tuple(semantic_components)

        compute_diff = loss_mode in {"diff", "all"} and self.lambda_diff > 0.0
        compute_semantic = loss_mode in {"semantic", "all"}

        # Remove semantic components whose lambda is zero.
        active_semantic_components = []

        if "age" in semantic_components and self.lambda_age > 0.0:
            active_semantic_components.append("age")

        if "delta_age" in semantic_components and self.lambda_delta_age > 0.0:
            active_semantic_components.append("delta_age")

        if "id" in semantic_components and self.lambda_id > 0.0:
            active_semantic_components.append("id")

        if "perc" in semantic_components and self.lambda_perc > 0.0:
            active_semantic_components.append("perc")

        active_semantic_components = tuple(active_semantic_components)

        compute_semantic = compute_semantic and len(active_semantic_components) > 0

        # ----------------------------------------------------
        # Inputs
        # ----------------------------------------------------
        pixel_values = batch["pixel_values"].to(
            device=self.device,
            dtype=self.unet_dtype)

        B = pixel_values.shape[0]

        if source_prompts is None:
            source_prompts = batch["prompt"]

        if len(source_prompts) != B:
            raise ValueError(
                f"len(source_prompts)={len(source_prompts)} but batch size={B}.")

        # Source ages are only strictly necessary for semantic losses
        # involving delta-age, but they are useful for logging.
        if source_ages is None:
            if "age" in batch:
                source_ages = batch["age"]
            elif compute_semantic:
                raise ValueError(
                    "source_ages is None and batch does not contain batch['age']."
                )

        if source_ages is not None:
            source_ages = source_ages.to(
                device=self.device,
                dtype=torch.float32).view(-1)

            if source_ages.shape[0] != B:
                raise ValueError(
                    f"source_ages shape {source_ages.shape} but batch size={B}."
                )

        # Target checks only if semantic branch is active.
        if compute_semantic:
            if target_prompts is None:
                raise ValueError(
                    "target_prompts must be passed when loss_mode='semantic' or 'all' "
                    "with active semantic components."
                )

            if target_ages is None:
                raise ValueError(
                    "target_ages must be passed when loss_mode='semantic' or 'all' "
                    "with active semantic components."
                )

            if len(target_prompts) != B:
                raise ValueError(
                    f"len(target_prompts)={len(target_prompts)} but batch size={B}."
                )

            target_ages = target_ages.to(
                device=self.device,
                dtype=torch.float32,
            ).view(-1)

            if target_ages.shape[0] != B:
                raise ValueError(
                    f"target_ages shape {target_ages.shape} but batch size={B}."
                )

        else:
            target_ages = None

        # ----------------------------------------------------
        # Encode source image to z0
        # ----------------------------------------------------
        z0 = self.encode_images_to_latents(pixel_values)

        # ----------------------------------------------------
        # Defaults
        # ----------------------------------------------------
        loss_diff = torch.zeros((), device=self.device)
        loss_id = torch.zeros((), device=self.device)
        loss_age = torch.zeros((), device=self.device)
        loss_delta_age = torch.zeros((), device=self.device)
        loss_perc = torch.zeros((), device=self.device)

        timesteps_diff = None
        timesteps_semantic = None
        semantic_weights = None
        generated_images = None

        semantic_losses = {}

        # ----------------------------------------------------
        # Diff branch
        # ----------------------------------------------------
        if compute_diff:
            diff_out = self.diffusion_loss(
                z0=z0,
                prompts=source_prompts,)

            loss_diff = diff_out["loss_diff"]
            timesteps_diff = diff_out["timesteps_diff"]

        # ----------------------------------------------------
        # Semantic branch
        # ----------------------------------------------------
        if compute_semantic:
            sem_out = self.semantic_forward(
                z0=z0,
                target_prompts=target_prompts,
                source_prompts=source_prompts,
                clean_source_images=pixel_values.float(),)

            generated_images = sem_out["x0_hat_images"]
            semantic_weights = sem_out["semantic_weights"]
            timesteps_semantic = sem_out["timesteps_semantic"]

            # In "1_step_per_loss" + anchor mode this is the source reconstructed
            # through the same blurry one-step path; in "full_ddim" it is the
            # clean input. Using it as the reference makes id/delta/perc measure
            # the actual age-induced change, not the VAE/one-step blur.
            source_ref_images = sem_out["source_ref_images"]

            semantic_losses = self.compute_semantic_losses(
                source_images=source_ref_images.float(),
                generated_images=generated_images.float(),
                source_ages=source_ages,
                target_ages=target_ages,
                semantic_weights=semantic_weights,
                semantic_components=active_semantic_components,)

            loss_id = semantic_losses["loss_id"]
            loss_age = semantic_losses["loss_age"]
            loss_delta_age = semantic_losses["loss_delta_age"]
            loss_perc = semantic_losses["loss_perc"]

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------
        total_loss = (
            self.lambda_diff * loss_diff
            + self.lambda_id * loss_id
            + self.lambda_age * loss_age
            + self.lambda_delta_age * loss_delta_age
            + self.lambda_perc * loss_perc)

        # ----------------------------------------------------
        # Output dictionary
        # ----------------------------------------------------
        out = {
            "loss": total_loss,
            "loss_mode": loss_mode,
            "semantic_components": active_semantic_components,

            "loss_diff": loss_diff.detach(),
            "loss_id": loss_id.detach(),
            "loss_age": loss_age.detach(),
            "loss_delta_age": loss_delta_age.detach(),
            "loss_perc": loss_perc.detach(),

            "compute_diff": compute_diff,
            "compute_semantic": compute_semantic,

            "lambda_diff": self.lambda_diff,
            "lambda_id": self.lambda_id,
            "lambda_age": self.lambda_age,
            "lambda_delta_age": self.lambda_delta_age,
            "lambda_perc": self.lambda_perc,

            "source_age": source_ages.detach() if source_ages is not None else None,
            "target_age": target_ages.detach() if target_ages is not None else None,

            "timesteps_diff": (
                timesteps_diff.detach()
                if timesteps_diff is not None
                else None
            ),
            "timesteps_semantic": (
                timesteps_semantic.detach()
                if timesteps_semantic is not None
                else None
            ),
            "semantic_weights": (
                semantic_weights.detach()
                if semantic_weights is not None
                else None
            ),

            "loss_id_unweighted": semantic_losses.get("loss_id_unweighted", None),
            "loss_age_unweighted": semantic_losses.get("loss_age_unweighted", None),
            "loss_delta_age_unweighted": semantic_losses.get("loss_delta_age_unweighted", None),
            "loss_perc_unweighted": semantic_losses.get("loss_perc_unweighted", None),

            "identity_similarity": semantic_losses.get("identity_similarity", None),
            "age_src_pred": semantic_losses.get("age_src_pred", None),
            "age_gen_pred": semantic_losses.get("age_gen_pred", None),
            "lpips_score": semantic_losses.get("lpips_score", None),
        }

        if return_generated_image and generated_images is not None:
            out["generated_image"] = generated_images.detach()

        return out
