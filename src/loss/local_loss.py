# ============================================================
# LOCAL LDLA-STYLE LOSS FOR LOCAL FACE AGING BRANCH
#
# Implements:
#   L_local = λ_full  * L_full
#           + λ_zone  * L_zone
#           + λ_score * L_score
#           + λ_cycle * L_cycle
#
# Main inputs expected from batch:
#   batch["pixel_values"] : Tensor [B, 3, 256, 256] in [-1, 1]
#   batch["score"]        : Tensor [B] in [0, 1]
#   batch["prompt"]       : List[str] full prompts P_full
#   batch["zone_prompt"]  : List[str] zone prompts P_zone
#
# Optional:
#   batch["target_prompt"] : List[str] P_target for cycle loss
#
# Assumes local_bundle has:
#   local_bundle["vae"]
#   local_bundle["unet"]
#   local_bundle["tokenizer"]
#   local_bundle["text_encoder"]
#   local_bundle["scheduler_train"]
#
# Important:
#   - ScoreNet receives decoded RGB images [B, 3, 256, 256], not latents.
#   - VAE/text_encoder are frozen.
#   - ScoreNet may be frozen, but its forward must NOT be under torch.no_grad()
#     if we want gradients to flow back to decoded images and then to the UNet.
# ============================================================

import math
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss.ddim_generation import ddim_generate_images
from src.loss.directional_score_loss import DirectionalScoreLoss


class LDLALocalAgingLoss(nn.Module):
    def __init__(
        self,
        local_bundle: Dict[str, Any],
        score_net: Optional[nn.Module] = None,
        lambda_full: float = 1.0,
        lambda_zone: float = 0.25,
        lambda_score: float = 0.05,
        lambda_cycle: float = 0.0,
        freeze_score_net: bool = True,
        score_net_input_dtype: torch.dtype = torch.float32,
        device: Optional[str] = None,
        score_timestep_min: int = 5,
        score_timestep_max: int = 150,

        # ----------------------------------------------------------------
        # Score-edit generation mode (Error 1 fix), mirrors the global loss.
        #   "1_step_per_loss": original single-step x0_hat (cheap, blurry).
        #   "full_ddim":       short DDIM trajectory -> sharper crop for the
        #                      frozen ScoreNet. Use a low full_ddim_max_timestep.
        # ----------------------------------------------------------------
        score_loss_mode: str = "1_step_per_loss",
        full_ddim_num_steps: int = 10,
        full_ddim_max_timestep: int = 120,

        # Min-SNR-gamma reweighting of the full/zone diffusion losses.
        # Enabled by default for a more stable timestep balance.
        use_min_snr: bool = True,
        min_snr_gamma: float = 5.0,

        # ----------------------------------------------------------------
        # Directional / contrastive score loss (Error 3). Off by default.
        # Teaches controllability ("more score token -> more aging") in a way
        # robust to the absolute ScoreNet bias. Activated from the wrapper.
        # ----------------------------------------------------------------
        use_directional_score: bool = False,
        lambda_direction: float = 0.0,
        direction_margin: float = 0.05,
        direction_high_score: float = 0.90,
        direction_low_score: float = 0.30,
        direction_timestep_min: int = 20,
        direction_timestep_max: int = 200,

        unet_dtype: Optional[torch.dtype] = None):

        super().__init__()

        valid_score_modes = {"1_step_per_loss", "full_ddim"}
        score_loss_mode = str(score_loss_mode).lower().strip()
        if score_loss_mode not in valid_score_modes:
            raise ValueError(
                f"Invalid score_loss_mode='{score_loss_mode}'. "
                f"Valid: {sorted(valid_score_modes)}"
            )

        self.local_bundle = local_bundle
        self.score_timestep_min = int(score_timestep_min)
        self.score_timestep_max = int(score_timestep_max)

        self.score_loss_mode = score_loss_mode
        self.full_ddim_num_steps = int(full_ddim_num_steps)
        self.full_ddim_max_timestep = int(full_ddim_max_timestep)
        self.use_min_snr = bool(use_min_snr)
        self.min_snr_gamma = float(min_snr_gamma)

        self.use_directional_score = bool(use_directional_score)
        self.lambda_direction = float(lambda_direction)

        if self.use_directional_score and self.lambda_direction > 0.0:
            self.directional_loss = DirectionalScoreLoss(
                margin=direction_margin,
                high_score=direction_high_score,
                low_score=direction_low_score,
                timestep_min=direction_timestep_min,
                timestep_max=direction_timestep_max,
            )
        else:
            self.directional_loss = None
        self.vae = local_bundle["vae"]
        self.unet = local_bundle["unet"]
        self.tokenizer = local_bundle["tokenizer"]
        self.text_encoder = local_bundle["text_encoder"]
        self.scheduler = local_bundle["scheduler_train"]

        self.score_net = score_net

        self.lambda_full = float(lambda_full)
        self.lambda_zone = float(lambda_zone)
        self.lambda_score = float(lambda_score)
        self.lambda_cycle = float(lambda_cycle)

        self.score_net_input_dtype = score_net_input_dtype

        if device is None:
            device = next(self.unet.parameters()).device
        self.device = torch.device(device)

        if unet_dtype is None:
            unet_dtype = next(self.unet.parameters()).dtype
        self.unet_dtype = unet_dtype

        # Frozen modules.
        self.vae.eval()
        self.text_encoder.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        if self.score_net is not None:
            if freeze_score_net:
                self.score_net.eval()
                for p in self.score_net.parameters():
                    p.requires_grad_(False)
            else:
                self.score_net.train()

        # Safety checks.
        if self.lambda_score > 0 and self.score_net is None:
            raise ValueError("lambda_score > 0 but score_net is None.")

        if self.directional_loss is not None and self.score_net is None:
            raise ValueError(
                "use_directional_score=True with lambda_direction>0 but score_net is None."
            )

        if self.lambda_cycle > 0:
            print(
                "[WARN] lambda_cycle > 0: cycle loss is expensive because it uses "
                "extra UNet passes. For the first local training run, consider "
                "lambda_cycle=0.0."
            )

    # --------------------------------------------------------
    # Text conditioning
    # --------------------------------------------------------

    @torch.no_grad()
    def encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """
        Encodes prompts using the checkpoint's own tokenizer/text_encoder.

        Returns:
            encoder_hidden_states: [B, 77, cross_attention_dim]
        """
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

        return outputs.last_hidden_state.to(device=self.device, dtype=self.unet_dtype)

    # --------------------------------------------------------
    # VAE helpers
    # --------------------------------------------------------

    @torch.no_grad()
    def encode_images_to_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values:
            [B, 3, 256, 256] in [-1, 1]

        returns:
            latents [B, 4, 32, 32]
        """
        x = pixel_values.to(device=self.device, dtype=self.unet_dtype)

        latents = self.vae.encode(x).latent_dist.mean
        latents = latents * self.vae.config.scaling_factor

        return latents.to(dtype=self.unet_dtype)

    def decode_latents_to_images(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents:
            [B, 4, 32, 32]

        returns:
            decoded images [B, 3, 256, 256] approximately in [-1, 1]

        Important:
            No torch.no_grad() here.
            We want gradients from ScoreNet(image) to flow back to latents
            and then to the UNet prediction.
        """
        latents_for_vae = latents / self.vae.config.scaling_factor
        latents_for_vae = latents_for_vae.to(device=self.device, dtype=self.unet_dtype)

        decoded = self.vae.decode(latents_for_vae, return_dict=True).sample
        decoded = decoded.clamp(-1.0, 1.0)

        return decoded

    # --------------------------------------------------------
    # Diffusion helpers
    # --------------------------------------------------------

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        timesteps = torch.randint(
            low=0,
            high=self.scheduler.config.num_train_timesteps,
            size=(batch_size,),
            device=self.device,
        ).long()

        return timesteps

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor) -> torch.Tensor:

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        return noisy_latents.to(dtype=self.unet_dtype)

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor) -> torch.Tensor:
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
        DDPM one-step x0 estimate:

            x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)

        Returns:
            x0_hat latent [B, 4, 32, 32]
        """
        alphas_cumprod = self.scheduler.alphas_cumprod.to(
            device=self.device,
            dtype=noisy_latents.dtype)

        alpha_bar_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        x0_hat = (
            noisy_latents - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t.clamp_min(1e-8)

        return x0_hat

    # --------------------------------------------------------
    # Individual losses
    # --------------------------------------------------------

    def min_snr_weight(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Min-SNR-gamma weight for epsilon-prediction:
            w(t) = min(SNR(t), gamma) / SNR(t),  SNR = alpha_bar / (1 - alpha_bar)
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

    def diffusion_prompt_loss(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        noisy_latents: torch.Tensor,
        prompts: List[str]):

        """
        Standard diffusion noise-prediction loss for one prompt type.

        Optionally reweighted with Min-SNR-gamma (use_min_snr=True).
        """
        encoder_hidden_states = self.encode_prompts(prompts)

        noise_pred = self.predict_noise(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden_states)

        if self.use_min_snr:
            per_sample = (noise_pred.float() - noise.float()).pow(2).mean(dim=[1, 2, 3])
            weights = self.min_snr_weight(timesteps)
            loss = (weights * per_sample).mean()
        else:
            loss = F.mse_loss(noise_pred.float(), noise.float())

        return loss, noise_pred

    def score_loss_from_x0_hat(
        self,
        x0_hat_latents: torch.Tensor,
        target_scores: torch.Tensor):

        """
        Computes:
            L_score = MSE(S(VAE.decode(x0_hat)), target_score)

        ScoreNet receives:
            decoded_crop: [B, 3, 256, 256] in [-1, 1]
        """
        if self.score_net is None:
            raise ValueError("score_net is None, cannot compute score loss.")

        decoded_crop = self.decode_latents_to_images(x0_hat_latents)

        # ScoreNet was trained in fp32 on RGB image tensors.
        decoded_for_score = decoded_crop.to(
            device=self.device,
            dtype=self.score_net_input_dtype,)

        target_scores = target_scores.to(
            device=self.device,
            dtype=self.score_net_input_dtype).view(-1)

        # No torch.no_grad() here: gradient must flow to decoded_for_score.
        # Keep ScoreNet in fp32 even when the diffusion loss is under bf16/fp16
        # autocast; auxiliary predictors are a common source of NaN gradients.
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            score_pred = self.score_net(decoded_for_score.float()).view(-1)

        loss_score = F.mse_loss(score_pred.float(), target_scores.float())

        return loss_score, score_pred, decoded_crop

    def generate_edited_crop_images(
        self,
        z0: torch.Tensor,
        prompts: List[str],
        timestep_min: int,
        timestep_max: int,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generates edited crops conditioned on `prompts`, in [-1, 1].

        Mode-aware (Error 1):
            "full_ddim":       short DDIM trajectory -> sharper crop.
            "1_step_per_loss": single-step x0_hat estimate (blurry).

        Used by both the score loss and the directional score loss. When `noise`
        is provided (1-step mode), it is shared across calls so directional
        high/low edits differ only by the prompt.
        """
        if self.score_loss_mode == "full_ddim":
            return ddim_generate_images(
                self,
                z0=z0,
                prompts=prompts,
                num_steps=self.full_ddim_num_steps,
                max_timestep=self.full_ddim_max_timestep,
                noise=noise,
            )

        B = z0.shape[0]
        if noise is None:
            noise = torch.randn_like(z0)

        timesteps = torch.randint(
            low=int(timestep_min),
            high=int(timestep_max) + 1,
            size=(B,),
            device=self.device,
        ).long()

        zt = self.add_noise(z0, noise, timesteps)
        hidden = self.encode_prompts(prompts)
        noise_pred = self.predict_noise(
            noisy_latents=zt,
            timesteps=timesteps,
            encoder_hidden_states=hidden,
        )
        x0_hat = self.predict_x0_from_noise(
            noisy_latents=zt,
            timesteps=timesteps,
            noise_pred=noise_pred,
        )
        return self.decode_latents_to_images(x0_hat)

    def directional_score_loss_fn(
        self,
        z0: torch.Tensor,
        high_prompts: List[str],
        low_prompts: List[str],
    ):
        """
        Relative/contrastive score loss (Error 3).

        Generates two edits of the SAME crop sharing the same noise but with a
        high-score and a low-score prompt, then requires ScoreNet to rank the
        high edit as more aged than the low edit by a margin. Robust to the
        absolute ScoreNet bias on VAE-decoded crops.
        """
        shared_noise = torch.randn_like(z0)

        imgs_high = self.generate_edited_crop_images(
            z0=z0,
            prompts=high_prompts,
            timestep_min=self.directional_loss.timestep_min,
            timestep_max=self.directional_loss.timestep_max,
            noise=shared_noise,
        )
        imgs_low = self.generate_edited_crop_images(
            z0=z0,
            prompts=low_prompts,
            timestep_min=self.directional_loss.timestep_min,
            timestep_max=self.directional_loss.timestep_max,
            noise=shared_noise,
        )

        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            s_high = self.score_net(
                imgs_high.to(self.score_net_input_dtype).float()
            ).view(-1)
            s_low = self.score_net(
                imgs_low.to(self.score_net_input_dtype).float()
            ).view(-1)

        loss_dir = self.directional_loss(score_high=s_high, score_low=s_low)
        return loss_dir, s_high, s_low

    def latent_cycle_loss(
        self,
        z0: torch.Tensor,
        source_prompts: List[str],
        target_prompts: List[str]):

        """
        Approximate LDLA latent cycle consistency.

        Concept:
            z0 --target prompt--> z0_target_hat
            z0_target_hat --source prompt--> z0_recon_hat
            L_cycle = MSE(z0_recon_hat, z0)

        This is an operational one-step approximation using DDPM x0 estimates.
        It is expensive: two extra UNet calls.
        """
        B = z0.shape[0]

        timesteps = self.sample_timesteps(B)
        noise = torch.randn_like(z0)

        zt = self.add_noise(z0, noise, timesteps)

        target_hidden = self.encode_prompts(target_prompts)
        eps_target = self.predict_noise(
            noisy_latents=zt,
            timesteps=timesteps,
            encoder_hidden_states=target_hidden,
        )

        z0_target_hat = self.predict_x0_from_noise(
            noisy_latents=zt,
            timesteps=timesteps,
            noise_pred=eps_target,
        )

        # Re-noise the target estimate and reconstruct back to source.
        noise_back = torch.randn_like(z0_target_hat)
        zt_back = self.add_noise(z0_target_hat, noise_back, timesteps)

        source_hidden = self.encode_prompts(source_prompts)
        eps_source = self.predict_noise(
            noisy_latents=zt_back,
            timesteps=timesteps,
            encoder_hidden_states=source_hidden,
        )

        z0_recon_hat = self.predict_x0_from_noise(
            noisy_latents=zt_back,
            timesteps=timesteps,
            noise_pred=eps_source,
        )

        loss_cycle = F.mse_loss(z0_recon_hat.float(), z0.float())

        return loss_cycle

    # --------------------------------------------------------
    # Main forward
    # --------------------------------------------------------

    def forward(
        self,
        batch: Dict[str, Any],

        # Which part of the local objective to compute.
        # Recommended for training:
        #   "full", "score", "zone"
        #
        # Debug / larger GPU:
        #   "full_score", "full_zone", "all"
        loss_mode: str = "all",

        # Prompts selected by the training chunk.
        # If None, defaults to batch["prompt"] and batch["zone_prompt"].
        source_prompts: Optional[List[str]] = None,
        zone_prompts: Optional[List[str]] = None,

        # Target conditioning for score/cycle.
        # These should be built in the local training chunk.
        target_prompts: Optional[List[str]] = None,
        target_scores: Optional[torch.Tensor] = None,

        # Directional/contrastive score loss prompts (Error 3). Built in the
        # local training chunk from the base prompt with high/low score tokens.
        direction_high_prompts: Optional[List[str]] = None,
        direction_low_prompts: Optional[List[str]] = None,

        return_decoded_score_image: bool = False) -> Dict[str, Any]:

        """
        Memory-aware LDLA-style local loss forward.

        Instead of always computing:
            L_full + L_zone + L_score + L_cycle

        this forward can compute only selected components via loss_mode.

        Valid loss_mode values:
            "full":
                only L_full

            "zone":
                only L_zone

            "score":
                only L_score

            "cycle":
                only L_cycle

            "full_zone":
                L_full + L_zone

            "full_score":
                L_full + L_score

            "score_zone":
                L_score + L_zone

            "all":
                all active components according to lambdas

        Recommended training on limited VRAM:
            sample loss_mode stochastically in the training wrapper:
                50% "full"
                35% "score"
                15% "zone"
        """

        # --------------------------------------------------------
        # Parse loss mode
        # --------------------------------------------------------
        loss_mode = str(loss_mode).lower().strip()

        valid_modes = {
            "full",
            "zone",
            "score",
            "cycle",
            "direction",
            "full_zone",
            "full_score",
            "score_zone",
            "all"}

        if loss_mode not in valid_modes:
            raise ValueError(
                f"Invalid loss_mode='{loss_mode}'. "
                f"Valid modes are: {sorted(valid_modes)}")

        compute_full = loss_mode in {"full", "full_zone", "full_score", "all"}
        compute_zone = loss_mode in {"zone", "full_zone", "score_zone", "all"}
        compute_score = loss_mode in {"score", "full_score", "score_zone", "all"}
        compute_cycle = loss_mode in {"cycle", "all"}
        compute_direction = loss_mode in {"direction", "all"}

        # Respect lambdas too.
        compute_full = compute_full and (self.lambda_full > 0.0)
        compute_zone = compute_zone and (self.lambda_zone > 0.0)
        compute_score = compute_score and (self.lambda_score > 0.0)
        compute_cycle = compute_cycle and (self.lambda_cycle > 0.0)
        compute_direction = (
            compute_direction
            and self.directional_loss is not None
            and self.lambda_direction > 0.0
        )

        # --------------------------------------------------------
        # Inputs
        # --------------------------------------------------------
        pixel_values = batch["pixel_values"].to(
            device=self.device,
            dtype=self.unet_dtype,)

        source_scores = batch["score"].to(
            device=self.device,
            dtype=torch.float32).view(-1)

        B = pixel_values.shape[0]

        # --------------------------------------------------------
        # Prompt defaults
        # --------------------------------------------------------
        if source_prompts is None:
            source_prompts = batch["prompt"]

        if zone_prompts is None:
            zone_prompts = batch["zone_prompt"]

        if len(source_prompts) != B:
            raise ValueError(
                f"len(source_prompts)={len(source_prompts)} but batch size={B}."
            )

        if len(zone_prompts) != B:
            raise ValueError(
                f"len(zone_prompts)={len(zone_prompts)} but batch size={B}.")

        # --------------------------------------------------------
        # Target conditioning checks
        # --------------------------------------------------------
        needs_target = compute_score or compute_cycle

        if needs_target:
            if target_prompts is None:
                raise ValueError(
                    "target_prompts must be passed when loss_mode requires "
                    "L_score or L_cycle."
                )

            if target_scores is None:
                raise ValueError(
                    "target_scores must be passed when loss_mode requires "
                    "L_score or L_cycle."
                )

            if len(target_prompts) != B:
                raise ValueError(
                    f"len(target_prompts)={len(target_prompts)} but batch size={B}."
                )

            target_scores = target_scores.to(
                device=self.device,
                dtype=torch.float32,
            ).view(-1)

            if target_scores.shape[0] != B:
                raise ValueError(
                    f"target_scores has shape {target_scores.shape}, "
                    f"but batch size is {B}."
                )

            target_scores = target_scores.clamp(0.0, 1.0)

        else:
            target_scores = None

        # --------------------------------------------------------
        # Directional prompt checks
        # --------------------------------------------------------
        if compute_direction:
            if direction_high_prompts is None or direction_low_prompts is None:
                raise ValueError(
                    "direction_high_prompts and direction_low_prompts must be "
                    "passed when loss_mode requires the directional score loss."
                )
            if len(direction_high_prompts) != B or len(direction_low_prompts) != B:
                raise ValueError(
                    "direction_high_prompts/direction_low_prompts length must equal "
                    f"batch size={B}."
                )

        # --------------------------------------------------------
        # Encode source crop to latent z0
        # --------------------------------------------------------
        # z0: [B, 4, 32, 32] for 256x256 crops.
        z0 = self.encode_images_to_latents(pixel_values)

        # --------------------------------------------------------
        # Initialize default outputs
        # --------------------------------------------------------
        loss_full = torch.zeros((), device=self.device)
        loss_zone = torch.zeros((), device=self.device)
        loss_score = torch.zeros((), device=self.device)
        loss_cycle = torch.zeros((), device=self.device)
        loss_direction = torch.zeros((), device=self.device)

        noise_pred_full = None
        noise_pred_zone = None
        noise_pred_target = None

        score_pred = None
        decoded_score_image = None
        x0_hat_target_latents = None

        direction_score_high = None
        direction_score_low = None

        timesteps_full_zone = None
        timesteps_score = None

        # --------------------------------------------------------
        # L_full and/or L_zone
        # --------------------------------------------------------
        # If both are requested, they share zt/noise/timesteps.
        # This is still two UNet forwards because prompts differ.
        # --------------------------------------------------------
        if compute_full or compute_zone:
            noise = torch.randn_like(z0)
            timesteps_full_zone = self.sample_timesteps(B)
            zt = self.add_noise(z0, noise, timesteps_full_zone)

            if compute_full:
                loss_full, noise_pred_full = self.diffusion_prompt_loss(
                    latents=z0,
                    noise=noise,
                    timesteps=timesteps_full_zone,
                    noisy_latents=zt,
                    prompts=source_prompts,
                )

            if compute_zone:
                loss_zone, noise_pred_zone = self.diffusion_prompt_loss(
                    latents=z0,
                    noise=noise,
                    timesteps=timesteps_full_zone,
                    noisy_latents=zt,
                    prompts=zone_prompts,
                )

        # --------------------------------------------------------
        # L_score
        # --------------------------------------------------------
        # Uses target prompt and target score.
        # This is the expensive branch:
        #   UNet target forward + one-step x0 + VAE decode + ScoreNet.
        # --------------------------------------------------------
        if compute_score:
            if not hasattr(self, "score_timestep_min"):
                raise AttributeError(
                    "self.score_timestep_min is missing. Add it in __init__."
                )

            if not hasattr(self, "score_timestep_max"):
                raise AttributeError(
                    "self.score_timestep_max is missing. Add it in __init__."
                )

            if self.score_loss_mode == "full_ddim":
                # Sharper edited crop via short DDIM trajectory, then ScoreNet.
                decoded_score_image = self.generate_edited_crop_images(
                    z0=z0,
                    prompts=target_prompts,
                    timestep_min=self.score_timestep_min,
                    timestep_max=self.score_timestep_max,
                )
                decoded_for_score = decoded_score_image.to(
                    device=self.device, dtype=self.score_net_input_dtype
                )
                tgt = target_scores.to(
                    device=self.device, dtype=self.score_net_input_dtype
                ).view(-1)
                with torch.amp.autocast(device_type=self.device.type, enabled=False):
                    score_pred = self.score_net(decoded_for_score.float()).view(-1)
                loss_score = F.mse_loss(score_pred.float(), tgt.float())
            else:
                noise_score = torch.randn_like(z0)

                timesteps_score = torch.randint(
                    low=int(self.score_timestep_min),
                    high=int(self.score_timestep_max) + 1,
                    size=(B,),
                    device=self.device,
                ).long()

                zt_score = self.add_noise(
                    latents=z0,
                    noise=noise_score,
                    timesteps=timesteps_score,
                )

                target_hidden = self.encode_prompts(target_prompts)

                noise_pred_target = self.predict_noise(
                    noisy_latents=zt_score,
                    timesteps=timesteps_score,
                    encoder_hidden_states=target_hidden,
                )

                x0_hat_target_latents = self.predict_x0_from_noise(
                    noisy_latents=zt_score,
                    timesteps=timesteps_score,
                    noise_pred=noise_pred_target,
                )

                loss_score, score_pred, decoded_score_image = self.score_loss_from_x0_hat(
                    x0_hat_latents=x0_hat_target_latents,
                    target_scores=target_scores,
                )

        # --------------------------------------------------------
        # L_direction (relative / contrastive score loss)
        # --------------------------------------------------------
        if compute_direction:
            loss_direction, direction_score_high, direction_score_low = (
                self.directional_score_loss_fn(
                    z0=z0,
                    high_prompts=direction_high_prompts,
                    low_prompts=direction_low_prompts,
                )
            )

        # --------------------------------------------------------
        # L_cycle
        # --------------------------------------------------------
        # Very expensive. Keep disabled for early training.
        # --------------------------------------------------------
        if compute_cycle:
            loss_cycle = self.latent_cycle_loss(
                z0=z0,
                source_prompts=source_prompts,
                target_prompts=target_prompts)

        # --------------------------------------------------------
        # Total loss
        # --------------------------------------------------------
        total_loss = (
            self.lambda_full * loss_full
            + self.lambda_zone * loss_zone
            + self.lambda_score * loss_score
            + self.lambda_cycle * loss_cycle
            + self.lambda_direction * loss_direction
        )

        # --------------------------------------------------------
        #  Output dictionary
        # --------------------------------------------------------
        out = {
            "loss": total_loss,
            "loss_mode": loss_mode,

            # Component losses.
            "loss_full": loss_full.detach(),
            "loss_zone": loss_zone.detach(),
            "loss_score": loss_score.detach(),
            "loss_cycle": loss_cycle.detach(),
            "loss_direction": loss_direction.detach(),

            # Active flags.
            "compute_full": compute_full,
            "compute_zone": compute_zone,
            "compute_score": compute_score,
            "compute_cycle": compute_cycle,
            "compute_direction": compute_direction,

            # Lambdas.
            "lambda_full": self.lambda_full,
            "lambda_zone": self.lambda_zone,
            "lambda_score": self.lambda_score,
            "lambda_cycle": self.lambda_cycle,
            "lambda_direction": self.lambda_direction,

            # Directional diagnostics.
            "direction_score_high": (
                direction_score_high.detach()
                if direction_score_high is not None
                else None
            ),
            "direction_score_low": (
                direction_score_low.detach()
                if direction_score_low is not None
                else None
            ),

            # Scores.
            "source_score": source_scores.detach(),
            "target_score": target_scores.detach() if target_scores is not None else None,
            "score_pred": score_pred.detach() if score_pred is not None else None,

            # Timesteps.
            "timesteps_full_zone": (
                timesteps_full_zone.detach()
                if timesteps_full_zone is not None
                else None
            ),
            "timesteps_score": (
                timesteps_score.detach()
                if timesteps_score is not None
                else None
            ),

            # Optional diagnostics.
            "noise_pred_full": (
                noise_pred_full.detach()
                if noise_pred_full is not None
                else None
            ),
            "noise_pred_zone": (
                noise_pred_zone.detach()
                if noise_pred_zone is not None
                else None
            ),
            "noise_pred_target": (
                noise_pred_target.detach()
                if noise_pred_target is not None
                else None
            ),
            "x0_hat_target_latents": (
                x0_hat_target_latents.detach()
                if x0_hat_target_latents is not None
                else None
            ),
        }

        if return_decoded_score_image and decoded_score_image is not None:
            out["decoded_score_image"] = decoded_score_image.detach()

        return out
