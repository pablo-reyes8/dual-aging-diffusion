# ============================================================
# SHARED MULTI-STEP DDIM GENERATION HELPER
#
# Used by GlobalAgingLoss and LDLALocalAgingLoss when
# semantic/score loss mode == "full_ddim".
#
# Motivation:
#   The original LDLA-style code estimates the edited image with a
#   SINGLE-STEP x0_hat (cheap, but very blurry). Blurry one-step
#   reconstructions create a domain gap for the frozen auxiliaries
#   (ViT-age, FaceNet, ScoreNet), which were trained on sharp images.
#
#   This helper runs a short deterministic DDIM trajectory instead,
#   producing a sharper edited latent/image. It is gradient-carrying
#   so the adapters still receive a learning signal.
#
# Duck typing:
#   `loss_obj` only needs to expose:
#       - loss_obj.device
#       - loss_obj.scheduler  (with .alphas_cumprod and
#                              .config.num_train_timesteps)
#       - loss_obj.encode_prompts(prompts) -> hidden states
#       - loss_obj.add_noise(latents, noise, timesteps)
#       - loss_obj.predict_noise(noisy_latents, timesteps, hidden)
#       - loss_obj.decode_latents_to_images(latents)
#
#   Both GlobalAgingLoss and LDLALocalAgingLoss satisfy this.
#
# Memory note:
#   Gradient flows through `num_steps` UNet forwards. Keep `num_steps`
#   small (default 10) and gradient checkpointing enabled. This mode is
#   off by default; "1_step_per_loss" remains the default everywhere.
# ============================================================

from typing import List, Optional

import torch


def _build_ddim_timestep_schedule(
    max_timestep: int,
    num_steps: int,
    num_train_timesteps: int,
) -> List[int]:
    """
    Builds a decreasing integer timestep schedule from `max_timestep`
    down to 0 with `num_steps` denoising steps.

    Returns a list of length num_steps + 1, e.g. [120, 108, ..., 0].
    """
    max_timestep = int(max(1, min(max_timestep, num_train_timesteps - 1)))
    num_steps = int(max(1, num_steps))

    ts = torch.linspace(
        float(max_timestep),
        0.0,
        steps=num_steps + 1,
    )
    ts = ts.round().long().clamp(0, num_train_timesteps - 1)

    return [int(v) for v in ts.tolist()]


def ddim_generate_latents(
    loss_obj,
    z0: torch.Tensor,
    prompts: List[str],
    num_steps: int = 10,
    max_timestep: int = 120,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Deterministic (eta=0) DDIM generation toward the conditioning `prompts`,
    starting from the clean latent `z0` noised up to `max_timestep`.

    Returns the final edited latent (approximately x0 at t ~ 0), with gradient
    to the UNet adapters.
    """
    device = loss_obj.device
    B = z0.shape[0]

    num_train = int(loss_obj.scheduler.config.num_train_timesteps)

    alphas_cumprod = loss_obj.scheduler.alphas_cumprod.to(
        device=device,
        dtype=torch.float32,
    )

    hidden = loss_obj.encode_prompts(prompts)

    if noise is None:
        noise = torch.randn_like(z0)

    schedule = _build_ddim_timestep_schedule(
        max_timestep=max_timestep,
        num_steps=num_steps,
        num_train_timesteps=num_train,
    )

    t_start = torch.full((B,), int(schedule[0]), device=device, dtype=torch.long)
    zt = loss_obj.add_noise(z0, noise, t_start)

    for i in range(len(schedule) - 1):
        t_cur = int(schedule[i])
        t_next = int(schedule[i + 1])

        t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)

        eps = loss_obj.predict_noise(
            noisy_latents=zt,
            timesteps=t_batch,
            encoder_hidden_states=hidden,
        )

        ab_t = alphas_cumprod[t_cur].clamp(1e-8, 1.0)
        ab_next = alphas_cumprod[t_next].clamp(1e-8, 1.0) if t_next > 0 else torch.tensor(
            1.0, device=device, dtype=torch.float32
        )

        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_one_minus_ab_t = torch.sqrt(1.0 - ab_t)

        eps = eps.to(dtype=torch.float32)
        zt_f = zt.to(dtype=torch.float32)

        x0 = (zt_f - sqrt_one_minus_ab_t * eps) / sqrt_ab_t

        if t_next > 0:
            sqrt_ab_next = torch.sqrt(ab_next)
            sqrt_one_minus_ab_next = torch.sqrt(1.0 - ab_next)
            zt_f = sqrt_ab_next * x0 + sqrt_one_minus_ab_next * eps
        else:
            zt_f = x0

        zt = zt_f.to(dtype=zt.dtype)

    return zt


def ddim_generate_images(
    loss_obj,
    z0: torch.Tensor,
    prompts: List[str],
    num_steps: int = 10,
    max_timestep: int = 120,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Same as ddim_generate_latents but returns decoded images in [-1, 1].
    """
    latents = ddim_generate_latents(
        loss_obj=loss_obj,
        z0=z0,
        prompts=prompts,
        num_steps=num_steps,
        max_timestep=max_timestep,
        noise=noise,
    )
    return loss_obj.decode_latents_to_images(latents)
