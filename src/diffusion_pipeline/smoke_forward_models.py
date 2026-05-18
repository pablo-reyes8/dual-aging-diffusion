# ============================================================
# GENERIC HELPERS FOR ANY BUNDLE
# ============================================================
import torch 
from src.diffusion_pipeline.load_diffusion_models import * 
import matplotlib.pyplot as plt

@torch.no_grad()
def encode_prompt_cfg_bundle(bundle, prompt, negative_prompt=""):
    tokenizer = bundle["tokenizer"]
    text_encoder = bundle["text_encoder"]

    prompts = [negative_prompt, prompt]

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )

    encoder_hidden_states = outputs.last_hidden_state
    uncond_emb = encoder_hidden_states[0:1]
    cond_emb = encoder_hidden_states[1:2]

    return uncond_emb, cond_emb


@torch.no_grad()
def encode_image_to_latent_bundle(bundle, image_tensor):
    vae = bundle["vae"]

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device=device, dtype=dtype)

    posterior = vae.encode(image_tensor).latent_dist
    latents = posterior.mean * vae.config.scaling_factor
    return latents


@torch.no_grad()
def decode_latent_to_image_bundle(bundle, latents):
    vae = bundle["vae"]

    latents = latents.to(device=device, dtype=dtype)
    latents = latents / vae.config.scaling_factor

    image = vae.decode(latents).sample
    image = image.clamp(-1, 1).float()
    return image


@torch.no_grad()
def img2img_single_bundle(
    bundle,
    image_tensor,
    prompt,
    negative_prompt="",
    strength=0.45,
    guidance_scale=7.5,
    num_inference_steps=40,
    seed=123,
):
    """
    Generic img2img for one bundle.
    image_tensor: [3,H,W] or [1,3,H,W], normalized in [-1,1]
    """
    unet = bundle["unet"]
    scheduler = bundle["scheduler_infer"]

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device=device, dtype=dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # 1. Encode image
    init_latents = encode_image_to_latent_bundle(bundle, image_tensor)

    # 2. Prepare scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)

    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    # 3. Add noise
    noise = torch.randn(
        init_latents.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    first_timestep = timesteps[0].repeat(init_latents.shape[0])
    latents = scheduler.add_noise(init_latents, noise, first_timestep)

    # 4. Text embeddings
    uncond_emb, cond_emb = encode_prompt_cfg_bundle(
        bundle=bundle,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    text_embeds = torch.cat([uncond_emb, cond_emb], dim=0).to(dtype=dtype)

    # 5. Denoising loop
    for t in timesteps:
        latent_model_input = torch.cat([latents] * 2, dim=0)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeds,
            return_dict=True,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(
            noise_pred,
            t,
            latents,
            return_dict=True,
        ).prev_sample

    # 6. Decode
    image = decode_latent_to_image_bundle(bundle, latents)
    return image


def tensor_to_img(x):
    if x.ndim == 4:
        x = x[0]
    return ((x.detach().cpu() + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()


def show_score_sweep(input_crop, outputs, prompts, title_prefix=""):
    n = 1 + len(outputs)

    plt.figure(figsize=(4 * n, 4))

    plt.subplot(1, n, 1)
    plt.imshow(tensor_to_img(input_crop))
    plt.title(f"{title_prefix}Original")
    plt.axis("off")

    for i, (out, prompt) in enumerate(zip(outputs, prompts), start=2):
        plt.subplot(1, n, i)
        plt.imshow(tensor_to_img(out))
        plt.axis("off")

        if "aging score of " in prompt:
            score_text = prompt.split("aging score of ")[-1].split("%")[0]
            plt.title(f"{title_prefix}score {score_text}%")
        elif "-year-old" in prompt:
            age_text = prompt.split("a ")[-1].split("-year-old")[0]
            plt.title(f"{title_prefix}age {age_text}")
        else:
            plt.title(f"{title_prefix}{i-2}")

    plt.tight_layout()
    plt.show()

    print("\nPrompts:")
    for p in prompts:
        print("-", p)