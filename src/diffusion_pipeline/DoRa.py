import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from src.diffusion_pipeline.LoRa import *
from src.utils.cuda_utils import * 

device, dtype = get_device_and_dtype()
class DoRALinear(nn.Module):
    """
    Manual DoRA wrapper around nn.Linear.

    Base Linear:
        y = x W^T + b

    DoRA:
        W_eff = m * normalize(W + scale * DeltaW)

        DeltaW = B @ A

    Where:
        W: frozen base weight [out_features, in_features]
        A: lora_down weight [rank, in_features]
        B: lora_up weight [out_features, rank]
        m: trainable magnitude [out_features]

    Initial state:
        lora_up = 0
        m = ||W|| per output row

    Therefore:
        W_eff starts exactly equal to W.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        eps: float = 1e-6):

        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"DoRALinear expects nn.Linear, got {type(base_layer)}")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.eps = eps

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze original layer.
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Low-rank direction update.
        self.lora_down = nn.Linear(in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # Magnitude vector initialized as row norm of base weight.
        with torch.no_grad():
            base_weight = self.base_layer.weight.detach().float()
            magnitude = torch.linalg.norm(base_weight, dim=1)

        self.magnitude = nn.Parameter(magnitude)

    def get_delta_weight(self):
        """
        Returns DeltaW with shape [out_features, in_features].
        """
        delta = self.lora_up.weight @ self.lora_down.weight
        delta = delta * self.scale
        return delta

    def get_effective_weight(self):
        """
        Computes DoRA effective weight:
            W_eff = m * normalize(W + DeltaW)
        """
        base_weight = self.base_layer.weight
        delta_weight = self.get_delta_weight().to(dtype=base_weight.dtype, device=base_weight.device)

        direction = base_weight + delta_weight

        direction_norm = torch.linalg.norm(direction.float(), dim=1, keepdim=True)
        direction_norm = direction_norm.clamp(min=self.eps).to(dtype=direction.dtype)

        magnitude = self.magnitude.to(dtype=direction.dtype, device=direction.device).view(-1, 1)

        effective_weight = magnitude * (direction / direction_norm)
        return effective_weight

    def forward(self, x):
        effective_weight = self.get_effective_weight()
        bias = self.base_layer.bias
        return F.linear(x, effective_weight, bias)
    

@torch.no_grad()
def encode_prompt_bundle(bundle, prompts):
    tokenizer = bundle["tokenizer"]
    text_encoder = bundle["text_encoder"]

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    out = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )

    return out.last_hidden_state


@torch.no_grad()
def encode_latents_bundle(bundle, x):
    vae = bundle["vae"]
    x = x.to(device=device, dtype=dtype)
    latents = vae.encode(x).latent_dist.mean
    latents = latents * vae.config.scaling_factor
    return latents


def lora_forward_shape_test(bundle, H, W, prompt):
    unet = bundle["unet"]
    scheduler = bundle["scheduler_train"]

    B = 2
    x = torch.rand(B, 3, H, W, device=device, dtype=dtype) * 2 - 1

    with torch.no_grad():
        latents = encode_latents_bundle(bundle, x)
        encoder_hidden_states = encode_prompt_bundle(bundle, [prompt] * B)

    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (B,),
        device=device,
    ).long()

    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    with torch.no_grad():
        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states.to(dtype=dtype),
            return_dict=True,
        ).sample

    print(f"\n{bundle['name']} shape test")
    print("x:", x.shape)
    print("latents:", latents.shape)
    print("encoder_hidden_states:", encoder_hidden_states.shape)
    print("noise_pred:", noise_pred.shape)


def inject_manual_dora_unet(
    unet: nn.Module,
    rank: int,
    alpha: int,
    dropout: float,
    target_suffixes=None,
    verbose: bool = True):

    """
    Freezes entire UNet and replaces target nn.Linear modules with DoRALinear.

    Trainable parameters:
        lora_down.weight
        lora_up.weight
        magnitude
    """

    if target_suffixes is None:
        target_suffixes = ["to_q", "to_k", "to_v", "to_out.0"]

    freeze_all_params(unet)

    modules_to_replace = []

    for name, module in unet.named_modules():
        if module_matches_targets(name, target_suffixes) and isinstance(module, nn.Linear):
            modules_to_replace.append((name, module))

    if verbose:
        print(f"\nFound Linear modules to DoRA-wrap: {len(modules_to_replace)}")
        for name, module in modules_to_replace[:30]:
            print(f"  {name:90s} | {module.in_features}->{module.out_features}")
        if len(modules_to_replace) > 30:
            print(f"  ... {len(modules_to_replace) - 30} more")

    for name, module in modules_to_replace:
        base_device = module.weight.device
        base_dtype = module.weight.dtype
        dora_module = DoRALinear(
            base_layer=module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        ).to(device=base_device, dtype=base_dtype)

        replace_module(unet, name, dora_module)

    # Safety: only DoRA parameters trainable.
    for name, p in unet.named_parameters():
        if (
            ("lora_down" in name)
            or ("lora_up" in name)
            or ("magnitude" in name)
        ):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    return unet
