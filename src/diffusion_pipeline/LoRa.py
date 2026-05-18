# ============================================================
# Manual LoRA Linear wrapper
# ============================================================

import torch
import torch.nn as nn
import math
from src.utils.cuda_utils import * 

device, dtype = get_device_and_dtype()

class LoRALinear(nn.Module):
    """
    Manual LoRA wrapper around nn.Linear.

    Original:
        y = W x + b

    LoRA:
        y = W x + b + scale * B(A(dropout(x)))

    Where:
        A: in_features  -> r
        B: r            -> out_features

    Base layer is frozen.
    Only A and B are trainable.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)}")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_down = nn.Linear(in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, out_features, bias=False)

        # Standard LoRA init:
        # A random, B zero, so initial function equals base model.
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.lora_up(self.lora_down(self.dropout(x))) * self.scale
        return base_out + lora_out


def freeze_all_params(model):
    for p in model.parameters():
        p.requires_grad_(False)


def get_parent_module(root: nn.Module, module_name: str):
    """
    Given:
        module_name = 'down_blocks.0.attentions.0...to_q'

    Returns:
        parent module and child attribute name.
    """
    parts = module_name.split(".")
    parent = root

    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)

    child_name = parts[-1]
    return parent, child_name


def replace_module(root: nn.Module, module_name: str, new_module: nn.Module):
    parent, child_name = get_parent_module(root, module_name)

    if child_name.isdigit():
        parent[int(child_name)] = new_module
    else:
        setattr(parent, child_name, new_module)


def module_matches_targets(module_name: str, target_suffixes):
    """
    Matches endings:
        'to_q'
        'to_k'
        'to_v'
        'to_out.0'
    """
    for suffix in target_suffixes:
        if module_name == suffix or module_name.endswith("." + suffix):
            return True
    return False


def inject_manual_lora_unet(
    unet: nn.Module,
    rank: int,
    alpha: int,
    dropout: float,
    target_suffixes=None,
    verbose: bool = True,
):
    """
    Freezes entire UNet and replaces target nn.Linear modules with LoRALinear.
    """

    if target_suffixes is None:
        target_suffixes = ["to_q", "to_k", "to_v", "to_out.0"]

    freeze_all_params(unet)

    # Collect first, replace after.
    modules_to_replace = []
    for name, module in unet.named_modules():
        if module_matches_targets(name, target_suffixes) and isinstance(module, nn.Linear):
            modules_to_replace.append((name, module))

    if verbose:
        print(f"\nFound Linear modules to LoRA-wrap: {len(modules_to_replace)}")
        for name, module in modules_to_replace[:30]:
            print(f"  {name:90s} | {module.in_features}->{module.out_features}")
        if len(modules_to_replace) > 30:
            print(f"  ... {len(modules_to_replace) - 30} more")

    for name, module in modules_to_replace:
        lora_module = LoRALinear(
            base_layer=module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,).to(device=device, dtype=dtype)

        replace_module(unet, name, lora_module)

    # Safety: make sure only LoRA params are trainable.
    for name, p in unet.named_parameters():
        if ("lora_down" in name) or ("lora_up" in name):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    return unet


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