# ============================================================
# MODULAR DIFFUSION BUNDLE BUILDER + MIXED ADAPTER SETUP
#
# GLOBAL = RealisticVision or any SD-compatible checkpoint
# LOCAL  = SD1.5 or RealisticVision noVAE
#
# ADAPTERS:
#   GLOBAL -> LoRA
#   LOCAL  -> DoRA
#
# IMPORTANT:
#   - This code loads the base GLOBAL and LOCAL bundles once.
#   - The mixed setup DOES NOT reload UNets.
#   - The mixed setup injects LoRA/DoRA into the already-instantiated UNets.
#   - If LOCAL_MODEL_ID is a noVAE checkpoint, it automatically uses GLOBAL_VAE_ID.
#
# Assumes already defined:
#   - inject_manual_lora_unet
#   - inject_manual_dora_unet
#   - print_gpu_mem
#   - print_trainable_report
#   - list_trainable_names
# ============================================================

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from src.diffusion_pipeline.DoRa import * 


# ============================================================
# DEFAULT MODEL IDS
# ============================================================

#GLOBAL_MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
#GLOBAL_VAE_ID = "stabilityai/sd-vae-ft-mse"

# You can now safely change this to:

# LOCAL_MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
#LOCAL_MODEL_ID = "runwayml/stable-diffusion-v1-5"



# ===========================================================
# PARAMETER UTILITIES
# ============================================================

def count_parameters(model):
    total_params = 0
    trainable_params = 0
    total_tensors = 0
    trainable_tensors = 0

    for _, p in model.named_parameters():
        n = p.numel()

        total_params += n
        total_tensors += 1

        if p.requires_grad:
            trainable_params += n
            trainable_tensors += 1

    frozen_params = total_params - trainable_params
    trainable_pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_pct": trainable_pct,
        "total_tensors": total_tensors,
        "trainable_tensors": trainable_tensors,
    }


def print_parameter_report(title, model):
    stats = count_parameters(model)

    print(f"\n[{title}]")
    print(f"Total params:      {stats['total_params']:,}")
    print(f"Trainable params:  {stats['trainable_params']:,}")
    print(f"Frozen params:     {stats['frozen_params']:,}")
    print(f"Trainable %:       {stats['trainable_pct']:.6f}%")
    print(f"Total tensors:     {stats['total_tensors']:,}")
    print(f"Trainable tensors: {stats['trainable_tensors']:,}")

    return stats


def get_trainable_params_and_names(model):
    params = []
    names = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
            names.append(name)

    return params, names


def cast_trainable_parameters_to_fp32(model, verbose=True):
    """
    Keeps adapter parameters in fp32 for optimizer stability.

    The base UNet can stay fp16/bf16 and autocast still controls compute, but
    AdamW should not update LoRA/DoRA weights stored in fp16.
    """
    converted = 0
    already_fp32 = 0

    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dtype == torch.float32:
            already_fp32 += 1
            continue
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()
        converted += 1

    if verbose:
        print(
            "[Adapter dtype] trainable params fp32:",
            f"converted={converted}",
            f"already_fp32={already_fp32}",
        )

    return model


# ============================================================
# FREEZE UTILITIES
# ============================================================

def freeze_model(model):
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


def freeze_all_parameters(model):
    for p in model.parameters():
        p.requires_grad_(False)

    return model


# ============================================================
# VAE RESOLUTION HELPERS
# ============================================================

def is_no_vae_checkpoint(model_id):
    """
    Detects checkpoints that are expected not to contain a VAE subfolder.

    Example:
        SG161222/Realistic_Vision_V6.0_B1_noVAE
    """
    if model_id is None:
        return False

    model_id_lower = str(model_id).lower()

    return (
        "novae" in model_id_lower
        or "no_vae" in model_id_lower
        or "no-vae" in model_id_lower
    )


def resolve_vae_loading_policy(
    model_id,
    vae_id=None,
    force_external_vae=None,
):
    """
    Decides whether to load the VAE from the checkpoint itself or from vae_id.

    Rules:
        1. If force_external_vae is explicitly provided, obey it.
        2. If model_id looks like noVAE, use external VAE.
        3. Otherwise, use the checkpoint's internal VAE.

    This is what allows:
        LOCAL_MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"

    without changing anything else.
    """
    if force_external_vae is not None:
        load_external_vae = bool(force_external_vae)
    else:
        load_external_vae = is_no_vae_checkpoint(model_id)

    if load_external_vae and vae_id is None:
        raise ValueError(
            f"Model appears to require an external VAE, but vae_id=None.\n"
            f"model_id={model_id}"
        )

    return load_external_vae


def make_bundle_name(branch_name, model_id, adapter_name=None):
    """
    Creates readable bundle names without hardcoding SD1.5 or RealisticVision.
    """
    short_model_name = str(model_id).split("/")[-1]

    if adapter_name is None:
        return f"{branch_name}_{short_model_name}"

    return f"{branch_name}_{adapter_name}_{short_model_name}"


# ============================================================
# LOAD DIFFUSION COMPONENTS
# ============================================================

def load_diffusion_components(
    model_id,
    vae_id=None,
    device=None,
    dtype=None,
    load_external_vae=None,
    scheduler_cls=DDPMScheduler,
):
    if device is None or dtype is None:
        device, dtype = get_device_and_dtype()

    load_external_vae = resolve_vae_loading_policy(
        model_id=model_id,
        vae_id=vae_id,
        force_external_vae=load_external_vae,
    )

    print("\n[Loading diffusion components]")
    print("Model id:", model_id)
    print("Use external VAE:", load_external_vae)

    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
    )

    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=dtype,
    ).to(device)

    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=dtype,
    ).to(device)

    scheduler_train = scheduler_cls.from_pretrained(
        model_id,
        subfolder="scheduler",
    )

    if load_external_vae:
        if vae_id is None:
            raise ValueError("vae_id must be provided when load_external_vae=True.")

        print("Loading external VAE:", vae_id)

        vae = AutoencoderKL.from_pretrained(
            vae_id,
            torch_dtype=dtype,
        ).to(device)

    else:
        print("Loading checkpoint VAE from:", model_id)

        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=dtype,
        ).to(device)

    vae = freeze_model(vae)
    text_encoder = freeze_model(text_encoder)
    unet = freeze_model(unet)

    scheduler_infer = DDIMScheduler.from_config(scheduler_train.config)

    components = {
        "vae": vae,
        "unet": unet,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "scheduler_train": scheduler_train,
        "scheduler_infer": scheduler_infer,
    }

    print("\n[Loaded Components]")
    print("Tokenizer:", type(tokenizer))
    print("Text encoder:", type(text_encoder))
    print("UNet:", type(unet))
    print("Scheduler train:", type(scheduler_train))
    print("Scheduler infer:", type(scheduler_infer))
    print("VAE:", type(vae))

    print("\n[Config Checks]")
    print("Bundle model id:", model_id)
    print("Bundle VAE id:", vae_id if load_external_vae else f"{model_id}/vae")
    print("VAE scaling factor:", vae.config.scaling_factor)
    print("UNet in_channels:", unet.config.in_channels)
    print("UNet cross_attention_dim:", unet.config.cross_attention_dim)
    print("Scheduler num_train_timesteps:", scheduler_train.config.num_train_timesteps)

    return components


# ============================================================
# BUILD SINGLE DIFFUSION BUNDLE
# ============================================================

def build_diffusion_bundle(
    name,
    model_id,
    vae_id=None,
    device=None,
    dtype=None,
    load_external_vae=None,
):
    components = load_diffusion_components(
        model_id=model_id,
        vae_id=vae_id,
        device=device,
        dtype=dtype,
        load_external_vae=load_external_vae,
    )

    bundle = {
        "name": name,
        "model_id": model_id,
        "vae_id": vae_id,
        "uses_external_vae": resolve_vae_loading_policy(
            model_id=model_id,
            vae_id=vae_id,
            force_external_vae=load_external_vae,
        ),
        **components,
    }

    return bundle


# ============================================================
# BUILD GLOBAL + LOCAL BUNDLES
# ============================================================

def build_global_local_bundles(
    global_model_id,
    global_vae_id,
    local_model_id,
    local_vae_id=None,
    device=None,
    dtype=None,
    print_memory=True,
):
    """
    Builds both bundles.

    Key behavior:
        - GLOBAL always uses global_vae_id if global_model_id is noVAE.
        - LOCAL automatically uses an external VAE if local_model_id is noVAE.
        - If local_vae_id is None and LOCAL is noVAE, it reuses global_vae_id.
        - Therefore, changing only LOCAL_MODEL_ID to RealisticVision noVAE works.

    Example:
        LOCAL_MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"

        global_bundle, local_bundle = build_global_local_bundles(
            global_model_id=GLOBAL_MODEL_ID,
            global_vae_id=GLOBAL_VAE_ID,
            local_model_id=LOCAL_MODEL_ID,
            device=device,
            dtype=dtype,
            print_memory=True,
        )
    """
    if device is None or dtype is None:
        device, dtype = get_device_and_dtype()

    global_uses_external_vae = resolve_vae_loading_policy(
        model_id=global_model_id,
        vae_id=global_vae_id,
        force_external_vae=None,
    )

    if local_vae_id is None and is_no_vae_checkpoint(local_model_id):
        local_vae_id = global_vae_id

    local_uses_external_vae = resolve_vae_loading_policy(
        model_id=local_model_id,
        vae_id=local_vae_id,
        force_external_vae=None,
    )

    print("\n" + "=" * 80)
    print("BUILDING GLOBAL + LOCAL BUNDLES")
    print("=" * 80)
    print("Global model id:", global_model_id)
    print("Global VAE id:", global_vae_id)
    print("Global uses external VAE:", global_uses_external_vae)
    print("-" * 80)
    print("Local model id:", local_model_id)
    print("Local VAE id:", local_vae_id)
    print("Local uses external VAE:", local_uses_external_vae)
    print("=" * 80)

    global_bundle = build_diffusion_bundle(
        name=make_bundle_name("Global", global_model_id),
        model_id=global_model_id,
        vae_id=global_vae_id,
        device=device,
        dtype=dtype,
        load_external_vae=global_uses_external_vae,
    )

    if print_memory:
        print_gpu_mem(prefix="[After loading GLOBAL bundle] ")

    local_bundle = build_diffusion_bundle(
        name=make_bundle_name("Local", local_model_id),
        model_id=local_model_id,
        vae_id=local_vae_id,
        device=device,
        dtype=dtype,
        load_external_vae=local_uses_external_vae,
    )

    if print_memory:
        print_gpu_mem(prefix="[After loading LOCAL bundle] ")

    print("\n[OK] Built global_bundle and local_bundle")
    print("Global bundle name:", global_bundle["name"])
    print("Local bundle name:", local_bundle["name"])

    return global_bundle, local_bundle


# ============================================================
# APPLY ADAPTER TO EXISTING BUNDLE
# ============================================================

def apply_adapter_to_existing_bundle(
    bundle,
    adapter_type,
    rank,
    alpha,
    dropout,
    target_suffixes=("to_q", "to_k", "to_v", "to_out.0"),
    freeze_before_injection=True,
    train_mode=True,
    verbose=True,
):
    """
    Applies LoRA or DoRA to an already instantiated bundle.

    IMPORTANT:
        - This function DOES NOT reload the UNet.
        - This function DOES NOT call from_pretrained.
        - It modifies bundle["unet"] in-place.
    """

    if "unet" not in bundle:
        raise KeyError("bundle must contain key 'unet'.")

    adapter_type = adapter_type.lower().strip()

    if adapter_type not in {"lora", "dora"}:
        raise ValueError(
            f"Unsupported adapter_type: {adapter_type}. "
            "Use 'lora' or 'dora'."
        )

    print(f"\n[Applying {adapter_type.upper()} to existing UNet]")
    print("Bundle:", bundle.get("name", "Unnamed bundle"))
    print("Model id:", bundle.get("model_id", None))
    print("Rank:", rank)
    print("Alpha:", alpha)
    print("Dropout:", dropout)
    print("Targets:", list(target_suffixes))

    if freeze_before_injection:
        freeze_all_parameters(bundle["unet"])

    if adapter_type == "lora":
        bundle["unet"] = inject_manual_lora_unet(
            unet=bundle["unet"],
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_suffixes=list(target_suffixes),
            verbose=verbose,
        )

    elif adapter_type == "dora":
        bundle["unet"] = inject_manual_dora_unet(
            unet=bundle["unet"],
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_suffixes=list(target_suffixes),
            verbose=verbose,
        )

    if train_mode:
        bundle["unet"].train()
    else:
        bundle["unet"].eval()

    cast_trainable_parameters_to_fp32(bundle["unet"], verbose=verbose)

    bundle["adapter_type"] = adapter_type
    bundle["adapter_config"] = {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "target_suffixes": list(target_suffixes),
    }

    return bundle


# ============================================================
# BUILD OPTIMIZER FOR EXISTING BUNDLE
# ============================================================

def build_optimizer_for_existing_bundle(
    bundle,
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    optimizer_cls=torch.optim.AdamW,
):
    """
    Builds optimizer only over trainable parameters.

    If LoRA/DoRA injection is correct, this optimizer only sees adapter params.
    """

    if "unet" not in bundle:
        raise KeyError("bundle must contain key 'unet'.")

    params, names = get_trainable_params_and_names(bundle["unet"])

    if len(params) == 0:
        raise ValueError(
            f"No trainable parameters found for bundle: "
            f"{bundle.get('name', 'Unnamed bundle')}. "
            "Check adapter injection."
        )

    optimizer = optimizer_cls(
        params,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )

    bundle["optimizer"] = optimizer
    bundle["trainable_param_names"] = names

    print("\n[Optimizer created]")
    print("Bundle:", bundle.get("name", "Unnamed bundle"))
    print("Adapter:", bundle.get("adapter_type", None))
    print("Trainable tensors:", len(params))
    print("LR:", lr)
    print("Weight decay:", weight_decay)

    return optimizer, names


# ============================================================
# PRINT BUNDLE REPORT
# ============================================================

def print_bundle_report(
    bundle,
    title=None,
    max_items=40,
    print_names=True,
):
    if title is None:
        title = bundle.get("name", "Unnamed bundle")

    stats = print_parameter_report(
        title=title,
        model=bundle["unet"],
    )

    if print_names:
        print_trainable_report(
            f"{title} trainable report",
            bundle["unet"],
        )

        list_trainable_names(
            bundle["unet"],
            max_items=max_items,
        )

    bundle["param_stats"] = stats

    return stats


# ============================================================
# FULL MIXED SETUP WITHOUT RELOADING ANY UNET
# ============================================================

def build_mixed_lora_dora_training_setup(
    global_bundle,
    local_bundle,
    global_adapter_config=None,
    local_adapter_config=None,
    optimizer_config=None,
    global_optimizer_config=None,
    local_optimizer_config=None,
    freeze_before_injection=True,
    print_memory=True,
    print_reports=True,
    verbose=True,
):
    """
    Builds the full mixed regime:

        GLOBAL branch:
            Existing global_bundle["unet"] + LoRA

        LOCAL branch:
            Existing local_bundle["unet"] + DoRA

    IMPORTANT:
        - This function DOES NOT reload any UNet.
        - This function DOES NOT call from_pretrained.
        - This function DOES NOT create new UNet objects.
        - It modifies the already-instantiated UNets in-place.
    """

    if global_adapter_config is None:
        global_adapter_config = {
            "adapter_type": "lora",
            "rank": 8,
            "alpha": 8,
            "dropout": 0.05,
            "target_suffixes": ["to_q", "to_k", "to_v", "to_out.0"],
        }

    if local_adapter_config is None:
        local_adapter_config = {
            "adapter_type": "dora",
            "rank": 16,
            "alpha": 16,
            "dropout": 0.05,
            "target_suffixes": ["to_q", "to_k", "to_v", "to_out.0"],
        }

    using_recommended_optimizer_defaults = optimizer_config is None
    if optimizer_config is None:
        optimizer_config = {
            "lr": 7e-5,
            "betas": (0.9, 0.999),
            "weight_decay": 1e-2,
        }

    if using_recommended_optimizer_defaults and global_optimizer_config is None:
        global_optimizer_config = {"lr": 5e-5}

    # Branch-specific overrides are optional. Existing calls that only pass
    # optimizer_config retain exactly the previous shared-optimizer behavior.
    global_optimizer_config = {
        **optimizer_config,
        **(global_optimizer_config or {}),
    }
    local_optimizer_config = {
        **optimizer_config,
        **(local_optimizer_config or {}),
    }

    global_adapter_name = global_adapter_config["adapter_type"].upper()
    local_adapter_name = local_adapter_config["adapter_type"].upper()

    global_bundle["name"] = make_bundle_name(
        branch_name="Mixed_Global",
        model_id=global_bundle["model_id"],
        adapter_name=global_adapter_name,
    )

    local_bundle["name"] = make_bundle_name(
        branch_name="Mixed_Local",
        model_id=local_bundle["model_id"],
        adapter_name=local_adapter_name,
    )

    if print_memory:
        print_gpu_mem(prefix="[Before adapter injection] ")

    # --------------------------------------------------------
    # GLOBAL: adapter on existing UNet
    # --------------------------------------------------------
    global_bundle = apply_adapter_to_existing_bundle(
        bundle=global_bundle,
        adapter_type=global_adapter_config["adapter_type"],
        rank=global_adapter_config["rank"],
        alpha=global_adapter_config["alpha"],
        dropout=global_adapter_config["dropout"],
        target_suffixes=global_adapter_config["target_suffixes"],
        freeze_before_injection=freeze_before_injection,
        train_mode=True,
        verbose=verbose,
    )

    if print_memory:
        print_gpu_mem(prefix="[After GLOBAL adapter injection] ")

    # --------------------------------------------------------
    # LOCAL: adapter on existing UNet
    # --------------------------------------------------------
    local_bundle = apply_adapter_to_existing_bundle(
        bundle=local_bundle,
        adapter_type=local_adapter_config["adapter_type"],
        rank=local_adapter_config["rank"],
        alpha=local_adapter_config["alpha"],
        dropout=local_adapter_config["dropout"],
        target_suffixes=local_adapter_config["target_suffixes"],
        freeze_before_injection=freeze_before_injection,
        train_mode=True,
        verbose=verbose,
    )

    if print_memory:
        print_gpu_mem(prefix="[After LOCAL adapter injection] ")

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------
    global_optimizer, global_trainable_names = build_optimizer_for_existing_bundle(
        bundle=global_bundle,
        lr=global_optimizer_config["lr"],
        betas=global_optimizer_config["betas"],
        weight_decay=global_optimizer_config["weight_decay"],
    )

    local_optimizer, local_trainable_names = build_optimizer_for_existing_bundle(
        bundle=local_bundle,
        lr=local_optimizer_config["lr"],
        betas=local_optimizer_config["betas"],
        weight_decay=local_optimizer_config["weight_decay"],
    )

    # --------------------------------------------------------
    # Final reports
    # --------------------------------------------------------
    if print_reports:
        print("\n" + "=" * 80)
        print("FINAL PARAMETER REPORT")
        print("=" * 80)

        global_stats = print_bundle_report(
            bundle=global_bundle,
            title=f"GLOBAL UNet / {global_bundle['model_id']} / {global_adapter_name}",
            max_items=40,
            print_names=True,
        )

        local_stats = print_bundle_report(
            bundle=local_bundle,
            title=f"LOCAL UNet / {local_bundle['model_id']} / {local_adapter_name}",
            max_items=40,
            print_names=True,
        )

        global_bundle["param_stats"] = global_stats
        local_bundle["param_stats"] = local_stats

    print("\n[OK] Mixed LoRA/DoRA setup built WITHOUT reloading UNets")
    print("  mixed_global_bundle['optimizer'] -> GLOBAL optimizer")
    print("  mixed_local_bundle['optimizer']  -> LOCAL optimizer")

    return global_bundle, local_bundle
