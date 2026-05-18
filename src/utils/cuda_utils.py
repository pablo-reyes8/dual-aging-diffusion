import gc
import torch
import sys
try:
    from IPython import get_ipython
except ImportError:
    def get_ipython():
        return None

def clear_old_models_from_memory():
    names_to_delete = [
        "vae",
        "unet",
        "tokenizer",
        "text_encoder",
        "noise_scheduler",
        "clip_tokenizer",
        "clip_text_encoder",
        "wrapper",
        "global_train_loader",
        "global_val_loader",
        "train_loader",
        "val_loader",
        "local_batch",
        "global_batch",
        "clip_out",
        "x256", "x512", "z256", "z512",
        "x256_hat", "x512_hat",
        "text_inputs", "encoder_hidden_states",
        "latents", "noise", "timesteps", "noisy_latents", "noise_pred"
    ]

    for name in names_to_delete:
        if name in globals():
            del globals()[name]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def print_gpu_mem(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(
            f"{prefix}allocated={allocated:.2f} GB | "
            f"reserved={reserved:.2f} GB | "
            f"max={max_allocated:.2f} GB"
        )
    else:
        print(f"{prefix}CUDA not available.")


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def print_trainable_report(name, model):
    total, trainable, frozen = count_params(model)
    print(f"\n========== {name} ==========")
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {frozen:,}")
    if total > 0:
        print(f"Trainable %:      {100 * trainable / total:.6f}%")


def list_trainable_names(model, max_items=50):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"\nTrainable tensors: {len(names)}")
    for n in names[:max_items]:
        print("  ", n)
    if len(names) > max_items:
        print(f"  ... {len(names) - max_items} more")


def get_device_and_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Device:", device)
    print("Dtype:", dtype)

    return device, dtype





def hard_cleanup_after_oom():
    # --------------------------------------------------------
    # Delete likely graph/tensor variables from globals
    # --------------------------------------------------------
    names_to_delete = [
        # outputs / losses
        "loss_out", "loss", "total_loss",
        "loss_full", "loss_zone", "loss_score", "loss_cycle",

        # batches / targets
        "batch", "batch_small", "small_batch",
        "target_prompts", "target_scores",
        "source_prompts", "zone_prompts",

        # latent/image intermediates
        "pixel_values", "x", "y", "pred",
        "z0", "zt", "zt_score",
        "noise", "noise_score",
        "timesteps", "timesteps_score",
        "noise_pred", "noise_pred_full", "noise_pred_zone", "noise_pred_target",
        "x0_hat_latents", "x0_hat_target_latents",
        "decoded_score_image", "decoded_crop",

        # hidden states
        "encoder_hidden_states", "target_hidden",
        "text_inputs", "input_ids", "attention_mask"]

    for name in names_to_delete:
        if name in globals():
            try:
                del globals()[name]
            except Exception:
                pass

    # --------------------------------------------------------
    # Clear Python exception traceback references
    # This is important after CUDA OOM in notebooks.
    # --------------------------------------------------------
    sys.last_type = None
    sys.last_value = None
    sys.last_traceback = None

    # --------------------------------------------------------
    # Clear IPython output cache if available
    # The variables _, __, ___ and Out can keep tensors alive.
    # --------------------------------------------------------
    try:
        ip = get_ipython()
        ip.user_ns["_"] = None
        ip.user_ns["__"] = None
        ip.user_ns["___"] = None
        ip.user_ns["_i"] = None
        ip.user_ns["_ii"] = None
        ip.user_ns["_iii"] = None
        ip.run_line_magic("reset", "-f out")
    except Exception as e:
        print("[WARN] Could not clear IPython output cache:", e)

    # --------------------------------------------------------
    #  Python + CUDA cleanup
    # --------------------------------------------------------
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    #  Report
    # --------------------------------------------------------
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        print("========== CUDA MEMORY AFTER CLEANUP ==========")
        print(f"allocated:     {allocated:.3f} GB")
        print(f"reserved:      {reserved:.3f} GB")
        print(f"max_allocated: {max_allocated:.3f} GB")

