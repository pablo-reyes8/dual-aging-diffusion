import gc
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from src.utils.cuda_utils import * 
from src.score_net.arquitecture import * 
from pathlib import Path


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def extract_state_dict_from_checkpoint(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """
    Robustly extracts a model state_dict from common checkpoint formats.

    Supported formats:
        1. raw state_dict
        2. {"model_state_dict": ...}
        3. {"state_dict": ...}
        4. {"score_net_state_dict": ...}
        5. {"model": ...}
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must be a dict or state_dict-like object. Got {type(checkpoint)}"
        )

    candidate_keys = [
        "model_state_dict",
        "state_dict",
        "score_net_state_dict",
        "score_net",
        "model",
        "net"]

    for key in candidate_keys:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    # If all values look like tensors, assume this is already a raw state_dict.
    if all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise KeyError(
        "Could not find a valid state_dict inside checkpoint. "
        f"Available keys: {list(checkpoint.keys())}"
    )


def strip_common_prefixes_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    prefixes=("module.", "score_net.", "model.", "net.")) -> Dict[str, torch.Tensor]:
    """
    Removes common prefixes caused by DataParallel or wrapper modules.
    """
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        cleaned[new_key] = value

    return cleaned


def load_score_net_safely(
    checkpoint_path: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    in_channels: int = 3,
    base_channels: int = 32,
    dropout: float = 0.15,
    strict: bool = True,
    freeze: bool = True,
    eval_mode: bool = True,
    print_report: bool = True):

    """
    Safely creates and loads LocalScoreNet.

    Args:
        checkpoint_path:
            Path to .pt/.pth checkpoint.

        device:
            "cuda" or "cpu". If None, auto-detects.

        dtype:
            Usually torch.float32 for ScoreNet.

        strict:
            If True, requires exact state_dict match.
            If False, allows missing/unexpected keys.

        freeze:
            If True, disables gradients for all ScoreNet parameters.
            Recommended when using ScoreNet only as frozen loss model.

        eval_mode:
            If True, puts ScoreNet in eval mode.
            Recommended for loss/evaluation.

    Returns:
        score_net
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ScoreNet checkpoint not found: {checkpoint_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    if print_report:
        print("\n========== SAFE SCORENET LOAD ==========")
        print("checkpoint_path:", checkpoint_path)
        print("target device:", device)
        print("dtype:", dtype)
        if torch.cuda.is_available():
            print_gpu_mem("[Before loading ScoreNet] ")

    # --------------------------------------------------------
    # Cleanup before loading
    # --------------------------------------------------------
    cleanup_memory()

    # --------------------------------------------------------
    #  Load checkpoint on CPU only
    # --------------------------------------------------------
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu")

    state_dict = extract_state_dict_from_checkpoint(checkpoint)
    state_dict = strip_common_prefixes_from_state_dict(state_dict)

    # --------------------------------------------------------
    # Instantiate model on CPU
    # --------------------------------------------------------
    score_net = LocalScoreNet(
        in_channels=in_channels,
        base_channels=base_channels,
        dropout=dropout,)

    # --------------------------------------------------------
    # Load weights on CPU
    # --------------------------------------------------------
    load_result = score_net.load_state_dict(
        state_dict,
        strict=strict,)

    # --------------------------------------------------------
    # Delete checkpoint before moving model to GPU
    # --------------------------------------------------------
    del checkpoint
    del state_dict
    cleanup_memory()

    # --------------------------------------------------------
    # Move model to target device
    # --------------------------------------------------------
    score_net = score_net.to(
        device=device,
        dtype=dtype)

    # --------------------------------------------------------
    # Freeze/eval if requested
    # --------------------------------------------------------
    if freeze:
        for p in score_net.parameters():
            p.requires_grad_(False)

    if eval_mode:
        score_net.eval()
    else:
        score_net.train()

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------
    if print_report:
        print("\n[OK] ScoreNet loaded safely")
        print("freeze:", freeze)
        print("eval_mode:", eval_mode)

        if not strict:
            print("\n[Non-strict load result]")
            print("missing_keys:", load_result.missing_keys)
            print("unexpected_keys:", load_result.unexpected_keys)

        total_params = sum(p.numel() for p in score_net.parameters())
        trainable_params = sum(p.numel() for p in score_net.parameters() if p.requires_grad)

        print("\n[ScoreNet params]")
        print(f"total params:     {total_params:,}")
        print(f"trainable params: {trainable_params:,}")
        print(f"frozen params:    {total_params - trainable_params:,}")

        if torch.cuda.is_available():
            print_gpu_mem("[After loading ScoreNet] ")

    return score_net