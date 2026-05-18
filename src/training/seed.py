import torch 
import random 
import numpy as np
import os

def set_seed(
    seed: int = 42,
    deterministic: bool = False,
    set_hash_seed: bool = True) -> None:
    """
    Fix random seeds for Python, NumPy and PyTorch.

    This is useful for:
        - dataloader split reproducibility,
        - target age / target score sampling reproducibility,
        - prompt dropout reproducibility,
        - diffusion noise sampling reproducibility during debug.

    Args:
        seed:
            Global random seed.

        deterministic:
            If True, enables deterministic CuDNN behavior.
            Useful for tests/debugging, but can slow down training.

            For long diffusion training, deterministic=False is usually better.

        set_hash_seed:
            If True, sets PYTHONHASHSEED.
            This is useful for more stable behavior involving hashing/order.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed)}")

    if set_hash_seed:
        os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Strict deterministic algorithms can break some operations.
        # Enable only for very controlled tests, not normal training.
        # torch.use_deterministic_algorithms(True)

    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(
        f"[Seed set] seed={seed} | "
        f"deterministic={deterministic} | "
        f"hash_seed={set_hash_seed}")