"""
Run detailed audits for the local/global dataloaders.

Examples:
    python -m data.analyze_loaders --branch local --num-workers 0
    python -m data.analyze_loaders --branch global --num-workers 0
    python -m data.analyze_loaders --branch both --n-batches 50
"""

import argparse

from data.create_data import build_global_dataloaders, build_local_dataloaders
from data.global_utils import audit_global_loader_distribution, audit_global_samples
from data.local_utils import (
    audit_sampled_loader_distribution,
    count_dataset_sizes,
    dataset_audit,
)
import matplotlib.pyplot as plt
import torch 

def analyze_local(batch_size: int, num_workers: int, n_batches: int, skip_regions=None) -> None:
    objects = build_local_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        skip_regions=skip_regions,
    )

    count_dataset_sizes(
        local_samples=objects["samples"],
        train_samples=objects["train_samples"],
        val_samples=objects["val_samples"],
        train_dataset=objects["train_dataset"],
        val_dataset=objects["val_dataset"],
    )
    dataset_audit(objects["samples"])
    audit_sampled_loader_distribution(objects["train_loader"], n_batches=n_batches)

    batch = next(iter(objects["train_loader"]))
    print("\n========== LOCAL BATCH KEYS ==========")
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"{key:20s}: {tuple(value.shape)}")
        else:
            print(f"{key:20s}: {type(value).__name__} len={len(value)}")


def analyze_global(batch_size: int, num_workers: int, n_batches: int, skip_regions=None) -> None:
    objects = build_global_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        skip_regions=skip_regions,
    )

    audit_global_samples(objects["samples"])
    audit_global_loader_distribution(objects["val_loader"], n_batches=n_batches)

    batch = next(iter(objects["val_loader"]))
    print("\n========== GLOBAL BATCH KEYS ==========")
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"{key:20s}: {tuple(value.shape)}")
        else:
            print(f"{key:20s}: {type(value).__name__} len={len(value)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--branch",
        choices=["local", "global", "both"],
        default="local",
        help="Which dataloader branch to audit.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-batches", type=int, default=100)
    return parser.parse_args()

def summarize_batch(batch, name="batch", max_items=3):
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    if not isinstance(batch, dict):
        print("Batch type:", type(batch))
        print(batch)
        return

    print("\n[KEYS]")
    for k in batch.keys():
        print(" -", k)

    print("\n[VALUES]")
    for k, v in batch.items():
        print(f"\n{k}:")
        print("  type:", type(v))

        if torch.is_tensor(v):
            print("  shape:", tuple(v.shape))
            print("  dtype:", v.dtype)
            print("  device:", v.device)
            if v.numel() > 0 and v.dtype.is_floating_point:
                print("  min:", float(v.min()))
                print("  max:", float(v.max()))
                print("  mean:", float(v.mean()))
            elif v.numel() > 0:
                print("  first values:", v.flatten()[:10].tolist())

        elif isinstance(v, (list, tuple)):
            print("  len:", len(v))
            for i, item in enumerate(v[:max_items]):
                print(f"  [{i}] {type(item)}:", item)

        elif isinstance(v, dict):
            print("  dict keys:", list(v.keys()))
            for kk, vv in list(v.items())[:max_items]:
                print(f"    {kk}: {type(vv)} -> {vv}")

        else:
            print("  value:", v)


def tensor_to_image(x):
    """
    Converts a CHW tensor to HWC image for matplotlib.
    Assumes either [-1, 1] or [0, 1].
    """
    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")

    x = x.detach().cpu().float()

    # If diffusion-normalized [-1, 1], map to [0, 1]
    if float(x.min()) < 0:
        x = (x + 1.0) / 2.0

    x = x.clamp(0, 1)
    x = x.permute(1, 2, 0).numpy()
    return x


def show_batch_images(batch, image_keys=None, n=4, title_prefix=""):
    """
    Tries to display image tensors from a batch.
    """
    if image_keys is None:
        image_keys = [
            "pixel_values",
            "image",
            "images",
            "global_image",
            "local_image",
            "crop",
            "crops",
        ]

    found = False

    for key in image_keys:
        if key not in batch:
            continue

        value = batch[key]

        if not torch.is_tensor(value):
            continue

        if value.ndim == 4:
            # B, C, H, W
            found = True
            b = min(n, value.shape[0])

            for i in range(b):
                plt.figure(figsize=(4, 4))
                plt.imshow(tensor_to_image(value[i]))
                plt.axis("off")
                plt.title(f"{title_prefix}{key}[{i}]")
                plt.show()

        elif value.ndim == 5:
            # B, N, C, H, W, useful for fused local crops
            found = True
            b = min(value.shape[0], 1)
            num_crops = min(n, value.shape[1])

            for bi in range(b):
                for ci in range(num_crops):
                    plt.figure(figsize=(4, 4))
                    plt.imshow(tensor_to_image(value[bi, ci]))
                    plt.axis("off")
                    plt.title(f"{title_prefix}{key}[batch={bi}, crop={ci}]")
                    plt.show()

    if not found:
        print("No obvious image tensor key found.")

def print_prompt_like_fields(batch, max_items=5):
    print("\n[PROMPT / TEXT / METADATA FIELDS]")

    keywords = [
        "prompt",
        "text",
        "caption",
        "region",
        "age",
        "score",
        "filename",
        "image",
        "stem",
        "path",
        "bbox",
        "box",
    ]

    for k, v in batch.items():
        k_lower = k.lower()
        if any(word in k_lower for word in keywords):
            print(f"\n{k}:")
            if torch.is_tensor(v):
                print("  tensor shape:", tuple(v.shape), "dtype:", v.dtype)
                print("  values:", v.flatten()[:max_items].detach().cpu().tolist())
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v[:max_items]):
                    print(f"  [{i}] {item}")
            else:
                print(" ", v)


if __name__ == "__main__":
    args = parse_args()

    if args.branch in {"local", "both"}:
        analyze_local(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_batches=args.n_batches,
        )

    if args.branch in {"global", "both"}:
        analyze_global(
            batch_size=min(args.batch_size, 4),
            num_workers=args.num_workers,
            n_batches=args.n_batches,
        )
