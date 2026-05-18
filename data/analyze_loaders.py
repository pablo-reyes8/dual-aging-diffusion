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


def analyze_local(batch_size: int, num_workers: int, n_batches: int) -> None:
    objects = build_local_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
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


def analyze_global(batch_size: int, num_workers: int, n_batches: int) -> None:
    objects = build_global_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
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
