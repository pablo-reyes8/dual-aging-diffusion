"""
Build dataset objects and dataloaders for the local and global branches.

Examples:
    python -m data.create_data --branch local
    python -m data.create_data --branch global
    python -m data.create_data --branch both
"""

import argparse
from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader

from data.local_path_dataset import (
    JSON_DIR,
    IMAGE_DIR,
    LOCAL_RESOLUTION,
    LocalAgingCropDataset,
    build_local_samples,
    local_collate_fn,
    make_local_score_sampler,
    prepare_local_assets,
    split_samples_by_image_id,
)
from data.global_path_datasets import (
    DRIVE_ZIPS,
    GLOBAL_CSV_PATH,
    GLOBAL_IMAGE_DIR,
    GLOBAL_RESOLUTION,
    GlobalAgingFaceDataset,
    build_global_samples_from_attribute_csv,
    global_debug_collate_fn,
    global_train_collate_fn,
    prepare_global_assets,
)


def build_local_dataloaders(
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Dict[str, Any]:
    json_dir, image_dir, image_index = prepare_local_assets()
    local_samples = build_local_samples(json_dir, image_index)

    train_samples, val_samples = split_samples_by_image_id(
        local_samples,
        val_fraction=0.15,
        seed=42,
    )

    train_dataset = LocalAgingCropDataset(
        samples=train_samples,
        resolution=LOCAL_RESOLUTION,
        context_scale=1.20,
        virtual_repeats=10,
        train=True,
        jitter_scores_enabled=True,
        enable_horizontal_flip=False,
        normalize_for_diffusion=True,
        drop_regions=None,
    )

    val_dataset = LocalAgingCropDataset(
        samples=val_samples,
        resolution=LOCAL_RESOLUTION,
        context_scale=1.20,
        virtual_repeats=1,
        train=False,
        jitter_scores_enabled=False,
        enable_horizontal_flip=False,
        normalize_for_diffusion=True,
        drop_regions=None,
    )

    train_sampler = make_local_score_sampler(
        train_dataset,
        w_00_10=0.70,
        w_10_25=0.85,
        w_25_40=1.00,
        w_40_60=0.90,
        w_60_75=1.10,
        w_75_90=3.50,
        w_90_100=3.00,
        region_balance_strength=0.35,
        use_anatomical_prior=True,
        min_weight=0.50,
        max_weight=5.50,
        verbose=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=local_collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=local_collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {
        "json_dir": json_dir,
        "image_dir": image_dir,
        "image_index": image_index,
        "samples": local_samples,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


def build_global_dataloaders(
    batch_size: int = 4,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Dict[str, Any]:
    global_image_index = prepare_global_assets(
        zip_paths=DRIVE_ZIPS,
        output_dir=GLOBAL_IMAGE_DIR,
        force_reextract=False,
    )

    global_samples = build_global_samples_from_attribute_csv(
        csv_path=GLOBAL_CSV_PATH,
        image_index=global_image_index,
        filename_col="filename",
        age_col="age_pred",
        min_age=None,
        max_age=None,
        require_existing_image=True,
    )

    global_train_dataset = GlobalAgingFaceDataset(
        samples=global_samples,
        resolution=GLOBAL_RESOLUTION,
        train=True,
        normalize_for_diffusion=True,
    )

    global_val_dataset = GlobalAgingFaceDataset(
        samples=global_samples,
        resolution=GLOBAL_RESOLUTION,
        train=False,
        normalize_for_diffusion=True,
    )

    global_train_loader = DataLoader(
        global_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=global_train_collate_fn,
        pin_memory=pin_memory,
    )

    global_val_loader = DataLoader(
        global_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=global_debug_collate_fn,
        pin_memory=pin_memory,
    )

    return {
        "image_dir": GLOBAL_IMAGE_DIR,
        "image_index": global_image_index,
        "samples": global_samples,
        "train_dataset": global_train_dataset,
        "val_dataset": global_val_dataset,
        "train_loader": global_train_loader,
        "val_loader": global_val_loader,
    }


def print_local_sanity(objects: Dict[str, Any]) -> None:
    print("\n[Local paths]")
    print("JSON_DIR:", objects["json_dir"])
    print("IMAGE_DIR:", objects["image_dir"])

    print("\n[Local dataset sizes]")
    print("train_dataset virtual size:", len(objects["train_dataset"]))
    print("val_dataset size:", len(objects["val_dataset"]))

    batch = next(iter(objects["train_loader"]))
    print("\n[Local batch check]")
    print("pixel_values:", batch["pixel_values"].shape)
    print("score:", batch["score"].shape)
    print("prompt[0]:", batch["prompt"][0])
    print("zone_prompt[0]:", batch["zone_prompt"][0])
    print("region_key[0]:", batch["region_key"][0])
    print(
        "score original -> augmented:",
        batch["score_original"][0].item(),
        "->",
        batch["score_raw"][0].item(),
    )


def print_global_sanity(objects: Dict[str, Any]) -> None:
    print("\n[Global paths]")
    print("GLOBAL_IMAGE_DIR:", objects["image_dir"])
    print("GLOBAL_CSV_PATH:", GLOBAL_CSV_PATH)

    batch = next(iter(objects["val_loader"]))
    print("\n[Global batch check]")
    print("pixel_values:", batch["pixel_values"].shape)
    print("age:", batch["age"].shape)
    print("age_norm:", batch["age_norm"].shape)
    print("prompt example:", batch["prompt"][0])
    print("gender label example:", batch["gender_label"][0])
    print("gender confidence example:", batch["gender_confidence"][0])
    print("filename example:", batch["filename"][0])


def build_requested(branch: str, batch_size: int, num_workers: int) -> Tuple[Any, Any]:
    local_objects = None
    global_objects = None

    if branch in {"local", "both"}:
        local_objects = build_local_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
        )
        print_local_sanity(local_objects)

    if branch in {"global", "both"}:
        global_objects = build_global_dataloaders(
            batch_size=min(batch_size, 4),
            num_workers=num_workers,
        )
        print_global_sanity(global_objects)

    return local_objects, global_objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--branch",
        choices=["local", "global", "both"],
        default="local",
        help="Which dataloader branch to build. Default keeps local PC usage working.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_requested(
        branch=args.branch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
