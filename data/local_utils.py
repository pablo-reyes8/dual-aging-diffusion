import torch 
from data.local_path_dataset import *


def audit_sampled_loader_distribution(loader, n_batches=100):
    all_scores_aug = []
    all_scores_original = []
    all_regions = []

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        all_scores_aug.extend(batch["score_raw"].detach().cpu().tolist())
        all_scores_original.extend(batch["score_original"].detach().cpu().tolist())
        all_regions.extend(batch["region_key"])

    scores_aug = torch.tensor(all_scores_aug, dtype=torch.float32)
    scores_original = torch.tensor(all_scores_original, dtype=torch.float32)

    def print_bins(scores, title):
        print(f"\n[{title}]")

        bins = [
            (0, 10),
            (10, 25),
            (25, 40),
            (40, 60),
            (60, 75),
            (75, 90),
            (90, 101)]

        total = len(scores)

        for low, high in bins:
            if high == 101:
                mask = (scores >= low) & (scores <= 100)
                label = f"{low:02d}-100"
            else:
                mask = (scores >= low) & (scores < high)
                label = f"{low:02d}-{high:02d}"

            count = int(mask.sum().item())
            pct = 100.0 * count / max(total, 1)
            bar = "█" * int(round(pct / 2.0))
            print(f"{label:8s}: {count:4d} | {pct:6.2f}% | {bar}")

    print("\n========== SAMPLED LOADER AUDIT ==========")
    print("Inspected samples:", len(scores_aug))

    print(
        "\n[Original score stats inside sampled loader]"
        f"\nmin/mean/median/max/std: "
        f"{scores_original.min().item():.1f} / "
        f"{scores_original.mean().item():.1f} / "
        f"{scores_original.median().item():.1f} / "
        f"{scores_original.max().item():.1f} / "
        f"{scores_original.std(unbiased=False).item():.1f}")

    print(
        "\n[Augmented score stats inside sampled loader]"
        f"\nmin/mean/median/max/std: "
        f"{scores_aug.min().item():.1f} / "
        f"{scores_aug.mean().item():.1f} / "
        f"{scores_aug.median().item():.1f} / "
        f"{scores_aug.max().item():.1f} / "
        f"{scores_aug.std(unbiased=False).item():.1f}")

    print_bins(scores_original, "Original scores sampled by loader")
    print_bins(scores_aug, "Augmented/jittered scores seen by model")

    print("\n[Important ranges: augmented scores]")
    for label, mask in {
        "score <= 10": scores_aug <= 10,
        "score < 25": scores_aug < 25,
        "score >= 75": scores_aug >= 75,
        "score >= 90": scores_aug >= 90,
        "score == 0": scores_aug == 0,
        "score == 100": scores_aug == 100}.items():

        count = int(mask.sum().item())
        pct = 100.0 * count / max(len(scores_aug), 1)
        print(f"{label:15s}: {count:4d} | {pct:6.2f}%")

    print("\n[Sampled regions]")
    region_counts = Counter(all_regions)

    for region, count in region_counts.most_common():
        pct = 100.0 * count / max(len(all_regions), 1)
        print(f"{region:32s}: {count:4d} | {pct:6.2f}%")


def denorm_for_plot(x: torch.Tensor) -> torch.Tensor:
    """
    Converts image tensor from [-1, 1] to [0, 1].
    """
    return (x * 0.5 + 0.5).clamp(0, 1)


def show_local_batch(batch: Dict[str, Any], n: int = 8, print_prompts: bool = True):
    n = min(n, batch["pixel_values"].shape[0])

    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4))
    if n == 1:
        axes = [axes]

    for i in range(n):
        img = denorm_for_plot(batch["pixel_values"][i]).permute(1, 2, 0).cpu().numpy()
        region = batch["region_key"][i]
        score_o = batch["score_original"][i].item()
        score_a = batch["score_raw"][i].item()

        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(
            f"{region}\norig={score_o:.0f} aug={score_a:.0f}",
            fontsize=8,
        )

    plt.tight_layout()
    plt.show()

    if print_prompts:
        print("\n[Prompts]")
        for i in range(n):
            print(f"{i}: {batch['prompt'][i]}")


# ============================================================
# Visual audit by specific region
# ============================================================

def make_region_loader(
    samples: List[Dict[str, Any]],
    region_key: str,
    batch_size: int = 8,
    train: bool = False,
    virtual_repeats: int = 1):
    region_samples = [s for s in samples if s["region_key"] == region_key]

    if len(region_samples) == 0:
        raise ValueError(f"No samples for region_key={region_key}")

    dataset = LocalAgingCropDataset(
        samples=region_samples,
        resolution=LOCAL_RESOLUTION,
        context_scale=1.20,
        virtual_repeats=virtual_repeats,
        train=train,
        jitter_scores_enabled=train,
        enable_horizontal_flip=False,
        normalize_for_diffusion=True,
        drop_regions=None)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        collate_fn=local_collate_fn,
        pin_memory=True,
        drop_last=False)

    return loader


def audit_all_regions(samples: List[Dict[str, Any]], n: int = 8):
    for region_key in REGION_ORDER:
        region_samples = [s for s in samples if s["region_key"] == region_key]
        if len(region_samples) == 0:
            print(f"[SKIP] {region_key}: no samples")
            continue

        print(f"\n========== REGION: {region_key} | samples={len(region_samples)} ==========")
        loader = make_region_loader(
            samples=samples,
            region_key=region_key,
            batch_size=n,
            train=False,
            virtual_repeats=1)

        region_batch = next(iter(loader))
        show_local_batch(region_batch, n=n, print_prompts=True)

# ============================================================
#  Dataset audit summary
# ============================================================

def dataset_audit(samples: List[Dict[str, Any]]):
    print("\n========== DATASET AUDIT ==========")

    print("\n[Samples]")
    print("Total crop samples:", len(samples))
    print("Unique images:", len(set(s["image_id"] for s in samples)))

    # ============================================================
    # Counts by region
    # ============================================================

    print("\n[Counts by region]")
    region_counts = Counter(s["region_key"] for s in samples)
    for region_key in REGION_ORDER:
        print(f"{region_key:32s}: {region_counts.get(region_key, 0)}")

    # ============================================================
    # Global score distribution
    # ============================================================

    all_scores = torch.tensor(
        [float(s["score_raw"]) for s in samples],
        dtype=torch.float32,
    )

    print("\n[Global score stats]")
    print(f"n:      {len(all_scores)}")
    print(f"min:    {all_scores.min().item():6.2f}")
    print(f"mean:   {all_scores.mean().item():6.2f}")
    print(f"median: {all_scores.median().item():6.2f}")
    print(f"max:    {all_scores.max().item():6.2f}")
    print(f"std:    {all_scores.std(unbiased=False).item():6.2f}")

    quantiles = torch.quantile(
        all_scores,
        torch.tensor([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]),
    )

    print("\n[Global score percentiles]")
    for q, v in zip([1, 5, 10, 25, 50, 75, 90, 95, 99], quantiles):
        print(f"p{q:02d}: {v.item():6.2f}")

    # ============================================================
    # Score bins
    # ============================================================

    bins = [
        (0, 10),
        (10, 25),
        (25, 40),
        (40, 60),
        (60, 75),
        (75, 90),
        (90, 101),
    ]

    print("\n[Global score bins]")
    total = len(all_scores)

    global_bin_counts = {}

    for low, high in bins:
        if high == 101:
            mask = (all_scores >= low) & (all_scores <= 100)
            label = f"{low:02d}-100"
        else:
            mask = (all_scores >= low) & (all_scores < high)
            label = f"{low:02d}-{high:02d}"

        count = int(mask.sum().item())
        pct = 100.0 * count / max(total, 1)
        global_bin_counts[label] = count

        bar = "█" * int(round(pct / 2.0))
        print(f"{label:8s}: {count:4d} | {pct:6.2f}% | {bar}")

    # ============================================================
    # Important aging ranges
    # ============================================================

    print("\n[Important score ranges]")
    ranges = {
        "very_low     score <= 10": all_scores <= 10,
        "low          score <  25": all_scores < 25,
        "middle       40 <= score < 60": (all_scores >= 40) & (all_scores < 60),
        "high         score >= 75": all_scores >= 75,
        "very_high    score >= 90": all_scores >= 90,
        "exact_zero   score == 0": all_scores == 0,
        "exact_100    score == 100": all_scores == 100,
    }

    for name, mask in ranges.items():
        count = int(mask.sum().item())
        pct = 100.0 * count / max(total, 1)
        print(f"{name:30s}: {count:4d} | {pct:6.2f}%")

    # ============================================================
    # Score stats by region
    # ============================================================

    print("\n[Score stats by region]")
    by_region = defaultdict(list)

    for s in samples:
        by_region[s["region_key"]].append(float(s["score_raw"]))

    for region_key in REGION_ORDER:
        values = by_region.get(region_key, [])
        if not values:
            continue

        values_t = torch.tensor(values, dtype=torch.float32)

        q25 = torch.quantile(values_t, 0.25).item()
        q50 = torch.quantile(values_t, 0.50).item()
        q75 = torch.quantile(values_t, 0.75).item()

        print(
            f"{region_key:32s} "
            f"n={len(values):3d} "
            f"min={values_t.min().item():5.1f} "
            f"mean={values_t.mean().item():5.1f} "
            f"p25={q25:5.1f} "
            f"p50={q50:5.1f} "
            f"p75={q75:5.1f} "
            f"max={values_t.max().item():5.1f}"
        )

    # ============================================================
    # Score bins by region
    # ============================================================

    print("\n[Score bins by region]")
    bin_labels = []
    for low, high in bins:
        if high == 101:
            bin_labels.append(f"{low:02d}-100")
        else:
            bin_labels.append(f"{low:02d}-{high:02d}")

    header = f"{'region':32s} " + " ".join([f"{b:>8s}" for b in bin_labels])
    print(header)
    print("-" * len(header))

    for region_key in REGION_ORDER:
        values = by_region.get(region_key, [])
        if not values:
            continue

        values_t = torch.tensor(values, dtype=torch.float32)

        row_counts = []

        for low, high in bins:
            if high == 101:
                mask = (values_t >= low) & (values_t <= 100)
            else:
                mask = (values_t >= low) & (values_t < high)

            row_counts.append(int(mask.sum().item()))

        row = f"{region_key:32s} " + " ".join([f"{c:8d}" for c in row_counts])
        print(row)

    # ============================================================
    # High-score concentration by region
    # ============================================================

    print("\n[High-score concentration by region]")
    print(f"{'region':32s} {'n':>5s} {'>=75':>6s} {'>=90':>6s} {'>=75 %':>8s} {'>=90 %':>8s}")
    print("-" * 72)

    for region_key in REGION_ORDER:
        values = by_region.get(region_key, [])
        if not values:
            continue

        values_t = torch.tensor(values, dtype=torch.float32)
        n = len(values_t)

        n75 = int((values_t >= 75).sum().item())
        n90 = int((values_t >= 90).sum().item())

        p75 = 100.0 * n75 / max(n, 1)
        p90 = 100.0 * n90 / max(n, 1)

        print(f"{region_key:32s} {n:5d} {n75:6d} {n90:6d} {p75:8.2f} {p90:8.2f}")

    # ============================================================
    # Demographic contexts
    # ============================================================

    print("\n[Demographic contexts]")
    demo_counts = Counter(s["demographic_context"] for s in samples)

    for k, v in demo_counts.most_common():
        print(f"{k:30s}: {v}")


def count_dataset_sizes(local_samples, train_samples=None, val_samples=None, train_dataset=None, val_dataset=None):
    print("========== SIZE AUDIT ==========")

    # Real full images
    unique_images_all = sorted(set(s["image_id"] for s in local_samples))
    print(f"Real unique images total: {len(unique_images_all)}")

    # Real annotated crops
    print(f"Real annotated local crops total: {len(local_samples)}")

    # Counts by region before virtual augmentation
    print("\nReal crops by region:")
    region_counts = Counter(s["region_key"] for s in local_samples)
    for k, v in region_counts.items():
        print(f"  {k:32s}: {v}")

    if train_samples is not None:
        train_images = sorted(set(s["image_id"] for s in train_samples))
        print("\nTrain split:")
        print(f"  Real unique images: {len(train_images)}")
        print(f"  Real crops:         {len(train_samples)}")

    if val_samples is not None:
        val_images = sorted(set(s["image_id"] for s in val_samples))
        print("\nValidation split:")
        print(f"  Real unique images: {len(val_images)}")
        print(f"  Real crops:         {len(val_samples)}")

    if train_dataset is not None:
        print("\nVirtual train dataset:")
        print(f"  virtual_repeats:    {train_dataset.virtual_repeats}")
        print(f"  Virtual crops/epoch:{len(train_dataset)}")

    if val_dataset is not None:
        print("\nVirtual validation dataset:")
        print(f"  virtual_repeats:    {val_dataset.virtual_repeats}")
        print(f"  Virtual crops/epoch:{len(val_dataset)}")

    print("\nImportant:")
    print("  No augmented images were saved to disk.")
    print("  Augmentations are generated online every time __getitem__ is called.")