
from collections import Counter
from typing import Any, Dict, List

import torch 
import matplotlib.pyplot as plt


def denorm_for_plot(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).clamp(0, 1)


def show_global_batch(batch: Dict[str, Any], n: int = 4):
    n = min(n, batch["pixel_values"].shape[0])

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5))
    if n == 1:
        axes = [axes]

    for i in range(n):
        img = denorm_for_plot(batch["pixel_values"][i]).permute(1, 2, 0).cpu().numpy()
        age = batch["age"][i].item()
        prompt = batch["prompt"][i]

        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"age={age:.1f}", fontsize=9)

    plt.tight_layout()
    plt.show()

    print("\n[Prompts]")
    for i in range(n):
        print(f"{i}: {batch['prompt'][i]}")


def audit_global_samples(samples: List[Dict[str, Any]]) -> None:
    print("\n========== GLOBAL DATASET AUDIT ==========")
    print("\n[Samples]")
    print("Total full-face samples:", len(samples))

    ages = torch.tensor([float(s["age"]) for s in samples], dtype=torch.float32)
    print("\n[Age stats]")
    print(f"n:      {len(ages)}")
    print(f"min:    {ages.min().item():6.2f}")
    print(f"mean:   {ages.mean().item():6.2f}")
    print(f"median: {ages.median().item():6.2f}")
    print(f"max:    {ages.max().item():6.2f}")
    print(f"std:    {ages.std(unbiased=False).item():6.2f}")

    quantiles = torch.quantile(
        ages,
        torch.tensor([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]),
    )
    print("\n[Age percentiles]")
    for q, v in zip([1, 5, 10, 25, 50, 75, 90, 95, 99], quantiles):
        print(f"p{q:02d}: {v.item():6.2f}")

    bins = [(0, 10), (10, 18), (18, 30), (30, 45), (45, 60), (60, 75), (75, 101)]
    print("\n[Age bins]")
    for low, high in bins:
        if high == 101:
            mask = (ages >= low) & (ages <= 100)
            label = f"{low:02d}-100"
        else:
            mask = (ages >= low) & (ages < high)
            label = f"{low:02d}-{high:02d}"
        count = int(mask.sum().item())
        pct = 100.0 * count / max(len(ages), 1)
        bar = "#" * int(round(pct / 2.0))
        print(f"{label:8s}: {count:5d} | {pct:6.2f}% | {bar}")

    print("\n[Gender labels]")
    for label, count in Counter(s.get("gender_label") for s in samples).most_common():
        pct = 100.0 * count / max(len(samples), 1)
        print(f"{str(label):16s}: {count:5d} | {pct:6.2f}%")

    print("\n[Skin tone labels]")
    for label, count in Counter(s.get("skin_tone_label") for s in samples).most_common():
        pct = 100.0 * count / max(len(samples), 1)
        print(f"{str(label):24s}: {count:5d} | {pct:6.2f}%")


def audit_global_loader_distribution(loader, n_batches: int = 100) -> None:
    all_ages = []
    all_filenames = []
    all_gender_labels = []

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        all_ages.extend(batch["age"].detach().cpu().tolist())
        all_filenames.extend(batch.get("filename", []))
        all_gender_labels.extend(batch.get("gender_label", []))

    ages = torch.tensor(all_ages, dtype=torch.float32)
    print("\n========== GLOBAL LOADER AUDIT ==========")
    print("Inspected samples:", len(ages))
    print("Unique filenames:", len(set(all_filenames)) if all_filenames else "not available")
    print(
        "\n[Age stats inside loader]"
        f"\nmin/mean/median/max/std: "
        f"{ages.min().item():.1f} / "
        f"{ages.mean().item():.1f} / "
        f"{ages.median().item():.1f} / "
        f"{ages.max().item():.1f} / "
        f"{ages.std(unbiased=False).item():.1f}"
    )

    if all_gender_labels:
        print("\n[Gender labels inside loader]")
        for label, count in Counter(all_gender_labels).most_common():
            pct = 100.0 * count / max(len(all_gender_labels), 1)
            print(f"{str(label):16s}: {count:5d} | {pct:6.2f}%")
