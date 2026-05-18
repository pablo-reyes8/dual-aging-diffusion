from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import yaml
from PIL import Image

from data.local_path_dataset import build_image_index, build_local_samples, prepare_local_assets


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root() / path


def load_dataset_version(config_path: str | Path, version: str) -> Dict[str, Any]:
    with open(resolve_path(config_path), "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    versions = payload.get("dataset_versions", {})
    if version not in versions:
        raise KeyError(f"Unknown dataset version '{version}'. Available: {sorted(versions)}")
    return versions[version]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root().resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def grayscale_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32)


def laplacian_variance(gray: np.ndarray) -> float:
    padded = np.pad(gray, 1, mode="edge")
    lap = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(np.var(lap))


def box_blur_3x3(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="reflect")
    acc = np.zeros_like(gray, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy:dy + gray.shape[0], dx:dx + gray.shape[1]]
    return acc / 9.0


def noise_score(gray: np.ndarray) -> float:
    residual = gray - box_blur_3x3(gray)
    return float(np.std(residual))


def bbox_is_invalid(bbox: Dict[str, Any], width: int, height: int, min_area: float) -> bool:
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["w"])
        h = float(bbox["h"])
    except Exception:
        return True

    if w <= 0 or h <= 0:
        return True
    if w * h < float(min_area):
        return True
    if x < 0 or y < 0:
        return True
    if x + w > width or y + h > height:
        return True
    return False


def records_by_image(samples: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["image_stem"])].append(sample)
    return grouped


def quality_flags(record: Dict[str, Any], thresholds: Dict[str, Any]) -> List[str]:
    flags = []

    if record["width"] < int(thresholds["min_width"]) or record["height"] < int(thresholds["min_height"]):
        flags.append("low_resolution")
    if record["blur_variance"] < float(thresholds["blur_variance_min"]):
        flags.append("blur")
    if record["mean_luma"] < float(thresholds["mean_luma_min"]):
        flags.append("underexposure")
    if record["mean_luma"] > float(thresholds["mean_luma_max"]):
        flags.append("overexposure")
    if record["dark_pixel_fraction"] > float(thresholds["dark_pixel_fraction_max"]):
        flags.append("excessive_dark_pixels")
    if record["bright_pixel_fraction"] > float(thresholds["bright_pixel_fraction_max"]):
        flags.append("excessive_bright_pixels")
    if record["noise_score"] > float(thresholds["noise_score_max"]):
        flags.append("excessive_noise")
    if record["bytes_per_pixel"] < float(thresholds["bytes_per_pixel_min"]):
        flags.append("strong_compression_proxy")
    if record["invalid_crop_count"] > 0:
        flags.append("invalid_crop")
    if record["annotation_count"] == 0:
        flags.append("missing_annotation")

    return flags


def audit_image(
    image_id: str,
    image_path: Path,
    annotations: List[Dict[str, Any]],
    thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
        gray = grayscale_array(image)
        fmt = image.format
        mode = image.mode

    file_size = image_path.stat().st_size
    invalid_crop_count = sum(
        bbox_is_invalid(sample["bbox"], width, height, float(thresholds["crop_min_area"]))
        for sample in annotations
    )

    record = {
        "image_id": image_id,
        "path": display_path(image_path),
        "sha256": sha256_file(image_path),
        "width": int(width),
        "height": int(height),
        "format": fmt,
        "mode": mode,
        "file_size_bytes": int(file_size),
        "blur_variance": round(laplacian_variance(gray), 6),
        "mean_luma": round(float(np.mean(gray)), 6),
        "dark_pixel_fraction": round(float(np.mean(gray <= 20.0)), 6),
        "bright_pixel_fraction": round(float(np.mean(gray >= 235.0)), 6),
        "noise_score": round(noise_score(gray), 6),
        "bytes_per_pixel": round(float(file_size) / max(float(width * height), 1.0), 6),
        "annotation_count": len(annotations),
        "invalid_crop_count": int(invalid_crop_count),
        "regions": sorted({str(sample["region_key"]) for sample in annotations}),
    }
    record["flags"] = quality_flags(record, thresholds)
    return record


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    flag_counts = Counter(flag for record in records for flag in record["flags"])
    region_counts = Counter(region for record in records for region in record["regions"])

    def mean_value(key: str) -> float:
        if not records:
            return 0.0
        return round(float(np.mean([float(r[key]) for r in records])), 6)

    return {
        "image_count": len(records),
        "total_annotations": int(sum(int(r["annotation_count"]) for r in records)),
        "images_with_flags": int(sum(1 for r in records if r["flags"])),
        "flag_counts": dict(sorted(flag_counts.items())),
        "region_counts": dict(sorted(region_counts.items())),
        "mean_width": mean_value("width"),
        "mean_height": mean_value("height"),
        "mean_blur_variance": mean_value("blur_variance"),
        "mean_luma": mean_value("mean_luma"),
        "mean_noise_score": mean_value("noise_score"),
        "mean_bytes_per_pixel": mean_value("bytes_per_pixel"),
    }


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "path",
        "sha256",
        "width",
        "height",
        "format",
        "mode",
        "file_size_bytes",
        "blur_variance",
        "mean_luma",
        "dark_pixel_fraction",
        "bright_pixel_fraction",
        "noise_score",
        "bytes_per_pixel",
        "annotation_count",
        "invalid_crop_count",
        "regions",
        "flags",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["regions"] = "|".join(record["regions"])
            row["flags"] = "|".join(record["flags"])
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_local_quality_manifest(config_path: str | Path, version: str, output_dir: str | Path) -> Dict[str, Any]:
    cfg = load_dataset_version(config_path, version)
    thresholds = cfg["thresholds"]

    json_dir, image_dir, image_index = prepare_local_assets(
        json_dir=resolve_path(cfg["annotation_dir"]),
        json_zip_path=resolve_path(cfg["annotation_zip"]),
        zip_path=resolve_path(cfg["image_zip"]),
        image_dir=resolve_path(cfg["image_dir"]),
    )
    image_index = build_image_index(image_dir)
    samples = build_local_samples(json_dir, image_index)
    grouped = records_by_image(samples)

    records = []
    for image_id, image_path in sorted(image_index.items()):
        records.append(
            audit_image(
                image_id=image_id,
                image_path=image_path,
                annotations=grouped.get(image_id, []),
                thresholds=thresholds,
            )
        )

    created_at = datetime.now(timezone.utc).isoformat()
    summary = summarize_records(records)
    payload = {
        "dataset_version": version,
        "created_at_utc": created_at,
        "source": cfg.get("source"),
        "config": cfg,
        "summary": summary,
        "records": records,
    }

    out_dir = resolve_path(output_dir)
    csv_path = out_dir / f"{version}_quality_manifest.csv"
    json_path = out_dir / f"{version}_quality_manifest.json"
    snapshot_path = out_dir / f"{version}_snapshot.json"

    write_csv(csv_path, records)
    write_json(json_path, payload)
    write_json(
        snapshot_path,
        {
            "dataset_version": version,
            "created_at_utc": created_at,
            "source": cfg.get("source"),
            "image_zip": cfg.get("image_zip"),
            "annotation_zip": cfg.get("annotation_zip"),
            "image_count": summary["image_count"],
            "total_annotations": summary["total_annotations"],
            "flag_counts": summary["flag_counts"],
            "manifest_csv": display_path(csv_path),
            "manifest_json": display_path(json_path),
        },
    )

    print("[OK] Wrote:", csv_path)
    print("[OK] Wrote:", json_path)
    print("[OK] Wrote:", snapshot_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic image quality manifests.")
    parser.add_argument("--dataset-version", default="data/configs/dataset_versions.yaml")
    parser.add_argument("--version", default="local_subset_v1")
    parser.add_argument("--output-dir", default="data/manifests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_local_quality_manifest(
        config_path=args.dataset_version,
        version=args.version,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
