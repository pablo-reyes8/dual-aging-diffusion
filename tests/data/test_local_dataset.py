import json
from pathlib import Path

import torch
from PIL import Image

from data.local_path_dataset import (
    LOCAL_RESOLUTION,
    LocalAgingCropDataset,
    build_image_index,
    build_local_samples,
    local_collate_fn,
    make_local_score_sampler,
    prepare_local_assets,
    split_samples_by_image_id,
)
from data.local_fused_dataset import SinglePersonSamplingDataset


def _write_image(path: Path, size=(96, 96), color=(128, 96, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _write_annotation(path: Path, image_id: str, score: float = 70.0) -> None:
    payload = {
        "image_id": image_id,
        "all_slots": [
            {
                "region_key": "surcos_nasogenianos",
                "region_name": "Surcos nasogenianos",
                "region_alias": "nasolabial folds",
                "label_id": "label-1",
                "box_index": 0,
                "bbox": {"x": 24, "y": 28, "w": 28, "h": 32},
                "score": score,
                "omitted": False,
                "ethnicity": "A portrait of a middle-aged white person with visible signs of aging.",
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prepare_local_assets_extracts_zip_fixtures(tmp_path):
    import zipfile

    image_zip = tmp_path / "subset_50.zip"
    json_zip = tmp_path / "annotations.zip"

    image_src = tmp_path / "src_img" / "00001.png"
    json_src = tmp_path / "src_json" / "00001_annotations.json"
    _write_image(image_src)
    _write_annotation(json_src, "00001.png")

    with zipfile.ZipFile(image_zip, "w") as zf:
        zf.write(image_src, "subset_50/00001.png")
    with zipfile.ZipFile(json_zip, "w") as zf:
        zf.write(json_src, "00001_annotations.json")

    json_dir, image_dir, image_index = prepare_local_assets(
        json_dir=tmp_path / "json_out",
        json_zip_path=json_zip,
        zip_path=image_zip,
        image_dir=tmp_path / "image_out",
    )

    assert json_dir.exists()
    assert image_dir.exists()
    assert "00001" in image_index
    assert list(json_dir.glob("*.json"))


def test_local_sample_build_dataset_item_collate_and_sampler(tmp_path):
    image_dir = tmp_path / "images"
    json_dir = tmp_path / "json"
    _write_image(image_dir / "00001.png")
    _write_image(image_dir / "00002.png", color=(64, 128, 96))
    _write_annotation(json_dir / "00001_annotations.json", "00001.png", score=82.0)
    _write_annotation(json_dir / "00002_annotations.json", "00002.png", score=15.0)

    image_index = build_image_index(image_dir)
    samples = build_local_samples(json_dir, image_index)
    assert len(samples) == 2
    assert {sample["image_stem"] for sample in samples} == {"00001", "00002"}
    assert all(Path(sample["image_path"]).exists() for sample in samples)

    train_samples, val_samples = split_samples_by_image_id(samples, val_fraction=0.5, seed=123)
    assert len(train_samples) == 1
    assert len(val_samples) == 1
    assert train_samples[0]["image_id"] != val_samples[0]["image_id"]

    dataset = LocalAgingCropDataset(
        samples=samples,
        resolution=32,
        context_scale=1.2,
        virtual_repeats=2,
        train=False,
        jitter_scores_enabled=False,
        normalize_for_diffusion=True,
    )
    assert len(dataset) == 4

    item = dataset[0]
    assert item["pixel_values"].shape == (3, 32, 32)
    assert item["pixel_values"].dtype == torch.float32
    assert item["score"].ndim == 0
    assert 0.0 <= item["score"].item() <= 1.0
    assert item["bbox_crop"].shape == (4,)
    assert "aging score" in item["prompt"]

    batch = local_collate_fn([dataset[0], dataset[1]])
    assert batch["pixel_values"].shape == (2, 3, 32, 32)
    assert batch["score"].shape == (2,)
    assert len(batch["prompt"]) == 2
    assert batch["bbox_crop"].shape == (2, 4)

    sampler = make_local_score_sampler(dataset, verbose=False)
    sampled_indices = list(iter(sampler))
    assert len(sampled_indices) == len(dataset)
    assert all(0 <= idx < len(dataset) for idx in sampled_indices)


def test_local_resolution_constant_is_training_compatible():
    assert LOCAL_RESOLUTION == 256


def test_single_person_sampling_dataset_returns_global_and_eight_local_zones(tmp_path):
    image_dir = tmp_path / "images"
    _write_image(image_dir / "09501.png", size=(128, 128))

    samples = []
    for idx, region_key in enumerate([
        "frente",
        "glabela_entrecejo",
        "patas_de_gallo",
        "bajo_ojo_ojeras",
        "surcos_nasogenianos",
        "labio_superior",
        "comisuras_lineas_marioneta",
        "puente_nasal",
    ]):
        samples.append({
            "image_id": "09501.png",
            "image_stem": "09501",
            "image_path": str(image_dir / "09501.png"),
            "region_key": region_key,
            "bbox": {"x": 32, "y": 32, "w": 32, "h": 32},
            "score_raw": 20.0,
            "ethnicity_raw": "A portrait of a white person.",
        })

    dataset = SinglePersonSamplingDataset(
        samples=samples,
        image_stem="09501",
        target_age=75,
        local_target_score=85.0,
        full_resolution=64,
        local_resolution=32,
        global_csv_path=tmp_path / "missing.csv",
    )
    item = dataset[0]

    assert item["x_orig"].shape == (3, 64, 64)
    assert "75-year-old" in item["global_prompt"]
    assert len(item["zones"]) == 8
    assert item["zones"][0]["crop"].shape == (3, 32, 32)
    assert item["zones"][0]["mask"].shape == (1, 32, 32)
    assert item["zones"][0]["mask"].min().item() == 0.0
    assert item["zones"][0]["mask"].max().item() == 1.0


def test_single_person_sampling_dataset_accepts_region_score_overrides(tmp_path):
    image_dir = tmp_path / "images"
    _write_image(image_dir / "09501.png", size=(128, 128))
    samples = [
        {
            "image_id": "09501.png",
            "image_stem": "09501",
            "image_path": str(image_dir / "09501.png"),
            "region_key": "frente",
            "bbox": {"x": 32, "y": 32, "w": 32, "h": 32},
            "score_raw": 20.0,
            "ethnicity_raw": "A portrait of a white person.",
        },
        {
            "image_id": "09501.png",
            "image_stem": "09501",
            "image_path": str(image_dir / "09501.png"),
            "region_key": "surcos_nasogenianos",
            "bbox": {"x": 32, "y": 32, "w": 32, "h": 32},
            "score_raw": 20.0,
            "ethnicity_raw": "A portrait of a white person.",
        },
    ]

    dataset = SinglePersonSamplingDataset(
        samples=samples,
        image_stem="09501",
        local_target_score={"frente": 85.0, "surcos_nasogenianos": 100.0},
        full_resolution=64,
        local_resolution=32,
        global_csv_path=tmp_path / "missing.csv",
    )
    prompts = {zone["zone_name"]: zone["prompt"] for zone in dataset[0]["zones"]}

    assert "85%" in prompts["frente"]
    assert "100%" in prompts["surcos_nasogenianos"]
