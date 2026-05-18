from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from data.global_path_datasets import (
    GLOBAL_RESOLUTION,
    GlobalAgingFaceDataset,
    build_global_samples_from_attribute_csv,
    build_image_index,
    global_debug_collate_fn,
    global_train_collate_fn,
    make_global_prompt_from_attributes,
)


def _write_image(path: Path, size=(96, 96), color=(32, 96, 160)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _attribute_row(filename: str, age: float = 47.2):
    return {
        "filename": filename,
        "age_pred": age,
        "gender_pred": "male",
        "gender_label": "male",
        "gender_confidence": 0.91,
        "skin_tone_label": "medium skin tone",
        "skin_tone_confidence": 0.72,
        "hair_label": "gray hair",
        "hair_confidence": 0.82,
        "glasses_label": "without glasses",
        "glasses_confidence": 0.95,
        "face_attribute_phrase": "male, medium skin tone, gray hair",
        "enriched_prompt": "a portrait photo",
        "error": "",
    }


def test_build_global_samples_dataset_and_collates(tmp_path):
    image_dir = tmp_path / "images"
    csv_path = tmp_path / "attributes.csv"
    _write_image(image_dir / "00001.png")
    _write_image(image_dir / "00002.png", color=(160, 96, 32))
    pd.DataFrame([_attribute_row("00001.png"), _attribute_row("00002.png", age=71.6)]).to_csv(
        csv_path,
        index=False,
    )

    image_index = build_image_index(image_dir)
    samples = build_global_samples_from_attribute_csv(csv_path, image_index)
    assert len(samples) == 2
    assert samples[0]["image_path"].endswith("00001.png")
    assert "year-old" in samples[0]["prompt"]
    assert 0.0 <= samples[0]["age_norm"] <= 1.0

    dataset = GlobalAgingFaceDataset(
        samples=samples,
        resolution=32,
        train=False,
        normalize_for_diffusion=True,
    )
    item = dataset[0]
    assert item["pixel_values"].shape == (3, 32, 32)
    assert item["age"].shape == ()
    assert item["age_norm"].shape == ()
    assert item["filename"] == "00001.png"

    train_batch = global_train_collate_fn([dataset[0], dataset[1]])
    assert train_batch["pixel_values"].shape == (2, 3, 32, 32)
    assert train_batch["age"].shape == (2,)
    assert len(train_batch["prompt"]) == 2

    debug_batch = global_debug_collate_fn([dataset[0], dataset[1]])
    assert debug_batch["pixel_values"].shape == (2, 3, 32, 32)
    assert debug_batch["filename"] == ["00001.png", "00002.png"]
    assert len(debug_batch["image_path"]) == 2


def test_global_prompt_confidence_thresholds_are_conservative():
    high_conf = make_global_prompt_from_attributes(
        age=66.4,
        gender_label="female",
        gender_confidence=0.99,
        skin_tone_label="dark skin tone",
        skin_tone_confidence=0.9,
        hair_label="white hair",
        hair_confidence=0.9,
        glasses_label="wearing glasses",
        glasses_confidence=0.9,
    )
    assert "66-year-old woman" in high_conf
    assert "dark skin tone" in high_conf
    assert "wearing glasses" in high_conf

    low_conf = make_global_prompt_from_attributes(
        age=22,
        gender_label="female",
        gender_confidence=0.2,
        skin_tone_label="medium skin tone",
        skin_tone_confidence=0.1,
        hair_label="black hair",
        hair_confidence=0.1,
    )
    assert low_conf == "a portrait photo of a 22-year-old person"


def test_global_resolution_constant_is_training_compatible():
    assert GLOBAL_RESOLUTION == 512
