import zipfile
from pathlib import Path

from PIL import Image

import data.paired_aging_dataset as paired_module
from data.paired_aging_dataset import (
    PairedAgingDataset,
    download_kaggle_paired_dataset,
    ensure_paired_aging_dataset,
    discover_aging_records,
    make_aging_pairs,
    split_records_by_identity,
)


def _image(path: Path, size=(180, 220)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 100, 80)).save(path)


def test_fgnet_pairs_are_forward_and_identity_disjoint(tmp_path):
    for identity in ("001", "002", "003"):
        for age in (10, 20, 35):
            _image(tmp_path / f"{identity}A{age:02d}.JPG")

    records = discover_aging_records(tmp_path, dataset="fgnet", min_image_side=128)
    train_records, val_records = split_records_by_identity(
        records, val_fraction=1 / 3, seed=7
    )
    assert {r.identity for r in train_records}.isdisjoint(
        {r.identity for r in val_records}
    )

    pairs = make_aging_pairs(
        train_records, min_age_gap=5, max_age_gap=30, max_pairs_per_identity=2
    )
    assert all(5 <= pair.age_gap <= 30 for pair in pairs)
    assert all(pair.source.identity == pair.target.identity for pair in pairs)

    sample = PairedAgingDataset(pairs, resolution=64)[0]
    assert sample["source_pixel_values"].shape == (3, 64, 64)
    assert sample["target_pixel_values"].shape == (3, 64, 64)
    assert sample["target_age"] > sample["source_age"]


def test_agedb_parser_filters_tiny_images(tmp_path):
    _image(tmp_path / "Jane Doe" / "1_JaneDoe_20_f.jpg", size=(64, 64))
    _image(tmp_path / "Jane Doe" / "2_JaneDoe_40_f.jpg")
    _image(tmp_path / "Jane Doe" / "3_JaneDoe_55_f.jpg")

    records = discover_aging_records(tmp_path, dataset="agedb", min_image_side=128)
    assert [record.age for record in records] == [40, 55]
    assert all(record.identity == "Jane Doe" for record in records)
    assert all(record.gender == "female" for record in records)


def test_kaggle_download_is_verified_and_reused(tmp_path, monkeypatch):
    source = tmp_path / "source.JPG"
    _image(source)
    calls = []

    def fake_urlretrieve(url, destination):
        calls.append(url)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.write(source, arcname="FGNET/images/001A10.JPG")
        return str(destination), None

    monkeypatch.setattr(paired_module.urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setitem(paired_module.EXPECTED_MIN_IMAGES, "fgnet", 1)

    first = download_kaggle_paired_dataset("fgnet", tmp_path / "cache")
    second = download_kaggle_paired_dataset("fgnet", tmp_path / "cache")

    assert first == second
    assert len(calls) == 1
    assert (first / paired_module.COMPLETE_MARKER).is_file()
    assert len(list(first.rglob("*.JPG"))) == 1


def test_custom_root_wins_without_downloading(tmp_path, monkeypatch):
    custom_root = tmp_path / "my_fgnet"
    _image(custom_root / "001A10.JPG")

    def fail_download(*args, **kwargs):
        raise AssertionError("A valid custom root must not trigger a download")

    monkeypatch.setattr(paired_module, "download_kaggle_paired_dataset", fail_download)
    resolved = ensure_paired_aging_dataset(
        "fgnet",
        root=custom_root,
        cache_dir=tmp_path / "cache",
    )
    assert resolved == custom_root
