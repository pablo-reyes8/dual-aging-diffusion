import torch

from src.training.training_sampling_helpers import (
    parse_sampling_global_batch,
    parse_sampling_local_batch,
    run_deterministic_training_reconstruction_sample,
)


def test_sampling_batch_parsers_accept_flexible_keys():
    global_batch = {
        "pixel_values": torch.zeros(1, 3, 16, 16),
        "prompt": ["a portrait photo of a person"],
        "filename": ["sample.png"],
    }
    parsed_global = parse_sampling_global_batch(global_batch)
    assert parsed_global["x_orig"].shape == (1, 3, 16, 16)
    assert parsed_global["global_prompt"] == "a portrait photo of a person"
    assert parsed_global["sample_id"] == "sample.png"

    local_batch = {
        "pixel_values": torch.zeros(2, 3, 8, 8),
        "prompt": ["zone one", "zone two"],
        "bbox": [(0, 0, 8, 8), (8, 8, 16, 16)],
    }
    zones = parse_sampling_local_batch(local_batch)
    assert len(zones) == 2
    assert zones[0]["crop"].shape == (1, 3, 8, 8)
    assert zones[1]["bbox"] == (8, 8, 16, 16)


def test_deterministic_training_reconstruction_sample_uses_inference_fusion(tmp_path):
    global_batch = {
        "pixel_values": torch.zeros(1, 3, 32, 32),
        "prompt": ["a portrait photo of a 60-year-old person"],
        "filename": ["person_001.png"],
    }
    local_batch = {
        "zones": [
            {
                "crop": torch.zeros(1, 3, 8, 8),
                "prompt": "local forehead aging",
                "bbox": (8, 8, 16, 16),
                "mask": torch.ones(1, 1, 8, 8),
            }
        ]
    }

    def sample_global_forward_fn(**kwargs):
        return torch.ones(1, 3, 32, 32) * 0.25

    def sample_local_forward_fn(**kwargs):
        return [
            {
                "zone_name": "forehead",
                "aged_crop": torch.ones(1, 3, 8, 8) * 0.75,
                "bbox": (8, 8, 16, 16),
                "mask": torch.ones(1, 1, 8, 8),
                "prompt": "local forehead aging",
            }
        ]

    result = run_deterministic_training_reconstruction_sample(
        mixed_global_bundle={},
        mixed_local_bundle={},
        sampling_loader_global=global_batch,
        sampling_loader_local=local_batch,
        device=torch.device("cpu"),
        run_name="pytest",
        epoch=0,
        output_dir=tmp_path,
        sample_global_forward_fn=sample_global_forward_fn,
        sample_local_forward_fn=sample_local_forward_fn,
        save_grid=True,
        verbose=False,
    )

    assert result["sample_id"] == "person_001.png"
    assert result["fusion_out"]["mode"] == "deterministic"
    assert result["fusion_out"]["x_final"].shape == (1, 3, 32, 32)
    assert result["paths"]["x_final"].exists()
    assert result["paths"]["grid"].exists()
