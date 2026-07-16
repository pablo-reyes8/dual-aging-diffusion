from pathlib import Path

import pytest
import torch

from src.training.chekpoints import (
    atomic_torch_save,
    get_adapter_state_dict,
    load_adapter_state_dict,
    torch_load_cpu,
)
from src.training.metrics import MetricsTracker, count_modes, tensor_fraction, tensor_mean_float
from src.training.mixed_precision import autocast_ctx, move_batch_to_device, resolve_device
from src.training.scheduler_warmup import (
    WarmupCosineLR,
    compute_warmup_steps,
    estimate_optimizer_steps,
)
from src.training.target_prompt_building import (
    build_global_prompt_pack_from_loader_prompts,
    build_local_prompt_pack_from_loader_prompts,
    extract_age_from_prompt,
    extract_score_from_local_prompt,
    remove_age_from_global_prompt,
    remove_score_from_local_prompt,
)
from src.training.training_loss_helpers import (
    next_cycling_batch,
    paired_supervision_enabled,
    should_run_paired_supervision,
)


def test_scheduler_warmup_cosine_state_and_lr_bounds():
    param = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    scheduler = WarmupCosineLR(optimizer, total_steps=10, warmup_steps=2, min_lr=1e-5)

    assert scheduler.get_lr() == pytest.approx(0.0)
    optimizer.step()
    scheduler.step()
    assert scheduler.get_lr() == pytest.approx(5e-4)
    for _ in range(20):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_lr() == pytest.approx(1e-5)

    state = scheduler.state_dict()
    restored = WarmupCosineLR(optimizer, total_steps=10, warmup_steps=2, min_lr=1e-5)
    restored.load_state_dict(state)
    assert restored.step_num == scheduler.step_num


def test_scheduler_step_estimation_and_warmup_validation():
    assert estimate_optimizer_steps(7, num_epochs=3, grad_accum_steps=4) == 6
    assert estimate_optimizer_steps(7, num_epochs=3, grad_accum_steps=4, drop_last_accum=True) == 5
    assert compute_warmup_steps(total_steps=100, warmup_ratio=0.1, min_warmup_steps=3) == 10
    assert compute_warmup_steps(total_steps=4, warmup_ratio=0.5, min_warmup_steps=10) < 4


def test_mixed_precision_batch_mover_preserves_metadata():
    device = resolve_device("cpu")
    batch = {
        "pixel_values": torch.ones(2, 3, 4, 4),
        "prompt": ["a", "b"],
        "nested": (torch.zeros(1), {"id": "sample"}),
    }
    moved = move_batch_to_device(batch, device)
    assert moved["pixel_values"].device.type == "cpu"
    assert moved["prompt"] == ["a", "b"]
    assert moved["nested"][0].device.type == "cpu"
    assert moved["nested"][1]["id"] == "sample"

    with autocast_ctx(device=device, enabled=True, amp_dtype="bf16"):
        out = moved["pixel_values"] + 1
    assert out.shape == (2, 3, 4, 4)


def test_prompt_builders_preserve_lengths_and_ranges():
    global_prompts = [
        "a portrait photo of a 42-year-old man, medium skin tone, gray hair",
        "a portrait photo of a 71-year-old woman, wearing glasses",
    ]
    assert extract_age_from_prompt(global_prompts[0]) == 42
    neutral = remove_age_from_global_prompt(global_prompts[0])
    assert "42-year-old" not in neutral
    assert "gray hair" not in neutral

    global_pack = build_global_prompt_pack_from_loader_prompts(
        global_prompts,
        torch.tensor([42.0, 71.0]),
        p_neutral=0.0,
    )
    assert len(global_pack["target_prompts"]) == 2
    assert global_pack["target_ages"].shape == (2,)
    assert torch.all((global_pack["target_ages"] >= 18) & (global_pack["target_ages"] <= 90))

    local_prompts = [
        "a tightly cropped, centered close-up of the forehead region, showing facial skin texture and local aging details, with an aging score of 70%, for a white person",
        "a tightly cropped, centered close-up of the under-eye region, showing facial skin texture and local aging details, with an aging score of 95%, for an Asian person",
    ]
    assert extract_score_from_local_prompt(local_prompts[0]) == 70
    assert "aging score" not in remove_score_from_local_prompt(local_prompts[0])

    local_pack = build_local_prompt_pack_from_loader_prompts(
        local_prompts,
        torch.tensor([0.70, 0.95]),
        p_neutral=0.0,
    )
    assert len(local_pack["target_prompts"]) == 2
    assert local_pack["target_scores"].shape == (2,)
    assert torch.all((local_pack["target_scores"] >= 0) & (local_pack["target_scores"] <= 1))


def test_metrics_tracker_and_basic_metric_helpers():
    assert tensor_mean_float(torch.tensor([1.0, 3.0])) == pytest.approx(2.0)
    assert tensor_fraction(torch.tensor([1.0, 2.0, 3.0]), lambda x: x > 1.5) == pytest.approx(2 / 3)
    assert count_modes(["aging", "aging", "anchor"]) == {
        "mode_frac/aging": pytest.approx(2 / 3),
        "mode_frac/anchor": pytest.approx(1 / 3),
    }

    tracker = MetricsTracker()
    tracker.update({"loss/total": torch.tensor(2.0)}, n=2)
    tracker.update({"loss/total": 4.0}, n=1)
    assert tracker.compute()["loss/total"] == pytest.approx(8 / 3)


def test_adapter_checkpoint_roundtrip_and_atomic_save(tmp_path):
    from src.diffusion_pipeline.LoRa import LoRALinear

    model = torch.nn.Sequential(LoRALinear(torch.nn.Linear(4, 3), rank=2, alpha=2))
    state = get_adapter_state_dict(model)
    assert state
    assert all("lora_" in key for key in state)

    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data.add_(1.0)

    result = load_adapter_state_dict(model, state, strict=True)
    assert result["n_loaded"] == len(state)

    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"adapter": state}, path)
    loaded = torch_load_cpu(path)
    assert loaded["adapter"].keys() == state.keys()


def test_optional_paired_supervision_schedule_and_loader_cycle():
    loader = [[{"value": 1}], [{"value": 2}]]
    loss_fn = lambda batch: {"loss": torch.ones(())}
    assert not paired_supervision_enabled(None, None, 0, 0.0)
    assert paired_supervision_enabled(loader, loss_fn, 4, 0.25)
    assert [should_run_paired_supervision(i, 4) for i in range(5)] == [
        False, False, False, True, False
    ]

    iterator = iter(loader)
    first, iterator = next_cycling_batch(loader, iterator)
    second, iterator = next_cycling_batch(loader, iterator)
    restarted, iterator = next_cycling_batch(loader, iterator)
    assert first == restarted
    assert first != second
