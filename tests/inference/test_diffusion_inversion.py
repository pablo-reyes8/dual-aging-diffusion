from types import SimpleNamespace

import pytest
import torch

from src.inference.diffusion_inversion import (
    DDIMInversionEditor,
    InversionConfig,
    InversionEditResult,
)
from src.inference.fusion_bundle_config import FusionModelConfig
from src.inference.fusion_refiner_helpers import apply_fusion_refiner_if_available


class TinyLatentDist:
    def __init__(self, mean):
        self.mean = mean


class TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=2.0)

    def encode(self, image):
        latent = torch.nn.functional.avg_pool2d(image, 8)
        latent = torch.cat([latent, torch.zeros_like(latent[:, :1])], dim=1)
        return SimpleNamespace(latent_dist=TinyLatentDist(latent))

    def decode(self, latent, return_dict=True):
        image = torch.nn.functional.interpolate(latent[:, :3], scale_factor=8, mode="nearest")
        return SimpleNamespace(sample=image)


class TinyTokenizer:
    model_max_length = 4

    def __call__(self, prompts, **kwargs):
        values = [[sum(map(ord, prompt)) % 17] * self.model_max_length for prompt in prompts]
        ids = torch.tensor(values, dtype=torch.long)
        return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))


class TinyTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2) / 100.0
        return SimpleNamespace(last_hidden_state=hidden)


class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, sample, timestep, encoder_hidden_states, return_dict=True):
        value = encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1)
        return SimpleNamespace(sample=torch.ones_like(sample) * value * self.adapter_weight)


class TinySchedulerBase:
    def __init__(self, config=None):
        self.config = config or SimpleNamespace(prediction_type="epsilon")
        self.timesteps = torch.empty(0, dtype=torch.long)

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def scale_model_input(self, sample, timestep):
        return sample


class TinyForwardScheduler(TinySchedulerBase):
    def set_timesteps(self, num_steps, device=None):
        self.timesteps = torch.arange(num_steps - 1, -1, -1, device=device)

    def step(self, model_output, timestep, sample, return_dict=True):
        return SimpleNamespace(prev_sample=sample - model_output * 0.01)


class TinyInverseScheduler(TinySchedulerBase):
    def set_timesteps(self, num_steps, device=None):
        self.timesteps = torch.arange(num_steps, device=device)

    def step(self, model_output, timestep, sample, return_dict=True):
        return SimpleNamespace(prev_sample=sample + model_output * 0.01)


def make_editor(*, reconstruction=True, cache=True):
    unet = TinyUNet()
    config = InversionConfig(
        enabled=True,
        num_steps=4,
        strength=0.5,
        inversion_guidance_scale=1.0,
        edit_guidance_scale=1.0,
        return_source_reconstruction=reconstruction,
        cache_enabled=cache,
    )
    bundle = {
        "vae": TinyVAE(),
        "unet": unet,
        "tokenizer": TinyTokenizer(),
        "text_encoder": TinyTextEncoder(),
        "scheduler_infer": TinyForwardScheduler(),
        "model_id": "tiny-local",
        "adapter_type": "dora",
    }
    editor = DDIMInversionEditor.from_bundle(
        bundle,
        device="cpu",
        config=config,
        scheduler_classes=(TinyForwardScheduler, TinyInverseScheduler),
    )
    return editor, bundle


def test_ddim_inversion_shapes_roundtrip_and_adapter_reuse():
    editor, bundle = make_editor()
    image = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    result = editor.edit(
        image,
        source_prompt="source skin score 20",
        target_prompt="target skin score 80",
        negative_prompt="artifact",
        return_inverted_latent=True,
    )

    assert editor.unet is bundle["unet"]
    assert result.inverted_latent.shape == (1, 4, 4, 4)
    assert result.image.shape == (1, 3, 32, 32)
    assert result.reconstruction.shape == result.image.shape
    assert result.image.min() >= -1 and result.image.max() <= 1
    assert result.diagnostics["adapter_type"] == "dora"
    assert result.diagnostics["effective_steps"] == 2
    assert result.diagnostics["reconstruction_metrics"]["mse"] >= 0


def test_ddim_inversion_is_deterministic_and_cacheable():
    editor, _ = make_editor()
    image = torch.rand(1, 3, 32, 32) * 2 - 1
    kwargs = dict(
        source_prompt="same source",
        target_prompt="same target",
        negative_prompt="bad",
        cache_key="subject/forehead",
    )
    first = editor.edit(image, **kwargs)
    second = editor.edit(image, **kwargs)

    assert torch.equal(first.image, second.image)
    assert first.diagnostics["cache_hit"] is False
    assert second.diagnostics["cache_hit"] is True


def test_target_equal_source_matches_source_reconstruction():
    editor, _ = make_editor()
    image = torch.rand(1, 3, 32, 32) * 2 - 1
    result = editor.edit(
        image,
        source_prompt="observed crop",
        target_prompt="observed crop",
    )
    assert torch.allclose(result.image, result.reconstruction, atol=1e-7, rtol=0)


def test_legacy_config_defaults_to_disabled_and_validates_method():
    config = InversionConfig.from_mapping(None)
    assert config.enabled is False
    assert config.method == "ddim"
    with pytest.raises(ValueError, match="supports only 'ddim'"):
        InversionConfig.from_mapping({"enabled": True, "method": "null_text"})


class TinyRefinerPipe:
    def __init__(self):
        self.unet = torch.nn.Linear(1, 1)
        self.vae = torch.nn.Linear(1, 1)
        self.text_encoder = torch.nn.Linear(1, 1)
        self.text_encoder_2 = torch.nn.Linear(1, 1)
        self.calls = 0

    def to(self, device):
        return self

    def __call__(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(images=[kwargs["image"]])


def test_refiner_inversion_failure_falls_back_to_historical_img2img(monkeypatch):
    import src.inference.diffusion_inversion as inversion_module

    def fail_inversion(**kwargs):
        raise RuntimeError("unsupported fake SDXL contract")

    monkeypatch.setattr(inversion_module, "edit_sdxl_refiner_with_ddim_inversion", fail_inversion)
    pipe = TinyRefinerPipe()
    cfg = FusionModelConfig(
        inversion=InversionConfig(enabled=True, fallback_to_img2img=True)
    )
    bundle = {"pipe": pipe, "config": cfg}
    source = torch.rand(1, 3, 16, 16)

    with pytest.warns(RuntimeWarning, match="falling back"):
        result = apply_fusion_refiner_if_available(source, bundle, device="cpu")

    assert pipe.calls == 1
    assert torch.allclose(result, source, atol=1 / 255 + 1e-6)
    assert bundle["last_inversion_diagnostics"]["status"] == "fallback_to_img2img"


def test_local_inversion_failure_falls_back_without_changing_fusion_contract(monkeypatch):
    import src.training.training_sampling_helpers as sampling_helpers

    _editor, bundle = make_editor()

    def fail_edit(self, *args, **kwargs):
        raise RuntimeError("inverse scheduler unavailable")

    def historical_stub(**kwargs):
        return kwargs["image"]

    monkeypatch.setattr(DDIMInversionEditor, "edit", fail_edit)
    monkeypatch.setattr(sampling_helpers, "_img2img_tensor_bundle_safe", historical_stub)
    crop = torch.rand(1, 3, 32, 32) * 2 - 1
    with pytest.warns(RuntimeWarning, match="falling back to img2img"):
        outputs = sampling_helpers.default_sample_local_forward(
            mixed_local_bundle=bundle,
            zones=[
                {
                    "zone_name": "forehead",
                    "crop": crop,
                    "prompt": "local skin with an aging score of 80%",
                    "source_score": 20,
                    "bbox": (0, 0, 32, 32),
                    "mask": None,
                }
            ],
            device=torch.device("cpu"),
            strength=0.2,
            guidance_scale=1.0,
            num_inference_steps=4,
            generation_method="ddim_inversion",
            inversion_config={
                "enabled": True,
                "num_steps": 4,
                "strength": 0.5,
                "fallback_to_img2img": True,
            },
        )

    assert outputs[0]["inversion_fallback"] is True
    assert outputs[0]["aged_crop"] is crop
    assert outputs[0]["bbox"] == (0, 0, 32, 32)
    assert outputs[0]["mask"] is None


def test_zone_fallback_never_uses_target_score_as_source_condition(monkeypatch):
    import src.training.training_sampling_helpers as sampling_helpers

    _editor, bundle = make_editor()
    captured = {}

    def capture_edit(self, image, **kwargs):
        captured.update(kwargs)
        return InversionEditResult(image=image, diagnostics=dict(kwargs["diagnostics"]))

    monkeypatch.setattr(DDIMInversionEditor, "edit", capture_edit)
    crop = torch.rand(1, 3, 32, 32) * 2 - 1
    outputs = sampling_helpers.default_sample_local_forward(
        mixed_local_bundle=bundle,
        zones=[
            {
                "zone_name": "forehead",
                "crop": crop,
                "prompt": "close-up skin, with a pronounced local aging score of 80%, for a person",
                "bbox": (0, 0, 32, 32),
            }
        ],
        device=torch.device("cpu"),
        strength=0.2,
        guidance_scale=1.0,
        num_inference_steps=4,
        generation_method="ddim_inversion",
        inversion_config={"enabled": True, "num_steps": 4, "strength": 0.5},
    )

    assert "80%" not in captured["source_prompt"]
    assert captured["diagnostics"]["source_score_origin"] == "zone_fallback"
    assert outputs[0]["inversion_fallback"] is False
