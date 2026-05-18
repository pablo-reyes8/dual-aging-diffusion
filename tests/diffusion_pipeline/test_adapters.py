import torch

from src.diffusion_pipeline.DoRa import DoRALinear, inject_manual_dora_unet
from src.diffusion_pipeline.LoRa import LoRALinear, inject_manual_lora_unet


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(4, 4)
        self.to_k = torch.nn.Linear(4, 4)
        self.to_v = torch.nn.Linear(4, 4)
        self.to_out = torch.nn.Sequential(torch.nn.Linear(4, 4))

    def forward(self, x):
        return self.to_out(self.to_q(x) + self.to_k(x) + self.to_v(x))


def test_lora_linear_initially_matches_frozen_base_and_trains_only_adapters():
    base = torch.nn.Linear(5, 3)
    wrapped = LoRALinear(base, rank=2, alpha=2)
    x = torch.randn(4, 5)

    assert torch.allclose(wrapped(x), base(x), atol=1e-6)
    assert not any(p.requires_grad for p in wrapped.base_layer.parameters())
    assert wrapped.lora_down.weight.requires_grad
    assert wrapped.lora_up.weight.requires_grad
    assert wrapped(x).shape == (4, 3)


def test_dora_linear_initially_matches_base_shape_and_trainable_params():
    base = torch.nn.Linear(5, 3)
    wrapped = DoRALinear(base, rank=2, alpha=2)
    x = torch.randn(4, 5)

    assert wrapped(x).shape == base(x).shape
    assert wrapped.get_delta_weight().shape == base.weight.shape
    assert not any(p.requires_grad for p in wrapped.base_layer.parameters())
    assert wrapped.magnitude.requires_grad


def test_lora_and_dora_injection_replace_only_target_linear_modules():
    lora_model = inject_manual_lora_unet(TinyAttention(), rank=2, alpha=2, dropout=0.0, verbose=False)
    assert isinstance(lora_model.to_q, LoRALinear)
    assert isinstance(lora_model.to_out[0], LoRALinear)
    assert all(
        (not param.requires_grad) or ("lora_down" in name or "lora_up" in name)
        for name, param in lora_model.named_parameters()
    )

    dora_model = inject_manual_dora_unet(TinyAttention(), rank=2, alpha=2, dropout=0.0, verbose=False)
    assert isinstance(dora_model.to_q, DoRALinear)
    assert isinstance(dora_model.to_out[0], DoRALinear)
    assert all(
        (not param.requires_grad) or ("lora_down" in name or "lora_up" in name or "magnitude" in name)
        for name, param in dora_model.named_parameters()
    )
