# ============================================================
# GLOBAL LOSS AUXILIARY BUNDLE
#
# Frozen auxiliary models for GlobalAgingLoss:
#   1. Age estimator A(x)
#   2. Identity encoder F(x)
#   3. Optional perceptual model P(x_src, x_gen)
#
# Expected image input:
#   Tensor [B, 3, H, W] in [-1, 1]
#
# Important:
#   - Models are frozen.
#   - For generated images, DO NOT use torch.no_grad(),
#     because gradients must flow to generated image.
#   - For source images, use no_grad to save memory.
#   - This bundle does NOT decide which losses are active.
#     That belongs to GlobalAgingLoss via loss_mode.
# ============================================================

import re
import gc
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic helpers
# ============================================================

def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def image_minus1_1_to_0_1(x: torch.Tensor) -> torch.Tensor:
    """
    Converts images from [-1, 1] to [0, 1].
    """
    return (x.float() * 0.5 + 0.5).clamp(0.0, 1.0)


def normalize_with_mean_std(
    x_01: torch.Tensor,
    mean: List[float],
    std: List[float],
) -> torch.Tensor:
    """
    x_01: [B, 3, H, W] in [0, 1]
    """
    device = x_01.device
    dtype = x_01.dtype

    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)

    return (x_01 - mean_t) / std_t


def parse_age_value_from_label(label: str, fallback: float) -> float:
    """
    Converts labels like:
        "0-2"          -> 1.0
        "3-9"          -> 6.0
        "10-19"        -> 14.5
        "60-69"        -> 64.5
        "more than 70" -> 70.0
        "45"           -> 45.0

    If parsing fails, returns fallback.
    """
    if label is None:
        return float(fallback)

    text = str(label).lower().strip()

    nums = re.findall(r"\d+\.?\d*", text)
    nums = [float(n) for n in nums]

    if len(nums) >= 2:
        return float(sum(nums[:2]) / 2.0)

    if len(nums) == 1:
        return float(nums[0])

    return float(fallback)


def build_age_values_from_id2label(id2label: Dict[Any, Any]) -> torch.Tensor:
    """
    Builds a tensor of age values from a classification head id2label.

    Example:
        {0: "0-2", 1: "3-9", 2: "10-19"}
        -> tensor([1.0, 6.0, 14.5])
    """
    keys = sorted([int(k) for k in id2label.keys()])
    values = []

    for idx in keys:
        label = id2label[idx]
        values.append(parse_age_value_from_label(label, fallback=float(idx)))

    return torch.tensor(values, dtype=torch.float32)


def cleanup_cuda_cache_only():
    """
    Safe cleanup. Does not delete variables.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ============================================================
# HF Age Estimator
# ============================================================

class HFAgeEstimator(nn.Module):
    """
    Hugging Face image-classification age estimator.

    Converts classifier logits into continuous apparent age:

        age_pred = sum_k softmax(logits)_k * age_value_k

    This works for age-bin classifiers.
    """

    def __init__(
        self,
        model_id: str = "nateraw/vit-age-classifier",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        image_size: int = 224,
    ):
        super().__init__()

        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = dtype
        self.image_size = int(image_size)

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id)

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        freeze_module(self.model)

        id2label = self.model.config.id2label
        age_values = build_age_values_from_id2label(id2label)

        self.register_buffer(
            "age_values",
            age_values.to(device=self.device, dtype=torch.float32),
            persistent=False,
        )

        self.image_mean = getattr(
            self.processor,
            "image_mean",
            [0.485, 0.456, 0.406],
        )

        self.image_std = getattr(
            self.processor,
            "image_std",
            [0.229, 0.224, 0.225],
        )

        print("[OK] Loaded HFAgeEstimator")
        print("model_id:", model_id)
        print("num age classes:", len(self.age_values))
        print("age values:", self.age_values.detach().cpu().tolist())

    def move_to(
        self,
        device: str,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Safely move model and internal device/dtype fields.
        """
        self.device = torch.device(device)

        if dtype is not None:
            self.dtype = dtype

        self.model = self.model.to(device=self.device, dtype=self.dtype)

        if hasattr(self, "age_values"):
            self.age_values = self.age_values.to(
                device=self.device,
                dtype=torch.float32,
            )

        self.model.eval()
        return self

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] in [-1, 1]

        returns:
            normalized [B, 3, image_size, image_size]
        """
        x_01 = image_minus1_1_to_0_1(x)

        x_resized = F.interpolate(
            x_01,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        x_norm = normalize_with_mean_std(
            x_resized,
            mean=self.image_mean,
            std=self.image_std,
        )

        return x_norm.to(device=self.device, dtype=self.dtype)

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ):
        """
        x: [B, 3, H, W] in [-1, 1]

        Returns:
            age_pred: [B] in years
        """
        x_in = self.preprocess(x)

        outputs = self.model(
            pixel_values=x_in,
            return_dict=True,
        )

        logits = outputs.logits.float()
        probs = torch.softmax(logits, dim=-1)

        age_pred = probs @ self.age_values.float()

        if return_logits:
            return age_pred, logits

        return age_pred


# ============================================================
# FaceNet Identity Encoder
# ============================================================

class FaceNetIdentityEncoder(nn.Module):
    """
    Identity encoder using facenet-pytorch InceptionResnetV1.

    Assumes FFHQ-style aligned/centered faces.
    For first version, we do not run detection/alignment here.
    """

    def __init__(
        self,
        pretrained: str = "vggface2",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        image_size: int = 160,
    ):
        super().__init__()

        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError as e:
            raise ImportError(
                "facenet-pytorch is required for FaceNetIdentityEncoder.\n"
                "Install after PyTorch with:\n"
                "!pip -q install --no-deps facenet-pytorch\n"
                "The --no-deps flag avoids reinstalling torch/torchvision."
            ) from e

        self.device = torch.device(device)
        self.dtype = dtype
        self.image_size = int(image_size)

        self.model = InceptionResnetV1(pretrained=pretrained).eval()
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        freeze_module(self.model)

        print("[OK] Loaded FaceNetIdentityEncoder")
        print("pretrained:", pretrained)

    def move_to(
        self,
        device: str,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Safely move model and internal device/dtype fields.
        """
        self.device = torch.device(device)

        if dtype is not None:
            self.dtype = dtype

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        return self

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        FaceNet commonly receives images in [-1, 1].
        We keep that range and resize to 160x160.
        """
        x = x.float().clamp(-1.0, 1.0)

        x_resized = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        return x_resized.to(device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] in [-1, 1]

        Returns:
            normalized identity embedding [B, D]
        """
        x_in = self.preprocess(x)
        emb = self.model(x_in)
        emb = F.normalize(emb.float(), dim=-1)

        return emb


# ============================================================
# Optional LPIPS perceptual model
# ============================================================

class LPIPSPerceptual(nn.Module):
    """
    LPIPS perceptual distance.

    Input expected:
        x_src, x_gen in [-1, 1]

    Output:
        distance [B]
    """

    def __init__(
        self,
        net: str = "alex",
        device: str = "cuda",
    ):
        super().__init__()

        try:
            import lpips
        except ImportError as e:
            raise ImportError(
                "lpips is required for LPIPSPerceptual.\n"
                "Install in Colab with:\n"
                "!pip -q install lpips"
            ) from e

        self.device = torch.device(device)

        self.model = lpips.LPIPS(net=net).to(self.device)
        freeze_module(self.model)

        print("[OK] Loaded LPIPSPerceptual")
        print("net:", net)

    def move_to(self, device: str):
        """
        Safely move model and internal device field.
        """
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        return self

    def forward(
        self,
        x_src: torch.Tensor,
        x_gen: torch.Tensor,
    ) -> torch.Tensor:
        x_src = x_src.to(self.device).float().clamp(-1.0, 1.0)
        x_gen = x_gen.to(self.device).float().clamp(-1.0, 1.0)

        d = self.model(x_src, x_gen)
        return d.view(-1)


# ============================================================
# Global loss auxiliary bundle
# ============================================================

class GlobalLossAuxBundle(nn.Module):
    """
    Bundle of frozen auxiliary models used by GlobalAgingLoss.

    It does NOT compute final global loss.

    Exposes:
        - predict_age(...)
        - identity_embedding(...)
        - identity_similarity(...)
        - perceptual_distance(...)

    Important:
        Calling .to(device) alone is not enough because wrapper classes
        store self.device internally. Use .move_to(device).
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,

        use_age: bool = True,
        age_model_id: str = "nateraw/vit-age-classifier",
        age_image_size: int = 224,

        use_identity: bool = True,
        identity_pretrained: str = "vggface2",
        identity_image_size: int = 160,

        use_lpips: bool = False,
        lpips_net: str = "alex",
    ):
        super().__init__()

        self.device = torch.device(device)
        self.dtype = dtype

        self.use_age = bool(use_age)
        self.use_identity = bool(use_identity)
        self.use_lpips = bool(use_lpips)

        if self.use_age:
            self.age_estimator = HFAgeEstimator(
                model_id=age_model_id,
                device=device,
                dtype=dtype,
                image_size=age_image_size,
            )
        else:
            self.age_estimator = None

        if self.use_identity:
            self.identity_encoder = FaceNetIdentityEncoder(
                pretrained=identity_pretrained,
                device=device,
                dtype=dtype,
                image_size=identity_image_size,
            )
        else:
            self.identity_encoder = None

        if self.use_lpips:
            self.perceptual_model = LPIPSPerceptual(
                net=lpips_net,
                device=device,
            )
        else:
            self.perceptual_model = None

        self.eval()

        print("\n========== GLOBAL LOSS AUX BUNDLE ==========")
        print("use_age:", self.use_age)
        print("use_identity:", self.use_identity)
        print("use_lpips:", self.use_lpips)
        print("device:", self.device)
        print("dtype:", self.dtype)

    # --------------------------------------------------------
    # Availability checks
    # --------------------------------------------------------

    def has_age(self) -> bool:
        return self.use_age and self.age_estimator is not None

    def has_identity(self) -> bool:
        return self.use_identity and self.identity_encoder is not None

    def has_lpips(self) -> bool:
        return self.use_lpips and self.perceptual_model is not None

    # --------------------------------------------------------
    # Safe device movement
    # --------------------------------------------------------

    def move_to(
        self,
        device: str,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Safely moves all auxiliary models and updates internal device fields.

        Use this instead of .to(device).
        """
        self.device = torch.device(device)

        if dtype is not None:
            self.dtype = dtype

        if self.age_estimator is not None:
            self.age_estimator.move_to(
                device=device,
                dtype=self.dtype,
            )

        if self.identity_encoder is not None:
            self.identity_encoder.move_to(
                device=device,
                dtype=self.dtype,
            )

        if self.perceptual_model is not None:
            self.perceptual_model.move_to(device=device)

        self.eval()
        cleanup_cuda_cache_only()

        return self

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    def predict_age(
        self,
        images: torch.Tensor,
        grad_to_input: bool = True,
        return_logits: bool = False,
    ):
        """
        images: [B, 3, H, W] in [-1, 1]

        grad_to_input:
            True for generated images, so gradients flow to image.
            False for source images, to save memory.
        """
        if self.age_estimator is None:
            raise RuntimeError("Age estimator is disabled.")

        with torch.set_grad_enabled(grad_to_input):
            return self.age_estimator(
                images,
                return_logits=return_logits,
            )

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    def identity_embedding(
        self,
        images: torch.Tensor,
        grad_to_input: bool = True,
    ) -> torch.Tensor:
        """
        images: [B, 3, H, W] in [-1, 1]
        """
        if self.identity_encoder is None:
            raise RuntimeError("Identity encoder is disabled.")

        with torch.set_grad_enabled(grad_to_input):
            emb = self.identity_encoder(images)

        return emb

    def identity_similarity(
        self,
        source_images: torch.Tensor,
        generated_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns cosine similarity [B].

        Source embedding is computed without grad.
        Generated embedding keeps grad to image.
        """
        src_emb = self.identity_embedding(
            source_images,
            grad_to_input=False,
        )

        gen_emb = self.identity_embedding(
            generated_images,
            grad_to_input=True,
        )

        sim = (src_emb * gen_emb).sum(dim=-1)
        return sim

    # --------------------------------------------------------
    # Perceptual
    # --------------------------------------------------------

    def perceptual_distance(
        self,
        source_images: torch.Tensor,
        generated_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns perceptual distance [B].

        Requires use_lpips=True.
        """
        if self.perceptual_model is None:
            raise RuntimeError("Perceptual model is disabled.")

        source_detached = source_images.detach()

        with torch.set_grad_enabled(True):
            d = self.perceptual_model(
                source_detached,
                generated_images,
            )

        return d

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    @torch.no_grad()
    def audit_on_real_batch(
        self,
        images: torch.Tensor,
        max_items: int = 4,
    ) -> Dict[str, Any]:
        """
        Quick sanity audit on real images.
        No gradients.
        """
        images = images[:max_items].to(self.device).float()

        out = {}

        if self.has_age():
            age_pred = self.predict_age(
                images,
                grad_to_input=False,
            )
            out["age_pred"] = age_pred.detach().cpu()

        if self.has_identity():
            emb = self.identity_embedding(
                images,
                grad_to_input=False,
            )
            out["identity_embedding_shape"] = tuple(emb.shape)
            out["identity_embedding_norm"] = emb.norm(dim=-1).detach().cpu()

        if self.has_lpips():
            d = self.perceptual_distance(images, images)
            out["lpips_self"] = d.detach().cpu()

        return out
