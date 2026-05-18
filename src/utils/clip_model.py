import torch
from typing import List, Dict, Union, Optional, Tuple
from transformers import CLIPTokenizer, CLIPTextModel


def load_clip_text_encoder(
    clip_model_id: str = "openai/clip-vit-large-patch14",
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    freeze: bool = True,
) -> Tuple[CLIPTokenizer, CLIPTextModel, str, torch.dtype]:
    """
    Loads a CLIP tokenizer and frozen CLIP text encoder in a modular way.

    Returns:
        tokenizer:
            CLIPTokenizer.
        text_encoder:
            CLIPTextModel loaded on device.
        device:
            Resolved device string.
        dtype:
            Resolved dtype.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = CLIPTokenizer.from_pretrained(clip_model_id)

    text_encoder = CLIPTextModel.from_pretrained(
        clip_model_id,
        torch_dtype=dtype,
    ).to(device)

    text_encoder.eval()

    if freeze:
        for p in text_encoder.parameters():
            p.requires_grad_(False)

    print("[OK] Loaded CLIP text encoder")
    print("Model id:", clip_model_id)
    print("Device:", device)
    print("Dtype:", dtype)
    print("Frozen:", freeze)

    return tokenizer, text_encoder, device, dtype


@torch.no_grad()
def encode_prompts_with_clip(
    prompts: Union[str, List[str]],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    max_length: int = 77,
    normalize_pooled: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Encodes one prompt or a list of prompts using a CLIP text encoder.

    Returns:
        input_ids:
            Tensor of shape [B, 77].
        attention_mask:
            Tensor of shape [B, 77].
        last_hidden_state:
            Tensor of shape [B, 77, hidden_dim].
            This is the sequence embedding used by Stable Diffusion cross-attention.
        pooled_output:
            Tensor of shape [B, hidden_dim].
            Global CLIP text embedding.
        pooled_output_norm:
            Tensor of shape [B, hidden_dim].
            Normalized pooled embedding if normalize_pooled=True.
    """

    if isinstance(prompts, str):
        prompts = [prompts]

    if device is None:
        device = next(text_encoder.parameters()).device

    if dtype is None:
        dtype = next(text_encoder.parameters()).dtype

    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True,
    )

    last_hidden_state = outputs.last_hidden_state.to(dtype)
    pooled_output = outputs.pooler_output.to(dtype)

    if normalize_pooled:
        pooled_output_norm = torch.nn.functional.normalize(
            pooled_output.float(),
            dim=-1,
        ).to(dtype)
    else:
        pooled_output_norm = pooled_output

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "last_hidden_state": last_hidden_state,
        "pooled_output": pooled_output,
        "pooled_output_norm": pooled_output_norm,
    }
