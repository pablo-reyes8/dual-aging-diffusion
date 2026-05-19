# ============================================================
# LOCAL FACE AGING DATASET + DATALOADER
# Portable local implementation
# UPDATED LOCAL PROMPTS:
#   "a tightly cropped, centered close-up of the {region}..."
# ============================================================

import os
import re
import json
import math
import random
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler

# ============================================================
# Paths
# ============================================================

DATA_DIR = Path(__file__).resolve().parent

JSON_ZIP_PATH = DATA_DIR / "results_labeling" / "results_translated_json.zip"
JSON_DIR = DATA_DIR / "results_labeling" / "results_translated_json"

ZIP_PATH = DATA_DIR / "data_subset" / "subset_50.zip"
IMAGE_DIR = DATA_DIR / "data_subset" / "subset_50_extracted"

LOCAL_RESOLUTION = 256


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)


# ============================================================
# Extract images
# ============================================================

def extract_zip_if_needed(zip_path: Path, out_dir: Path) -> Path:
    """
    Extracts the image zip only if the output folder does not already
    contain images.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_images = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        existing_images.extend(out_dir.rglob(ext))

    if len(existing_images) > 0:
        print(f"[OK] Images already extracted: {len(existing_images)}")
        return out_dir

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    extracted_images = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        extracted_images.extend(out_dir.rglob(ext))

    print(f"[OK] Extracted images: {len(extracted_images)}")
    return out_dir


def extract_json_zip_if_needed(zip_path: Path, out_dir: Path) -> Path:
    """
    Extracts translated annotation JSON files only when out_dir does not
    already contain JSON files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_jsons = list(out_dir.glob("*.json"))
    if len(existing_jsons) > 0:
        print(f"[OK] JSON annotations already extracted: {len(existing_jsons)}")
        return out_dir

    if not zip_path.exists():
        raise FileNotFoundError(f"JSON ZIP file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    extracted_jsons = list(out_dir.glob("*.json"))
    print(f"[OK] Extracted JSON annotations: {len(extracted_jsons)}")
    return out_dir


def prepare_local_assets(
    json_dir: Path = JSON_DIR,
    json_zip_path: Path = JSON_ZIP_PATH,
    zip_path: Path = ZIP_PATH,
    image_dir: Path = IMAGE_DIR,
) -> Tuple[Path, Path, Dict[str, Path]]:
    """
    Prepares local annotation and image assets and returns:
        json_dir, image_dir, image_index

    This is explicit so importing this module does not extract files or fail
    when only helper functions/classes are needed.
    """
    json_dir = extract_json_zip_if_needed(json_zip_path, json_dir)
    image_dir = extract_zip_if_needed(zip_path, image_dir)
    image_index = build_image_index(image_dir)
    return json_dir, image_dir, image_index


# ============================================================
# Image index
# ============================================================

def build_image_index(image_root: Path) -> Dict[str, Path]:
    """
    Builds an index:
        stem -> image path

    Example:
        '00632' -> data/data_subset/subset_50_extracted/.../00632.png
    """
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        image_paths.extend(image_root.rglob(ext))

    index = {}
    duplicates = defaultdict(list)

    for path in image_paths:
        stem = path.stem
        duplicates[stem].append(path)
        if stem not in index:
            index[stem] = path

    duplicated_stems = {k: v for k, v in duplicates.items() if len(v) > 1}
    if duplicated_stems:
        print(f"[WARN] Duplicated image stems found: {len(duplicated_stems)}")
        first_key = next(iter(duplicated_stems))
        print(f"Example duplicate stem: {first_key} -> {duplicated_stems[first_key]}")

    print(f"[OK] Indexed unique images: {len(index)}")
    return index


IMAGE_INDEX = None


# ============================================================
# Region names
# ============================================================

REGION_TO_ENGLISH = {
    "frente": "forehead wrinkles",
    "glabela_entrecejo": "glabellar lines between the eyebrows",
    "patas_de_gallo": "crow's feet wrinkles",
    "bajo_ojo_ojeras": "under-eye wrinkles",
    "surcos_nasogenianos": "nasolabial folds",
    "labio_superior": "upper lip wrinkles",
    "comisuras_lineas_marioneta": "mouth corner marionette lines",
    "puente_nasal": "inter-ocular nasal bridge aging signs",
}

# More local/crop-aware descriptions for prompts.
REGION_TO_LOCAL_PROMPT_TEXT = {
    "frente": "forehead region",
    "glabela_entrecejo": "glabellar region between the eyebrows",
    "patas_de_gallo": "crow's feet region at the outer eye corner",
    "bajo_ojo_ojeras": "under-eye region",
    "surcos_nasogenianos": "nasolabial fold region",
    "labio_superior": "upper lip region",
    "comisuras_lineas_marioneta": "mouth corner and marionette line region",
    "puente_nasal": "nasal bridge and inter-ocular region",
}

REGION_ORDER = [
    "frente",
    "glabela_entrecejo",
    "patas_de_gallo",
    "bajo_ojo_ojeras",
    "surcos_nasogenianos",
    "labio_superior",
    "comisuras_lineas_marioneta",
    "puente_nasal",
]


REGION_SKIP_ALIASES = {
    # Keep skip behavior one-to-one with REGION_ORDER. These aliases only
    # resolve common wording to a single canonical zone; they never expand to
    # multiple anatomical regions.
    "labio": ["labio_superior"],
    "labios": ["labio_superior"],
    "upper_lip": ["labio_superior"],
    "labio_superior": ["labio_superior"],
}


def normalize_region_skip_list(skip_regions: Optional[List[str]] = None) -> List[str]:
    """
    Normalizes public skip-region names to canonical local region keys.

    Examples:
        skip_regions=["labio_superior"]
            -> ["labio_superior"]

        skip_regions=["labios"]
            -> ["labio_superior"]
    """
    if skip_regions is None:
        return []

    if isinstance(skip_regions, str):
        skip_regions = [skip_regions]

    normalized = []
    for raw_name in skip_regions:
        key = str(raw_name).strip().lower().replace(" ", "_").replace("-", "_")
        if not key:
            continue

        mapped_keys = REGION_SKIP_ALIASES.get(key, [key])
        for region_key in mapped_keys:
            if region_key not in normalized:
                normalized.append(region_key)

    unknown = [k for k in normalized if k not in REGION_TO_ENGLISH]
    if unknown:
        valid = sorted(REGION_TO_ENGLISH)
        raise ValueError(
            f"Unknown skip region(s): {unknown}. "
            f"Use canonical region keys or supported aliases. Valid region keys: {valid}"
        )

    return normalized


# ============================================================
# Prompt cleaning
# ============================================================

AGE_WORDS_TO_REMOVE = [
    "very young",
    "young adult",
    "young",
    "middle-aged",
    "middle aged",
    "older",
    "elderly",
    "very elderly",
    "clearly senior",
    "senior",
    "adult",
]

HAIR_WORDS_TO_REMOVE = [
    "with dark hair",
    "with fair skin",
    "with noticeable signs of aging",
    "with visible facial signs of aging",
    "with pronounced facial signs of aging",
    "with pronounced signs of aging",
    "with clear visible signs of aging",
    "with visible facial aging",
    "with visible signs of aging",
    "with strong visible signs of aging",
    "with pronounced facial aging",
    "with a brown complexion",
    "with dark skin",
]

GENDER_WORDS = {
    "man": "man",
    "woman": "woman",
    "girl": "person",
    "boy": "person",
}

ETHNICITY_RULES = [
    ("african american", "African American"),
    ("african", "African"),
    ("asian", "Asian"),
    ("indian", "Indian"),
    ("caucasian", "Caucasian"),
    ("european white", "white European"),
    ("white european", "white European"),
    ("white american", "white American"),
    ("american", "American"),
    ("european", "European"),
    ("white", "white"),
]


def strip_portrait_prefix(text: str) -> str:
    t = str(text).strip()

    prefixes = [
        "A portrait of an ",
        "A portrait of a ",
        "A portrait of ",
        "a portrait of an ",
        "a portrait of a ",
        "a portrait of ",
    ]

    for p in prefixes:
        if t.startswith(p):
            t = t[len(p):]

    t = t.strip()
    if t.endswith("."):
        t = t[:-1].strip()

    return t


def clean_local_demographic_context(text: Optional[str]) -> str:
    """
    Converts long full-face descriptions into a stable crop-level demographic context.

    Input examples:
        "A portrait of a young Asian woman with dark hair."
        "A portrait of an elderly white man with pronounced facial signs of aging."

    Output examples:
        "Asian person"
        "white person"
        "African American person"

    Why:
        Local crops often do not show hair, global age, or full gender cues.
        The local score should carry the aging signal, not words like elderly/young.
    """
    if text is None:
        return "person"

    t = strip_portrait_prefix(text)
    t_low = t.lower()

    # Remove visual/global attributes that are not reliable in local skin crops.
    for phrase in HAIR_WORDS_TO_REMOVE:
        t_low = t_low.replace(phrase, "")

    for phrase in AGE_WORDS_TO_REMOVE:
        t_low = t_low.replace(phrase, "")

    # Normalize spaces and punctuation.
    t_low = re.sub(r"[,\.]", " ", t_low)
    t_low = re.sub(r"\s+", " ", t_low).strip()

    ethnicity = None
    for key, value in ETHNICITY_RULES:
        if key in t_low:
            ethnicity = value
            break

    if ethnicity is None:
        return "person"

    # Keep gender out by default. Safer for local crops.
    return f"{ethnicity} person"


def add_article_for_context(demographic_context: str) -> str:
    """
    Converts:
        'Asian person' -> 'an Asian person'
        'white person' -> 'a white person'
    """
    context = str(demographic_context).strip()

    if len(context) == 0:
        return "a person"

    first = context[0].lower()
    article = "an" if first in ["a", "e", "i", "o", "u"] else "a"

    return f"{article} {context}"


def make_local_prompt(region_key: str, score: float, ethnicity_text: Optional[str]) -> str:
    """
    Main P_full-like prompt for local branch.

    Updated logic:
    - Explicitly tells Stable Diffusion this is a local crop.
    - Avoids "image of ..." because it can trigger full-image/full-face priors.
    - Uses "tightly cropped, centered close-up".
    - Keeps demographic context but removes global age/hair/portrait words.
    - Score controls local aging.
    """
    region_text = REGION_TO_LOCAL_PROMPT_TEXT.get(region_key, "facial skin region")
    demographic_context = clean_local_demographic_context(ethnicity_text)
    demographic_phrase = add_article_for_context(demographic_context)

    score_int = int(round(float(score)))
    score_int = max(0, min(100, score_int))

    prompt = (
        f"a tightly cropped, centered close-up of the {region_text}, "
        f"showing facial skin texture and local aging details, "
        f"with an aging score of {score_int}%, "
        f"for {demographic_phrase}"
    )

    return prompt


def make_zone_prompt(region_key: str) -> str:
    """
    P_zone-like prompt for zone-specific loss.

    Updated:
    - Also tells the model this is a local crop/close-up.
    """
    region_text = REGION_TO_LOCAL_PROMPT_TEXT.get(region_key, "facial skin region")

    return (
        f"a tightly cropped, centered close-up of the {region_text}, "
        f"showing facial skin texture"
    )


if "translated_ethnicities" in globals():
    print("[Prompt cleaning examples]")
    for x in translated_ethnicities[:8]:
        print("RAW:   ", x)
        print("CLEAN: ", clean_local_demographic_context(x))
        print()


# ============================================================
# Score jitter: edge-aware and high-aging biased
# ============================================================

def score_jitter_sigma(score: float) -> float:
    """
    Heteroscedastic annotation-noise model.

    Interpretation:
        The manual score is an imperfect observation of a latent local-aging
        severity. Middle scores are more uncertain. Very low and very high
        scores are semantically clearer.

    Important:
        This sigma is only for local noise around the observed score.
        Edge-biased sampling is handled separately in jitter_score().
    """
    s = float(score)

    if s <= 10:
        return 1.5
    elif s <= 25:
        return 3.0
    elif s <= 45:
        return 5.0
    elif s <= 60:
        return 8.0       # middle scores are more subjective
    elif s <= 75:
        return 7.0
    elif s <= 90:
        return 5.0
    else:
        return 3.0       # very high scores should stay high


def clamp_score(score: float) -> float:
    return float(max(0.0, min(100.0, score)))


def sample_near_low_edge(score: float) -> float:
    """
    Push low-aging samples closer to the low edge, without destroying
    the original annotation.
    """
    s = float(score)

    # Mostly stay near the original low score.
    sampled = random.gauss(mu=min(s, 10.0), sigma=3.0)

    return clamp_score(round(sampled))


def sample_near_high_edge(score: float) -> float:
    """
    Push high-aging samples closer to the high edge.

    This is the important part for your setting:
    the model must learn to say "large" when aging evidence is strong.
    """
    s = float(score)

    # For already high labels, bias toward the upper tail.
    # Example:
    #   score 70 -> center around 80
    #   score 85 -> center around 92
    #   score 95 -> center around 97
    edge_center = 100.0 - 0.45 * (100.0 - s)

    sampled = random.gauss(mu=edge_center, sigma=5.0)

    return clamp_score(round(sampled))


def sample_middle_uncertain(score: float) -> float:
    """
    Middle scores are annotation-uncertain.

    But we do NOT want to randomly throw them to 0 or 100.
    That would create label noise. Instead, we allow a wider interval
    around the original score.
    """
    s = float(score)

    sampled = random.gauss(mu=s, sigma=score_jitter_sigma(s))

    # Keep middle labels from becoming artificial extremes.
    if 35.0 <= s <= 65.0:
        sampled = max(20.0, min(80.0, sampled))

    return clamp_score(round(sampled))


def jitter_score(
    score: float,
    enabled: bool = True,
    high_edge_prob: float = 0.45,
    low_edge_prob: float = 0.20,
    local_prob: float = 0.55,
) -> float:
    """
    Edge-aware score jitter.

    Design:
        - Low scores can be sampled closer to 0, but mildly.
        - High scores are sampled more aggressively toward 100.
        - Middle scores receive wider uncertainty but are not randomly
          converted into extremes.
        - This is intentionally asymmetric because the downstream task
          is mainly aging, not rejuvenation.

    Args:
        score:
            Original manual local aging score in [0, 100].

        enabled:
            Whether to apply jitter.

        high_edge_prob:
            Probability of high-edge biased sampling for already-high scores.

        low_edge_prob:
            Probability of low-edge biased sampling for already-low scores.

        local_prob:
            Probability of ordinary local Gaussian jitter.
            Kept for readability; the residual probability is assigned
            to the relevant edge behavior.
    """
    score = float(score)

    if not enabled:
        return clamp_score(round(score))

    u = random.random()

    # Very low local aging evidence.
    if score <= 15.0:
        if u < low_edge_prob:
            return sample_near_low_edge(score)
        else:
            sampled = random.gauss(score, score_jitter_sigma(score))
            return clamp_score(round(sampled))

    # Low-to-mild aging.
    elif score <= 35.0:
        sampled = random.gauss(score, score_jitter_sigma(score))
        return clamp_score(round(sampled))

    # Ambiguous middle region.
    elif score <= 65.0:
        return sample_middle_uncertain(score)

    # High local aging evidence.
    elif score <= 85.0:
      if u < high_edge_prob:
          return sample_near_high_edge(score)
      else:
          sampled = random.gauss(score, score_jitter_sigma(score))

          # Important:
          # Once a score is already high, do not allow jitter to move it
          # back into a weak/mid-aging target.
          if score >= 75.0:
              sampled = max(70.0, sampled)
          elif score >= 65.0:
              sampled = max(60.0, sampled)

          return clamp_score(round(sampled))

    # Very high local aging evidence.
    else:
      if u < 0.70:
          return sample_near_high_edge(score)
      else:
          sampled = random.gauss(score, score_jitter_sigma(score))
          sampled = max(85.0, sampled)
          return clamp_score(round(sampled))


# ============================================================
# BBox utilities
# ============================================================

def clamp(v: int, low: int, high: int) -> int:
    return max(low, min(v, high))


def square_crop_from_center(
    cx: float,
    cy: float,
    side: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    Creates a square crop around center, shifting instead of padding
    when it goes outside the image.
    """
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = int(round(cx + side / 2.0))
    bottom = int(round(cy + side / 2.0))

    if left < 0:
        right += -left
        left = 0

    if top < 0:
        bottom += -top
        top = 0

    if right > image_width:
        shift = right - image_width
        left -= shift
        right = image_width

    if bottom > image_height:
        shift = bottom - image_height
        top -= shift
        bottom = image_height

    left = clamp(left, 0, image_width - 1)
    top = clamp(top, 0, image_height - 1)
    right = clamp(right, left + 1, image_width)
    bottom = clamp(bottom, top + 1, image_height)

    return left, top, right, bottom


def region_aware_bbox_to_square(
    bbox: Dict[str, int],
    region_key: str,
    image_width: int,
    image_height: int,
    context_scale: float = 1.20,
) -> Tuple[int, int, int, int]:
    """
    Converts each annotated bbox into a square crop.

    Why square?
        Stable Diffusion training expects fixed square image resolution.

    Why region-aware?
        Some zones should not be expanded the same way:
        - upper lip must avoid full mouth/teeth.
        - forehead is naturally wide.
        - marionette lines are vertical.
    """
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = float(bbox["w"])
    h = float(bbox["h"])

    cx = x + w / 2.0
    cy = y + h / 2.0

    if region_key == "frente":
        side = max(w * 1.02, h * 1.55)

    elif region_key == "glabela_entrecejo":
        side = max(w, h) * 1.35

    elif region_key == "patas_de_gallo":
        side = max(w, h) * 1.35

    elif region_key == "bajo_ojo_ojeras":
        side = max(w, h) * 1.40
        cy = y + 0.48 * h

    elif region_key == "surcos_nasogenianos":
        side = max(w, h) * 1.35

    elif region_key == "labio_superior":
        # Target is skin between nose and upper lip, not teeth/full mouth.
        cy = y + 0.32 * h
        side = max(w * 1.10, h * 0.50)

    elif region_key == "comisuras_lineas_marioneta":
        side = max(w * 1.35, h * 1.03)

    elif region_key == "puente_nasal":
        side = max(w, h) * 1.20

    else:
        side = max(w, h) * context_scale

    return square_crop_from_center(
        cx=cx,
        cy=cy,
        side=side,
        image_width=image_width,
        image_height=image_height,
    )


# ============================================================
# Load annotations and build local samples
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_image_stem_from_annotation(ann: Dict[str, Any]) -> str:
    """
    Robust extraction of image stem.
    """
    if "image_id" in ann and ann["image_id"] is not None:
        return Path(ann["image_id"]).stem

    if "image_meta" in ann and "stem" in ann["image_meta"]:
        return str(ann["image_meta"]["stem"])

    if "image_path" in ann and ann["image_path"] is not None:
        return Path(ann["image_path"]).stem

    raise ValueError("Could not infer image stem from annotation.")


def make_local_score_sampler(
    dataset,
    # Score-bin weights.
    w_00_10: float = 0.70,
    w_10_25: float = 0.85,
    w_25_40: float = 1.00,
    w_40_60: float = 0.90,
    w_60_75: float = 1.10,
    w_75_90: float = 3.50,
    w_90_100: float = 3.00,

    # Region-aware correction for scarcity of high scores.
    region_balance_strength: float = 0.35,

    # Anatomical prior.
    use_anatomical_prior: bool = True,

    # Safety clamp.
    min_weight: float = 0.50,
    max_weight: float = 5.50,

    verbose: bool = True,
):
    """
    Weighted sampler for LocalAgingCropDataset.

    Design:
        1. Oversample high-aging local crops.
        2. Mildly downweight overrepresented low/mid ranges.
        3. Mildly compensate regions with scarce high-aging examples.
        4. Add anatomical prior: visually important aging regions are shown more often.

    Important:
        This does not modify the raw dataset. It only modifies the probability
        of sampling each crop during training.
    """

    n_real = len(dataset.samples)

    # ------------------------------------------------------------
    # Anatomical prior
    # ------------------------------------------------------------
    # These are not labels. They are sampling priors.
    # Keep values moderate: too large can make the model region-biased.
    anatomical_region_weight = {
        # Strong visible aging zones.
        "bajo_ojo_ojeras": 1.25,
        "surcos_nasogenianos": 1.25,
        "comisuras_lineas_marioneta": 1.25,

        # Important, but currently under-sampled in high scores.
        "frente": 1.35,
        "patas_de_gallo": 1.20,

        # Medium.
        "glabela_entrecejo": 1.05,

        # Lower priority / delicate zones.
        "labio_superior": 0.85,
        "puente_nasal": 0.85,
    }

    # ------------------------------------------------------------
    # Region-level high-score scarcity correction
    # ------------------------------------------------------------
    region_counts = Counter()
    region_high_counts = Counter()

    for sample in dataset.samples:
        region = sample["region_key"]
        score = float(sample["score_raw"])

        region_counts[region] += 1

        if score >= 75:
            region_high_counts[region] += 1

    region_high_share = {}

    for region, n in region_counts.items():
        n_high = region_high_counts[region]
        region_high_share[region] = n_high / max(n, 1)

    total_n = sum(region_counts.values())
    total_high = sum(region_high_counts.values())
    global_high_share = total_high / max(total_n, 1)

    region_factor = {}

    for region, share in region_high_share.items():
        if share <= 0:
            factor = 1.0 + region_balance_strength
        else:
            relative_gap = max(
                0.0,
                (global_high_share - share) / max(global_high_share, 1e-8),
            )
            factor = 1.0 + region_balance_strength * relative_gap

        region_factor[region] = factor

    # ------------------------------------------------------------
    # Score-bin base weight
    # ------------------------------------------------------------
    def score_bin_weight(score):
        s = float(score)

        if s < 10:
            return w_00_10
        elif s < 25:
            return w_10_25
        elif s < 40:
            return w_25_40
        elif s < 60:
            return w_40_60
        elif s < 75:
            return w_60_75
        elif s < 90:
            return w_75_90
        else:
            return w_90_100

    # ------------------------------------------------------------
    # Build virtual-index weights
    # ------------------------------------------------------------
    weights = []

    for idx in range(len(dataset)):
        real_idx = idx % n_real
        sample = dataset.samples[real_idx]

        score = float(sample["score_raw"])
        region = sample["region_key"]

        w = score_bin_weight(score)

        # Scarcity correction mostly matters for medium/high aging.
        if score >= 60:
            w = w * region_factor.get(region, 1.0)

        # Anatomical prior applies to all samples, but is intentionally mild.
        if use_anatomical_prior:
            w = w * anatomical_region_weight.get(region, 1.0)

        w = max(min_weight, min(max_weight, w))
        weights.append(w)

    weights = torch.DoubleTensor(weights)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True)

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------
    if verbose:
        print("\n========== LOCAL SCORE SAMPLER ==========")
        print(f"Dataset virtual length: {len(dataset)}")
        print(f"Real samples:           {n_real}")
        print(f"Global high share >=75: {100.0 * global_high_share:.2f}%")
        print(f"Anatomical prior:       {use_anatomical_prior}")

        print("\n[Region high-score shares]")
        for region in sorted(region_counts.keys()):
            n = region_counts[region]
            n_high = region_high_counts[region]
            share = 100.0 * region_high_share[region]
            factor = region_factor[region]
            anat = anatomical_region_weight.get(region, 1.0)

            print(
                f"{region:32s} "
                f"n={n:4d} "
                f">=75={n_high:3d} "
                f"share={share:6.2f}% "
                f"scarcity_factor={factor:.3f} "
                f"anatomical_factor={anat:.3f}"
            )

        print("\n[Weight stats]")
        print(f"min weight:  {weights.min().item():.3f}")
        print(f"mean weight: {weights.mean().item():.3f}")
        print(f"max weight:  {weights.max().item():.3f}")

    return sampler


def build_local_samples(
    json_dir: Path,
    image_index: Dict[str, Path],
    keep_missing_labio: bool = False,
    require_existing_image: bool = True) -> List[Dict[str, Any]]:
    """
    Creates one sample per annotated region slot.

    Skips:
        - omitted slots
        - bbox = None
        - score = None
        - missing images
    """
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    json_paths = sorted(json_dir.glob("*.json"))
    if len(json_paths) == 0:
        raise FileNotFoundError(f"No JSON files found in: {json_dir}")

    samples = []
    missing_images = []

    for json_path in json_paths:
        ann = load_json(json_path)

        stem = get_image_stem_from_annotation(ann)
        image_path = image_index.get(stem)

        if image_path is None:
            missing_images.append(stem)
            if require_existing_image:
                continue
            else:
                raise FileNotFoundError(f"No image found for stem={stem}")

        image_id = ann.get("image_id", f"{stem}.png")
        all_slots = ann.get("all_slots", [])

        for slot in all_slots:
            region_key = slot.get("region_key")
            bbox = slot.get("bbox")
            score = slot.get("score")
            omitted = bool(slot.get("omitted", False))

            if omitted:
                continue

            if bbox is None or score is None:
                if region_key == "labio_superior" and keep_missing_labio:
                    pass
                else:
                    continue

            if region_key not in REGION_TO_ENGLISH:
                print(f"[WARN] Unknown region_key={region_key} in {json_path.name}")
                continue

            ethnicity_text = slot.get("ethnicity", None)
            demographic_context = clean_local_demographic_context(ethnicity_text)

            sample = {
                "image_id": image_id,
                "image_stem": stem,
                "image_path": str(image_path),
                "json_path": str(json_path),

                "region_key": region_key,
                "region_name": slot.get("region_name"),
                "region_alias": slot.get("region_alias"),
                "label_id": slot.get("label_id"),
                "box_index": slot.get("box_index"),

                "bbox": bbox,
                "score_raw": float(score),
                "score_norm": float(score) / 100.0,

                "ethnicity_raw": ethnicity_text,
                "demographic_context": demographic_context,

                "prompt": make_local_prompt(region_key, float(score), ethnicity_text),
                "zone_prompt": make_zone_prompt(region_key),
            }
            samples.append(sample)

    print(f"[OK] JSON files found: {len(json_paths)}")
    print(f"[OK] Local annotated crop samples: {len(samples)}")

    if missing_images:
        print(f"[WARN] Missing images for {len(missing_images)} JSON files.")
        print("First missing stems:", missing_images[:20])

    return samples

# ============================================================
# Image augmentation
# ============================================================

class GentleLocalAugment:
    """
    PIL-based augmentation designed for local facial crops.

    Goals:
        - Make the same zone appear under slightly different local geometry.
        - Preserve the semantics of the annotated zone.
        - Avoid aggressive transformations that break reinsertion logic.

    We avoid horizontal flip by default because left/right facial geometry
    matters later for fusion, even if diffusion itself could learn from flips.
    """

    def __init__(
        self,
        resolution: int = 256,
        train: bool = True,
        enable_horizontal_flip: bool = False,
        normalize_for_diffusion: bool = True):

        self.resolution = resolution
        self.train = train
        self.enable_horizontal_flip = enable_horizontal_flip
        self.normalize_for_diffusion = normalize_for_diffusion

        if train:
            aug_list = [
                transforms.RandomResizedCrop(
                    size=resolution,
                    scale=(0.90, 1.00),
                    ratio=(0.96, 1.04),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=4,
                            translate=(0.025, 0.025),
                            scale=(0.97, 1.04),
                            shear=1.5,
                            interpolation=transforms.InterpolationMode.BICUBIC,
                            fill=128,
                        )
                    ],
                    p=0.45,
                ),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.045,
                            contrast=0.055,
                            saturation=0.025,
                            hue=0.005,
                        )
                    ],
                    p=0.35,
                ),
            ]

            if enable_horizontal_flip:
                aug_list.insert(1, transforms.RandomHorizontalFlip(p=0.35))

        else:
            aug_list = [
                transforms.Resize(
                    (resolution, resolution),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            ]

        aug_list.append(transforms.ToTensor())

        if normalize_for_diffusion:
            aug_list.append(
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            )

        self.transform = transforms.Compose(aug_list)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


# ============================================================
# Dataset
# ============================================================

class LocalAgingCropDataset(Dataset):
    """
    Dataset for local face aging diffusion.

    One item = one local crop.

    Returned fields:
        pixel_values: Tensor [3, resolution, resolution], normalized to [-1, 1]
        score: Tensor scalar in [0, 1], possibly jittered
        score_raw: Tensor scalar in [0, 100], possibly jittered
        score_original: original manual score
        prompt: P_full-like prompt with zone + score + demographic context
        zone_prompt: P_zone-like prompt
        metadata: image_id, region_key, bbox, etc.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        resolution: int = 256,
        context_scale: float = 1.20,
        virtual_repeats: int = 8,
        train: bool = True,
        jitter_scores_enabled: bool = True,
        enable_horizontal_flip: bool = False,
        normalize_for_diffusion: bool = True,
        drop_regions: Optional[List[str]] = None,
        skip: Optional[List[str]] = None,
        skip_regions: Optional[List[str]] = None):

        self.resolution = int(resolution)
        self.context_scale = float(context_scale)
        self.virtual_repeats = max(1, int(virtual_repeats))
        self.train = bool(train)
        self.jitter_scores_enabled = bool(jitter_scores_enabled)

        requested_skip_regions = skip_regions if skip_regions is not None else skip
        drop_regions = set(
            normalize_region_skip_list(
                requested_skip_regions
                if requested_skip_regions is not None
                else drop_regions
            )
        )
        self.samples = [s for s in samples if s["region_key"] not in drop_regions]

        if len(self.samples) == 0:
            raise ValueError("Dataset has zero samples after filtering.")

        self.transform = GentleLocalAugment(
            resolution=self.resolution,
            train=self.train,
            enable_horizontal_flip=enable_horizontal_flip,
            normalize_for_diffusion=normalize_for_diffusion)

    def __len__(self):
        return len(self.samples) * self.virtual_repeats

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = idx % len(self.samples)
        sample = self.samples[real_idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        W, H = image.size

        left, top, right, bottom = region_aware_bbox_to_square(
            bbox=sample["bbox"],
            region_key=sample["region_key"],
            image_width=W,
            image_height=H,
            context_scale=self.context_scale)

        crop = image.crop((left, top, right, bottom))

        score_original = float(sample["score_raw"])
        score_aug = jitter_score(
            score_original,
            enabled=(self.train and self.jitter_scores_enabled))

        prompt = make_local_prompt(
            region_key=sample["region_key"],
            score=score_aug,
            ethnicity_text=sample.get("ethnicity_raw", None))

        zone_prompt = make_zone_prompt(sample["region_key"])

        pixel_values = self.transform(crop)

        return {
            "pixel_values": pixel_values,

            "score": torch.tensor(score_aug / 100.0, dtype=torch.float32),
            "score_raw": torch.tensor(score_aug, dtype=torch.float32),
            "score_original": torch.tensor(score_original, dtype=torch.float32),

            "prompt": prompt,
            "zone_prompt": zone_prompt,

            "region_key": sample["region_key"],
            "region_name": sample["region_name"],
            "region_alias": sample["region_alias"],
            "image_id": sample["image_id"],
            "image_stem": sample["image_stem"],
            "box_index": sample["box_index"],
            "demographic_context": sample["demographic_context"],

            "bbox_original": sample["bbox"],
            "bbox_crop": torch.tensor([left, top, right, bottom], dtype=torch.long),
            "json_path": sample["json_path"]}


def local_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),

        "score": torch.stack([b["score"] for b in batch], dim=0),
        "score_raw": torch.stack([b["score_raw"] for b in batch], dim=0),
        "score_original": torch.stack([b["score_original"] for b in batch], dim=0),

        "prompt": [b["prompt"] for b in batch],
        "zone_prompt": [b["zone_prompt"] for b in batch],

        "region_key": [b["region_key"] for b in batch],
        "region_name": [b["region_name"] for b in batch],
        "region_alias": [b["region_alias"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
        "image_stem": [b["image_stem"] for b in batch],
        "box_index": [b["box_index"] for b in batch],
        "demographic_context": [b["demographic_context"] for b in batch],

        "bbox_original": [b["bbox_original"] for b in batch],
        "bbox_crop": torch.stack([b["bbox_crop"] for b in batch], dim=0),
        "json_path": [b["json_path"] for b in batch]}



def split_samples_by_image_id(
    samples: List[Dict[str, Any]],
    val_fraction: float = 0.15,
    seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits by image_id, not by crop.

    This prevents leakage:
        same person/image in train and validation.
    """
    rng = random.Random(seed)

    image_ids = sorted(list(set(s["image_id"] for s in samples)))
    rng.shuffle(image_ids)

    n_val = max(1, int(round(len(image_ids) * val_fraction)))
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])

    train_samples = [s for s in samples if s["image_id"] in train_ids]
    val_samples = [s for s in samples if s["image_id"] in val_ids]

    print(f"[OK] Unique images: {len(image_ids)}")
    print(f"[OK] Train images: {len(train_ids)} | train crops: {len(train_samples)}")
    print(f"[OK] Val images:   {len(val_ids)} | val crops:   {len(val_samples)}")

    return train_samples, val_samples
