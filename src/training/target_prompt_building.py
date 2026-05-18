# ============================================================
# PROMPT CONSTRUCTION HELPERS
# Global + Local
#
#
# Main updates:
#   1. Global prompt builder remains aging-biased.
#   2. Local prompt builder now handles high source scores intelligently:
#        - high scores become severe-aging anchors most of the time.
#        - small probability of mild contrast for calibration.
#        - no aggressive rejuvenation.
#   3. Local target prompt wording changes by mode:
#        - aging
#        - anchor
#        - contrast
#   4. Neutral prompts are cleaner:
#        - global neutral removes explicit age and age-correlated attributes.
#        - local neutral removes score and "local aging details".
#
# Use inside training loop:
#
#   global_pack = build_global_prompt_pack_from_loader_prompts(
#       loader_prompts=batch["prompt"],
#       source_ages=batch["age"],
#       p_neutral=0.10,
#   )
#
#   local_pack = build_local_prompt_pack_from_loader_prompts(
#       loader_prompts=batch["prompt"],
#       source_scores=batch["score_raw"],
#       p_neutral=0.10,
#   )
# ============================================================

import re
import random
from typing import List, Dict, Any, Optional, Union

import torch


# ============================================================
# Generic utilities
# ============================================================

def _to_float_list(x):
    """
    Converts tensor/list/scalar into a Python list of floats.
    """
    if torch.is_tensor(x):
        return x.detach().cpu().float().view(-1).tolist()

    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            if torch.is_tensor(v):
                out.append(float(v.detach().cpu().item()))
            else:
                out.append(float(v))
        return out

    return [float(x)]


def _clamp_int(x, low, high):
    return int(max(low, min(high, round(float(x)))))


def _clean_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+%", "%", text)
    return text.strip()


def _choose_prompt(cond_prompt: str, neutral_prompt: str, p_neutral: float):
    """
    CFG-style prompt dropout.
    """
    if random.random() < float(p_neutral):
        return neutral_prompt
    return cond_prompt


def _article_for_phrase(phrase: str) -> str:
    phrase = str(phrase).strip()
    if len(phrase) == 0:
        return "a"
    return "an" if phrase[0].lower() in ["a", "e", "i", "o", "u"] else "a"


# ============================================================
# Gender / age token logic
# ============================================================

def infer_gender_word_from_prompt(prompt: str) -> str:
    """
    Conservative gender extraction from existing loader prompt.

    Returns:
        "man", "woman", or "person".

    We do not infer gender aggressively because accidental gender drift
    is harmful for identity preservation.
    """
    p = str(prompt).lower()

    # Match woman/female first because "male" is substring of "female".
    if re.search(r"\b(woman|female)\b", p):
        return "woman"

    if re.search(r"\b(man|male)\b", p):
        return "man"

    return "person"


def age_gender_person_token(age: Union[int, float], gender_word: str = "person") -> str:
    """
    Age-aware person token with smooth aging categories.

    Inspired by SelfAge-style prompt design, but made more conservative
    for identity preservation:
        < 5      -> baby
        5-14     -> boy/girl/child
        15-44    -> man/woman/person
        45-59    -> middle-aged ...
        60-69    -> older ...
        >=70     -> elderly ...

    For 65+ we preserve gender when known:
        "elderly man", "elderly woman", "elderly person".
    """
    age = float(age)
    gender_word = str(gender_word).lower().strip()

    if age < 5:
        return "baby"

    if age < 15:
        if gender_word == "man":
            return "boy"
        if gender_word == "woman":
            return "girl"
        return "child"

    if age < 45:
        if gender_word in ["man", "woman"]:
            return gender_word
        return "person"

    if age < 60:
        if gender_word == "man":
            return "middle-aged man"
        if gender_word == "woman":
            return "middle-aged woman"
        return "middle-aged person"

    if age < 70:
        if gender_word == "man":
            return "older man"
        if gender_word == "woman":
            return "older woman"
        return "older person"

    if gender_word == "man":
        return "elderly man"
    if gender_word == "woman":
        return "elderly woman"
    return "elderly person"


# ============================================================
# Global prompt helpers
# ============================================================

AGE_PATTERNS = [
    r"\b(\d{1,3})\s*-\s*year\s*-\s*old\b",
    r"\b(\d{1,3})\s+year\s+old\b",
    r"\b(\d{1,3})\s+years\s+old\b",
    r"\bas\s+(\d{1,3})\s*-\s*year\s*-\s*old\b",
    r"\bas\s+(\d{1,3})\s+year\s+old\b",
]


AGE_CORRELATED_ATTRIBUTE_PATTERNS = [
    r",?\s*gray hair",
    r",?\s*grey hair",
    r",?\s*white hair",
    r",?\s*silver hair",
    r",?\s*wrinkled skin",
    r",?\s*visible wrinkles",
    r",?\s*deep wrinkles",
    r",?\s*facial wrinkles",
    r",?\s*older",
    r",?\s*elderly",
    r",?\s*senior",
    r",?\s*middle-aged",
    r",?\s*middle aged",
]


def extract_age_from_prompt(prompt: str) -> Optional[int]:
    p = str(prompt)
    for pat in AGE_PATTERNS:
        m = re.search(pat, p, flags=re.IGNORECASE)
        if m:
            return _clamp_int(m.group(1), 0, 100)
    return None


def remove_age_correlated_attributes_from_neutral(prompt: str) -> str:
    """
    Removes global age-correlated words from neutral prompt.

    Important:
        The source prompt can keep attributes like gray hair.
        The neutral prompt should be cleaner for CFG-style dropout.
    """
    p = str(prompt)

    for pat in AGE_CORRELATED_ATTRIBUTE_PATTERNS:
        p = re.sub(pat, "", p, flags=re.IGNORECASE)

    return _clean_spaces(p)


def remove_age_from_global_prompt(
    prompt: str,
    remove_age_attributes: bool = True,
) -> str:
    """
    Removes explicit age expressions from a global prompt.

    Example:
        'a portrait photo of a 42-year-old man, medium skin tone, gray hair'
        -> 'a portrait photo of a man, medium skin tone'
    """
    p = str(prompt)

    # Remove 'as 42-year-old'
    p = re.sub(
        r"\s+as\s+\d{1,3}\s*-\s*year\s*-\s*old\b",
        "",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(
        r"\s+as\s+\d{1,3}\s+years?\s+old\b",
        "",
        p,
        flags=re.IGNORECASE,
    )

    # Replace 'a 42-year-old man' -> 'a man'
    p = re.sub(
        r"\ba\s+\d{1,3}\s*-\s*year\s*-\s*old\s+",
        "a ",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(
        r"\ban\s+\d{1,3}\s*-\s*year\s*-\s*old\s+",
        "an ",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(
        r"\ba\s+\d{1,3}\s+years?\s+old\s+",
        "a ",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(
        r"\ban\s+\d{1,3}\s+years?\s+old\s+",
        "an ",
        p,
        flags=re.IGNORECASE,
    )

    # Remove isolated age phrases.
    p = re.sub(
        r"\b\d{1,3}\s*-\s*year\s*-\s*old\b",
        "",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(
        r"\b\d{1,3}\s+years?\s+old\b",
        "",
        p,
        flags=re.IGNORECASE,
    )

    if remove_age_attributes:
        p = remove_age_correlated_attributes_from_neutral(p)

    return _clean_spaces(p)


def build_global_target_prompt_from_base(
    base_prompt: str,
    target_age: int,
    preserve_attributes: bool = True,
) -> str:
    """
    Builds a target-age global prompt from the loader prompt.

    Design:
        - Uses alpha-year-old.
        - Uses age-aware person token.
        - Preserves stable visual attributes from the loader prompt when possible.
    """
    base_prompt = str(base_prompt)
    gender = infer_gender_word_from_prompt(base_prompt)
    person_token = age_gender_person_token(target_age, gender)
    article = _article_for_phrase(person_token)

    attr_tail = ""
    if preserve_attributes and "," in base_prompt:
        attr_tail = "," + ",".join(base_prompt.split(",")[1:])
        attr_tail = _clean_spaces(attr_tail)

    prompt = (
        f"a portrait photo of {article} {person_token} "
        f"as {int(target_age)}-year-old"
    )

    if attr_tail:
        prompt = prompt + " " + attr_tail

    return _clean_spaces(prompt)


def sample_global_target_mode(
    source_age: float,
    p_aging: float = 0.78,
    p_anchor: float = 0.17,
    p_mild_younger: float = 0.05,
) -> str:
    """
    Global training target mode.

    Most cases are forward aging.
    Anchor cases teach calibration/reconstruction around the same age.
    Mild younger cases are only a small regularizer.
    """
    u = random.random()

    if u < p_aging:
        return "aging"

    if u < p_aging + p_anchor:
        return "anchor"

    return "mild_younger"


def sample_global_target_age_by_mode(
    source_age: float,
    mode: str,
    min_age: int = 18,
    max_age: int = 90) -> int:
    """
    Samples target age conditional on mode.
    """
    s = float(source_age)

    if mode == "aging":
        if s < 35:
            delta = random.triangular(18, 50, 32)
        elif s < 55:
            delta = random.triangular(10, 38, 22)
        elif s < 70:
            delta = random.triangular(5, 25, 13)
        else:
            delta = random.triangular(2, 15, 7)

        target = s + delta

    elif mode == "anchor":
        target = s + random.triangular(-3, 3, 0)

    elif mode == "mild_younger":
        target = s + random.triangular(-15, -2, -7)

    else:
        raise ValueError(f"Unknown global target mode: {mode}")

    return _clamp_int(target, min_age, max_age)


def build_global_prompt_pack_from_loader_prompts(
    loader_prompts: List[str],
    source_ages,
    p_neutral: float = 0.10,
    min_target_age: int = 18,
    max_target_age: int = 90) -> Dict[str, Any]:
    """
    Main global prompt function.

    Returns:
        source_prompts
        neutral_prompts
        selected_source_prompts
        target_prompts
        target_ages
        target_modes
    """
    source_ages_list = _to_float_list(source_ages)

    if len(loader_prompts) != len(source_ages_list):
        raise ValueError(
            f"Length mismatch: prompts={len(loader_prompts)} "
            f"ages={len(source_ages_list)}"
        )

    source_prompts = [str(p) for p in loader_prompts]
    neutral_prompts = [remove_age_from_global_prompt(p) for p in source_prompts]

    target_modes = [
        sample_global_target_mode(source_age=a)
        for a in source_ages_list
    ]

    target_ages = [
        sample_global_target_age_by_mode(
            source_age=a,
            mode=m,
            min_age=min_target_age,
            max_age=max_target_age,
        )
        for a, m in zip(source_ages_list, target_modes)
    ]

    target_prompts = [
        build_global_target_prompt_from_base(
            base_prompt=p,
            target_age=t,
            preserve_attributes=True,
        )
        for p, t in zip(source_prompts, target_ages)
    ]

    selected_source_prompts = [
        _choose_prompt(src, neu, p_neutral=p_neutral)
        for src, neu in zip(source_prompts, neutral_prompts)
    ]

    return {
        "source_prompts": source_prompts,
        "neutral_prompts": neutral_prompts,
        "selected_source_prompts": selected_source_prompts,
        "target_prompts": target_prompts,
        "target_ages": torch.tensor(target_ages, dtype=torch.float32),
        "target_modes": target_modes,
    }


# ============================================================
# Local score / prompt helpers
# ============================================================

def extract_score_from_local_prompt(prompt: str) -> Optional[int]:
    """
    Extracts local aging score from prompt.

    Matches:
        'aging score of 70%'
        'aging score 70%'
        'score of 70%'
    """
    p = str(prompt)

    patterns = [
        r"aging\s+score\s+of\s+(\d{1,3})\s*%",
        r"aging\s+score\s+(\d{1,3})\s*%",
        r"score\s+of\s+(\d{1,3})\s*%"]

    for pat in patterns:
        m = re.search(pat, p, flags=re.IGNORECASE)
        if m:
            return _clamp_int(m.group(1), 0, 100)

    return None


def remove_score_from_local_prompt(prompt: str) -> str:
    """
    Removes local aging score and explicit aging semantics for neutral local prompt.

    Example:
        '..., showing facial skin texture and local aging details,
          with an aging score of 87%, for a white person'

        -> '..., showing facial skin texture, for a white person'
    """
    p = str(prompt)

    p = re.sub(
        r",?\s*with\s+an\s+aging\s+score\s+of\s+\d{1,3}\s*%",
        "",
        p,
        flags=re.IGNORECASE,
    )

    p = re.sub(
        r",?\s*with\s+aging\s+score\s+of\s+\d{1,3}\s*%",
        "",
        p,
        flags=re.IGNORECASE,
    )

    p = re.sub(
        r",?\s*aging\s+score\s+of\s+\d{1,3}\s*%",
        "",
        p,
        flags=re.IGNORECASE)

    p = p.replace(" and local aging details", "")
    p = p.replace("local aging details", "facial skin texture")
    p = p.replace("age-related facial texture", "facial skin texture")

    return _clean_spaces(p)


def _split_local_prompt_before_for(prompt: str):
    """
    Splits a local prompt into:
        left part before ', for ...'
        demographic part after ', for ...'

    If no ', for ' exists, returns full prompt and None.
    """
    p = str(prompt)

    if ", for " in p:
        left, demo = p.rsplit(", for ", 1)
        return left.strip(), demo.strip()

    return p.strip(), None


def remove_existing_local_score_phrase(prompt: str) -> str:
    """
    Removes score phrase but keeps 'local aging details' if present.
    Used for rebuilding target prompts.
    """
    p = str(prompt)

    p = re.sub(
        r",?\s*with\s+an\s+aging\s+score\s+of\s+\d{1,3}\s*%",
        "",
        p,
        flags=re.IGNORECASE)

    p = re.sub(
        r",?\s*with\s+aging\s+score\s+of\s+\d{1,3}\s*%",
        "",
        p,
        flags=re.IGNORECASE)

    p = re.sub(
        r",?\s*aging\s+score\s+of\s+\d{1,3}\s*%",
        "",
        p,
        flags=re.IGNORECASE)

    return _clean_spaces(p)


def local_score_descriptor(score: int, mode: str) -> str:
    """
    Produces coherent local score wording depending on mode and score.

    The goal is especially to handle high source/target scores coherently:
        - 95-100 should become explicit severe-aging anchors.
        - anchor mode should tell the model this is stable/severe evidence,
          not a useless no-op.
    """
    s = int(score)
    mode = str(mode)

    if mode == "aging":
        if s >= 95:
            return f"with a severe local aging score of {s}%"
        if s >= 85:
            return f"with a strong local aging score of {s}%"
        if s >= 65:
            return f"with a pronounced local aging score of {s}%"
        return f"with an aging score of {s}%"

    if mode == "anchor":
        if s >= 95:
            return f"with a stable severe local aging score of {s}%"
        if s >= 85:
            return f"with a stable strong local aging score of {s}%"
        if s >= 65:
            return f"with a stable pronounced local aging score of {s}%"
        return f"with a stable local aging score of {s}%"

    if mode == "contrast":
        # This is not strong rejuvenation. It is calibration.
        if s >= 75:
            return f"with a reduced but still realistic local aging score of {s}%"
        return f"with a milder local aging score of {s}%"

    return f"with an aging score of {s}%"


def build_local_target_prompt_from_base(
    base_prompt: str,
    target_score: int,
    target_mode: str) -> str:
    """
    Builds a target local prompt from the loader prompt.

    It keeps:
        - crop wording
        - region wording
        - demographic phrase

    It replaces:
        - score phrase
        - score wording depending on target mode
    """
    target_score = _clamp_int(target_score, 0, 100)
    target_mode = str(target_mode)

    p_no_score = remove_existing_local_score_phrase(base_prompt)
    left, demo = _split_local_prompt_before_for(p_no_score)

    descriptor = local_score_descriptor(target_score, target_mode)

    prompt = f"{left}, {descriptor}"

    if demo is not None and len(demo) > 0:
        prompt = f"{prompt}, for {demo}"

    return _clean_spaces(prompt)


# ============================================================
# Local target mode and score sampling
# ============================================================

def sample_local_target_mode(
    source_score: float,
    p_aging: float = 0.75,
    p_anchor: float = 0.20,
    p_contrast: float = 0.05) -> str:
    """
    Samples local target mode.

    Core idea:
        Most samples still train forward local aging.

    Special high-score behavior:
        For source_score >= 95:
            These are already severe-aging examples.
            They should mostly act as positive anchors.
    """
    s = float(source_score)

    if s >= 95.0:
        p_aging = 0.20
        p_anchor = 0.65
        p_contrast = 0.15

    elif s >= 85.0:
        p_aging = 0.50
        p_anchor = 0.35
        p_contrast = 0.15

    u = random.random()

    if u < p_aging:
        return "aging"

    if u < p_aging + p_anchor:
        return "anchor"

    return "contrast"


def sample_local_target_score_by_mode(
    source_score: float,
    mode: str) -> int:
    """
    Samples local target score conditional on target mode.

    Modes:
        aging:
            target score is higher or high/severe.

        anchor:
            target score is equal or very close to source.
            For high scores, this teaches severe-aging appearance.

        contrast:
            mild downward contrast, mostly useful for high scores.
            This is NOT aggressive rejuvenation.
    """
    s = float(source_score)
    mode = str(mode)

    if mode == "aging":
        if s >= 95.0:
            # Already saturated; keep as severe positive example.
            target = random.triangular(95, 100, 100)

        elif s >= 85.0:
            # Push/keep near upper severity.
            target = random.triangular(max(s, 90), 100, 97)

        elif s >= 65.0:
            # High-ish source: likely 80-100.
            min_target = max(s + 8, 78)
            max_target = 100
            mode_value = min(95, max(85, s + 18))
            target = random.triangular(min_target, max_target, mode_value)

        else:
            # Low/mid source: strong forward local aging.
            min_target = max(s + 12, 45)
            max_target = 100
            mode_value = min(92, max(70, s + 30))
            target = random.triangular(min_target, max_target, mode_value)

    elif mode == "anchor":
        # Same score / almost same score.
        # For very high scores, keep very close.
        if s >= 95.0:
            target = random.gauss(s, 1.5)
            target = max(94.0, target)

        elif s >= 85.0:
            target = random.gauss(s, 2.5)
            target = max(82.0, target)

        elif s >= 65.0:
            target = random.gauss(s, 4.0)

        else:
            target = random.gauss(s, 5.0)

    elif mode == "contrast":
        # Mild downward calibration.
        # Do not make 100 -> 20. That would teach rejuvenation too strongly.
        if s >= 95.0:
            target = random.triangular(68, 90, 80)

        elif s >= 85.0:
            target = random.triangular(62, max(82, s - 2), 75)

        elif s >= 65.0:
            target = random.triangular(45, max(65, s - 4), 58)

        else:
            # For low/mid source, contrast is near-source, not strong downshift.
            target = random.triangular(max(0, s - 12), s + 4, max(0, s - 3))

    else:
        raise ValueError(f"Unknown local target mode: {mode}")

    return _clamp_int(target, 0, 100)


def build_local_prompt_pack_from_loader_prompts(
    loader_prompts: List[str],
    source_scores,
    p_neutral: float = 0.10,
) -> Dict[str, Any]:
    """
    Main local prompt function.

    Inputs:
        loader_prompts:
            batch["prompt"]

        source_scores:
            batch["score_raw"] or batch["score"].
            If batch["score"] is normalized [0,1], this function auto-detects it.

    Returns:
        source_prompts:
            original local prompts from loader.

        neutral_prompts:
            score and explicit aging wording removed.

        selected_source_prompts:
            source prompt after CFG-style dropout.
            Use this for L_full.

        target_prompts:
            local prompt with target score and target-mode coherent wording.
            Use this for L_score / target editing.

        target_scores:
            Tensor [B] in [0,1].

        target_scores_raw:
            Tensor [B] in [0,100].

        target_modes:
            list[str], one of {"aging", "anchor", "contrast"}.
    """
    source_scores_list = _to_float_list(source_scores)

    # Auto-detect normalized scores.
    if len(source_scores_list) > 0 and max(source_scores_list) <= 1.0:
        source_scores_list = [100.0 * s for s in source_scores_list]

    if len(loader_prompts) != len(source_scores_list):
        raise ValueError(
            f"Length mismatch: prompts={len(loader_prompts)} "
            f"scores={len(source_scores_list)}"
        )

    source_prompts = [str(p) for p in loader_prompts]
    neutral_prompts = [remove_score_from_local_prompt(p) for p in source_prompts]

    target_modes = [
        sample_local_target_mode(source_score=s)
        for s in source_scores_list
    ]

    target_scores_raw = [
        sample_local_target_score_by_mode(source_score=s, mode=m)
        for s, m in zip(source_scores_list, target_modes)
    ]

    target_prompts = [
        build_local_target_prompt_from_base(
            base_prompt=p,
            target_score=t,
            target_mode=m,
        )
        for p, t, m in zip(source_prompts, target_scores_raw, target_modes)
    ]

    selected_source_prompts = [
        _choose_prompt(src, neu, p_neutral=p_neutral)
        for src, neu in zip(source_prompts, neutral_prompts)
    ]

    target_scores_raw_tensor = torch.tensor(target_scores_raw, dtype=torch.float32)
    target_scores_tensor = target_scores_raw_tensor / 100.0

    return {
        "source_prompts": source_prompts,
        "neutral_prompts": neutral_prompts,
        "selected_source_prompts": selected_source_prompts,
        "target_prompts": target_prompts,
        "target_scores": target_scores_tensor,
        "target_scores_raw": target_scores_raw_tensor,
        "target_modes": target_modes,
    }


# ============================================================
# Debug helpers
# ============================================================

def print_prompt_pack_examples(prompt_pack: Dict[str, Any], n: int = 3):
    n = min(n, len(prompt_pack["source_prompts"]))

    for i in range(n):
        print("=" * 80)
        print("SOURCE:")
        print(prompt_pack["source_prompts"][i])

        print("\nNEUTRAL:")
        print(prompt_pack["neutral_prompts"][i])

        print("\nSELECTED SOURCE:")
        print(prompt_pack["selected_source_prompts"][i])

        print("\nTARGET:")
        print(prompt_pack["target_prompts"][i])

        if "target_ages" in prompt_pack:
            print("\nTARGET AGE:", prompt_pack["target_ages"][i].item())

        if "target_scores_raw" in prompt_pack:
            print("\nTARGET SCORE:", prompt_pack["target_scores_raw"][i].item())

        if "target_modes" in prompt_pack:
            print("TARGET MODE:", prompt_pack["target_modes"][i])


def audit_local_prompt_sampling(
    loader,
    n_batches: int = 20,
    p_neutral: float = 0.10,
):
    """
    Quick diagnostic for local prompt sampling.

    Useful to verify:
        - high scores produce many anchor modes.
        - target scores are coherent.
        - prompts look semantically correct.
    """
    all_source_scores = []
    all_target_scores = []
    all_modes = []

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        pack = build_local_prompt_pack_from_loader_prompts(
            loader_prompts=batch["prompt"],
            source_scores=batch["score_raw"],
            p_neutral=p_neutral,
        )

        source_scores = _to_float_list(batch["score_raw"])
        target_scores = _to_float_list(pack["target_scores_raw"])

        all_source_scores.extend(source_scores)
        all_target_scores.extend(target_scores)
        all_modes.extend(pack["target_modes"])

    source_t = torch.tensor(all_source_scores, dtype=torch.float32)
    target_t = torch.tensor(all_target_scores, dtype=torch.float32)

    print("\n========== LOCAL PROMPT SAMPLING AUDIT ==========")
    print("Inspected samples:", len(all_source_scores))

    print("\n[Source score stats]")
    print(
        f"min/mean/median/max: "
        f"{source_t.min().item():.1f} / "
        f"{source_t.mean().item():.1f} / "
        f"{source_t.median().item():.1f} / "
        f"{source_t.max().item():.1f}"
    )

    print("\n[Target score stats]")
    print(
        f"min/mean/median/max: "
        f"{target_t.min().item():.1f} / "
        f"{target_t.mean().item():.1f} / "
        f"{target_t.median().item():.1f} / "
        f"{target_t.max().item():.1f}"
    )

    print("\n[Modes]")
    mode_counts = {}
    for m in all_modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1

    total = max(1, len(all_modes))
    for m, c in sorted(mode_counts.items()):
        print(f"{m:10s}: {c:4d} | {100.0 * c / total:6.2f}%")

    print("\n[High-score behavior]")
    high_mask = source_t >= 95
    if high_mask.any():
        high_targets = target_t[high_mask]
        high_modes = [m for m, s in zip(all_modes, all_source_scores) if s >= 95]

        print("source >= 95 samples:", int(high_mask.sum().item()))
        print(
            f"target min/mean/median/max: "
            f"{high_targets.min().item():.1f} / "
            f"{high_targets.mean().item():.1f} / "
            f"{high_targets.median().item():.1f} / "
            f"{high_targets.max().item():.1f}"
        )

        high_mode_counts = {}
        for m in high_modes:
            high_mode_counts[m] = high_mode_counts.get(m, 0) + 1

        for m, c in sorted(high_mode_counts.items()):
            print(f"  high {m:10s}: {c:4d} | {100.0 * c / len(high_modes):6.2f}%")
    else:
        print("No source >= 95 found in inspected batches.")