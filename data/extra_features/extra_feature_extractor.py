# ============================================================
# INCREMENTAL FACE ATTRIBUTE PSEUDO-LABELING
# Only runs Hugging Face on images not already enriched
# ============================================================

%pip -q install transformers accelerate timm einops

from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd
import torch
from transformers import pipeline


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

GLOBAL_IMAGE_DIR = Path("/content/ffhq_subset_6k_extremes_local")

# Full base CSV with all images and age predictions
BASE_CSV_PATH = Path("/content/ffhq_predictions.csv")

# Existing enriched CSV, partial or complete
ENRICHED_CSV_PATH = Path("/content/ffhq_face_attribute_prompts.csv")

# Same file will be updated incrementally
OUT_CSV_PATH = ENRICHED_CSV_PATH

# Optional: save the subset that was newly processed
NEW_ROWS_CSV_PATH = Path("/content/ffhq_face_attribute_prompts_new_rows.csv")

# Optional: save images still missing from disk
MISSING_IMAGES_CSV_PATH = Path("/content/ffhq_missing_images_for_attribute_prompting.csv")

device = 0 if torch.cuda.is_available() else -1

print("Device:", "cuda" if device == 0 else "cpu")
print("Images:", GLOBAL_IMAGE_DIR)
print("Base CSV:", BASE_CSV_PATH)
print("Existing enriched CSV:", ENRICHED_CSV_PATH)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def valid_text(x) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0 and x.strip().lower() not in ["nan", "none", "null"]


def filename_key(x) -> str:
    return Path(str(x).strip()).name


def stem_key(x) -> str:
    return Path(str(x).strip()).stem


def clean_gender_label(raw_label):
    label = str(raw_label).lower()

    if "female" in label or "woman" in label:
        return "female"
    if "male" in label or "man" in label:
        return "male"

    if label in ["label_0", "0"]:
        return "male"
    if label in ["label_1", "1"]:
        return "female"

    return label


def simplify_zero_shot_label(label):
    label = str(label)

    replacements = {
        "a face with light skin tone": "light skin tone",
        "a face with medium skin tone": "medium skin tone",
        "a face with dark skin tone": "dark skin tone",
        "a face with black hair": "black hair",
        "a face with brown hair": "brown hair",
        "a face with blonde hair": "blonde hair",
        "a face with gray hair": "gray hair",
        "a face with white hair": "white hair",
        "a face with red hair": "red hair",
        "a bald or shaved head": "bald or shaved head",
        "a face wearing glasses": "wearing glasses",
        "a face without glasses": "without glasses",
    }

    return replacements.get(label, label)


def top_prediction(preds):
    pred = sorted(preds, key=lambda x: x["score"], reverse=True)[0]
    return pred["label"], float(pred["score"])


def make_enriched_prompt(
    age,
    gender_label=None,
    gender_confidence=None,
    skin_tone_label=None,
    skin_tone_confidence=None,
    hair_label=None,
    hair_confidence=None,
    glasses_label=None,
    glasses_confidence=None,
    gender_threshold=0.70,
    skin_threshold=0.40,
    hair_threshold=0.35,
    glasses_threshold=0.55,
):
    age_int = int(round(float(age)))
    age_int = max(0, min(100, age_int))

    # Gender/sex descriptor
    if gender_label is not None and gender_confidence is not None:
        if gender_confidence >= gender_threshold:
            if gender_label == "male":
                person_phrase = f"a {age_int}-year-old man"
            elif gender_label == "female":
                person_phrase = f"a {age_int}-year-old woman"
            else:
                person_phrase = f"a {age_int}-year-old person"
        else:
            person_phrase = f"a {age_int}-year-old person"
    else:
        person_phrase = f"a {age_int}-year-old person"

    # Visible attributes
    attr_parts = []

    if skin_tone_label is not None and skin_tone_confidence is not None:
        if skin_tone_confidence >= skin_threshold:
            attr_parts.append(skin_tone_label)

    if hair_label is not None and hair_confidence is not None:
        if hair_confidence >= hair_threshold:
            attr_parts.append(hair_label)

    if glasses_label is not None and glasses_confidence is not None:
        if glasses_confidence >= glasses_threshold and glasses_label == "wearing glasses":
            attr_parts.append("wearing glasses")

    if len(attr_parts) > 0:
        return "a portrait photo of " + person_phrase + ", " + ", ".join(attr_parts)

    return "a portrait photo of " + person_phrase


def make_fallback_prompt(age, gender_pred=None):
    age_int = int(round(float(age)))
    age_int = max(0, min(100, age_int))

    gender_label = clean_gender_label(gender_pred)

    if gender_label == "male":
        return f"a portrait photo of a {age_int}-year-old man"
    if gender_label == "female":
        return f"a portrait photo of a {age_int}-year-old woman"

    return f"a portrait photo of a {age_int}-year-old person"


# ------------------------------------------------------------
# Build image index
# ------------------------------------------------------------

image_paths = [
    p for p in GLOBAL_IMAGE_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTS
    and not p.name.startswith("._")
    and "__MACOSX" not in str(p)
]

image_index = {}
for p in image_paths:
    image_index[p.name] = p
    image_index[p.stem] = p

print("\n[Image index]")
print("Found image files:", len(image_paths))
print("Index keys:", len(image_index))


def resolve_image_path(filename):
    filename = str(filename)
    stem = Path(filename).stem
    name = Path(filename).name

    if filename in image_index:
        return image_index[filename]
    if name in image_index:
        return image_index[name]
    if stem in image_index:
        return image_index[stem]

    return None


# ------------------------------------------------------------
# Load base full CSV
# ------------------------------------------------------------

if not BASE_CSV_PATH.exists():
    raise FileNotFoundError(f"Base CSV not found: {BASE_CSV_PATH}")

base_df = pd.read_csv(BASE_CSV_PATH).copy()

if "filename" not in base_df.columns:
    raise ValueError(f"Expected column 'filename'. Found: {list(base_df.columns)}")

if "age_pred" not in base_df.columns:
    raise ValueError(f"Expected column 'age_pred'. Found: {list(base_df.columns)}")

if "gender_pred" not in base_df.columns:
    base_df["gender_pred"] = None

base_df["age_pred"] = pd.to_numeric(base_df["age_pred"], errors="coerce")
base_df = base_df.dropna(subset=["filename", "age_pred"]).reset_index(drop=True)

base_df["filename_key"] = base_df["filename"].apply(filename_key)
base_df["stem_key"] = base_df["filename"].apply(stem_key)
base_df["image_path"] = base_df["filename"].apply(resolve_image_path)

missing_image_df = base_df[base_df["image_path"].isna()].copy()
missing_image_df.to_csv(MISSING_IMAGES_CSV_PATH, index=False)

df_available = base_df.dropna(subset=["image_path"]).reset_index(drop=True)

print("\n[Base CSV]")
print("Base rows:", len(base_df))
print("Rows with local image:", len(df_available))
print("Rows missing local image:", len(missing_image_df))
print("Missing image CSV:", MISSING_IMAGES_CSV_PATH)


# ------------------------------------------------------------
# Load existing enriched CSV and identify already processed
# ------------------------------------------------------------

expected_enriched_cols = [
    "filename",
    "age_pred",
    "gender_label",
    "gender_confidence",
    "skin_tone_label",
    "skin_tone_confidence",
    "hair_label",
    "hair_confidence",
    "glasses_label",
    "glasses_confidence",
    "face_attribute_phrase",
    "enriched_prompt",
    "image_path",
]

if ENRICHED_CSV_PATH.exists():
    existing_df = pd.read_csv(ENRICHED_CSV_PATH).copy()

    if "filename" not in existing_df.columns:
        raise ValueError(
            f"Existing enriched CSV has no 'filename'. Columns: {list(existing_df.columns)}"
        )

    if "enriched_prompt" not in existing_df.columns:
        raise ValueError(
            f"Existing enriched CSV has no 'enriched_prompt'. Columns: {list(existing_df.columns)}"
        )

    existing_df["filename_key"] = existing_df["filename"].apply(filename_key)
    existing_df["stem_key"] = existing_df["filename"].apply(stem_key)

    existing_df["_has_valid_enriched_prompt"] = existing_df["enriched_prompt"].apply(valid_text)

    already_done_stems = set(
        existing_df.loc[existing_df["_has_valid_enriched_prompt"], "stem_key"].astype(str)
    )

    print("\n[Existing enriched CSV]")
    print("Existing rows:", len(existing_df))
    print("Already valid enriched prompts:", len(already_done_stems))

else:
    existing_df = pd.DataFrame(columns=expected_enriched_cols + ["filename_key", "stem_key"])
    already_done_stems = set()

    print("\n[Existing enriched CSV]")
    print("No existing enriched CSV found. Processing all available images.")


# ------------------------------------------------------------
# Keep only images that are not already enriched
# ------------------------------------------------------------

todo_df = df_available[~df_available["stem_key"].astype(str).isin(already_done_stems)].copy()
todo_df = todo_df.reset_index(drop=True)

print("\n[Todo]")
print("Images available:", len(df_available))
print("Already enriched:", len(already_done_stems))
print("Need HF forward:", len(todo_df))

if len(todo_df) == 0:
    print("\n[OK] Nothing new to process. Existing enriched CSV is already up to date.")
else:
    print("\n[First 10 to process]")



# ------------------------------------------------------------
# Stop early if nothing to process
# ------------------------------------------------------------

if len(todo_df) > 0:

    # ------------------------------------------------------------
    # Load models only if needed
    # ------------------------------------------------------------

    print("\n[Loading models]")

    gender_pipe = pipeline(
        task="image-classification",
        model="syntheticbot/gender-classification-clip",
        device=device,
    )

    zero_shot_pipe = pipeline(
        task="zero-shot-image-classification",
        model="openai/clip-vit-large-patch14",
        device=device,
    )

    # ------------------------------------------------------------
    # Candidate labels
    # ------------------------------------------------------------

    skin_tone_labels = [
        "a face with light skin tone",
        "a face with medium skin tone",
        "a face with dark skin tone",
    ]

    hair_labels = [
        "a face with black hair",
        "a face with brown hair",
        "a face with blonde hair",
        "a face with gray hair",
        "a face with white hair",
        "a face with red hair",
        "a bald or shaved head",
    ]

    glasses_labels = [
        "a face wearing glasses",
        "a face without glasses",
    ]

    # ------------------------------------------------------------
    # Inference loop only on missing rows
    # ------------------------------------------------------------

    new_rows = []

    for _, row in tqdm(todo_df.iterrows(), total=len(todo_df)):
        filename = row["filename"]
        image_path = Path(row["image_path"])
        age = float(row["age_pred"])
        gender_pred = row["gender_pred"] if "gender_pred" in row.index else None

        try:
            image = Image.open(image_path).convert("RGB")

            # Gender
            gender_preds = gender_pipe(image)
            raw_gender_label, gender_conf = top_prediction(gender_preds)
            gender_label = clean_gender_label(raw_gender_label)

            # Skin tone
            skin_preds = zero_shot_pipe(image, candidate_labels=skin_tone_labels)
            raw_skin_label, skin_conf = top_prediction(skin_preds)
            skin_label = simplify_zero_shot_label(raw_skin_label)

            # Hair
            hair_preds = zero_shot_pipe(image, candidate_labels=hair_labels)
            raw_hair_label, hair_conf = top_prediction(hair_preds)
            hair_label = simplify_zero_shot_label(raw_hair_label)

            # Glasses
            glasses_preds = zero_shot_pipe(image, candidate_labels=glasses_labels)
            raw_glasses_label, glasses_conf = top_prediction(glasses_preds)
            glasses_label = simplify_zero_shot_label(raw_glasses_label)

            face_attribute_phrase = ", ".join(
                [
                    x for x in [
                        gender_label if gender_conf >= 0.70 else None,
                        skin_label if skin_conf >= 0.40 else None,
                        hair_label if hair_conf >= 0.35 else None,
                        glasses_label if glasses_conf >= 0.55 and glasses_label == "wearing glasses" else None,
                    ]
                    if x is not None
                ]
            )

            enriched_prompt = make_enriched_prompt(
                age=age,
                gender_label=gender_label,
                gender_confidence=gender_conf,
                skin_tone_label=skin_label,
                skin_tone_confidence=skin_conf,
                hair_label=hair_label,
                hair_confidence=hair_conf,
                glasses_label=glasses_label,
                glasses_confidence=glasses_conf,
            )

            new_rows.append(
                {
                    "filename": filename,
                    "age_pred": age,
                    "gender_pred": gender_pred,
                    "gender_label": gender_label,
                    "gender_confidence": gender_conf,
                    "skin_tone_label": skin_label,
                    "skin_tone_confidence": skin_conf,
                    "hair_label": hair_label,
                    "hair_confidence": hair_conf,
                    "glasses_label": glasses_label,
                    "glasses_confidence": glasses_conf,
                    "face_attribute_phrase": face_attribute_phrase,
                    "enriched_prompt": enriched_prompt,
                    "image_path": str(image_path),
                    "error": None,
                }
            )

        except Exception as e:
            # Important:
            # We do NOT leave prompt empty. We create a safe fallback.
            # But error is stored so you know this row was not truly enriched.
            fallback_prompt = make_fallback_prompt(age=age, gender_pred=gender_pred)

            new_rows.append(
                {
                    "filename": filename,
                    "age_pred": age,
                    "gender_pred": gender_pred,
                    "gender_label": None,
                    "gender_confidence": None,
                    "skin_tone_label": None,
                    "skin_tone_confidence": None,
                    "hair_label": None,
                    "hair_confidence": None,
                    "glasses_label": None,
                    "glasses_confidence": None,
                    "face_attribute_phrase": "",
                    "enriched_prompt": fallback_prompt,
                    "image_path": str(image_path),
                    "error": str(e),
                }
            )

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(NEW_ROWS_CSV_PATH, index=False)

    print("\n[New rows]")
    print("New rows processed:", len(new_df))
    print("Saved new rows:", NEW_ROWS_CSV_PATH)

    # ------------------------------------------------------------
    # Merge existing + new, deduplicate by stem
    # ------------------------------------------------------------

    # Clean internal columns if present
    existing_clean = existing_df.drop(
        columns=["filename_key", "stem_key", "_has_valid_enriched_prompt"],
        errors="ignore",
    ).copy()

    combined_df = pd.concat([existing_clean, new_df], ignore_index=True)

    combined_df["filename_key"] = combined_df["filename"].apply(filename_key)
    combined_df["stem_key"] = combined_df["filename"].apply(stem_key)

    # Prefer rows without error and with valid enriched_prompt.
    combined_df["_valid_prompt"] = combined_df["enriched_prompt"].apply(valid_text)
    combined_df["_has_error"] = combined_df["error"].notna() if "error" in combined_df.columns else False

    combined_df = combined_df.sort_values(
        by=["stem_key", "_valid_prompt", "_has_error"],
        ascending=[True, False, True],
    )

    combined_df = combined_df.drop_duplicates(subset=["stem_key"], keep="first")

    combined_df = combined_df.drop(
        columns=["filename_key", "stem_key", "_valid_prompt", "_has_error"],
        errors="ignore",
    )

    # Keep a stable column order
    preferred_cols = [
        "filename",
        "age_pred",
        "gender_pred",
        "gender_label",
        "gender_confidence",
        "skin_tone_label",
        "skin_tone_confidence",
        "hair_label",
        "hair_confidence",
        "glasses_label",
        "glasses_confidence",
        "face_attribute_phrase",
        "enriched_prompt",
        "image_path",
        "error",
    ]

    ordered_cols = [c for c in preferred_cols if c in combined_df.columns]
    other_cols = [c for c in combined_df.columns if c not in ordered_cols]
    combined_df = combined_df[ordered_cols + other_cols]

    combined_df.to_csv(OUT_CSV_PATH, index=False)

    print("\n[Updated enriched CSV]")
    print("Saved:", OUT_CSV_PATH)
    print("Total enriched CSV rows:", len(combined_df))
    print("Rows with valid enriched_prompt:", int(combined_df["enriched_prompt"].apply(valid_text).sum()))



else:
    combined_df = existing_df.drop(
        columns=["filename_key", "stem_key", "_has_valid_enriched_prompt"],
        errors="ignore",
    ).copy()

    print("\n[No update needed]")
    print("Current enriched rows:", len(combined_df))
