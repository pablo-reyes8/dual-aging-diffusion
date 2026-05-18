
import json
import zipfile
import os
import re
from pathlib import Path
from collections import Counter

translated_ethnicities = [
    "A portrait of an older white man, clearly senior, with visible facial signs of aging.",
    "A portrait of a large-built Caucasian adult man.",
    "A portrait of an adult Asian woman with fair skin.",
    "A portrait of an older white man with noticeable signs of aging.",
    "A portrait of an elderly white man with pronounced facial signs of aging.",
    "A portrait of an adult African American woman.",
    "A portrait of an older African woman with clear visible signs of aging.",
    "A portrait of an older European white man with visible facial aging.",
    "A portrait of a young adult white woman.",
    "A portrait of a young adult Asian woman.",
    "A portrait of a young adult Asian woman with fair skin.",
    "A portrait of an elderly white American woman with pronounced signs of aging.",
    "A portrait of a white adult man.",
    "A portrait of a young white American girl.",
    "A portrait of a European white adult man.",
    "A portrait of a European white adult woman.",
    "A portrait of a young blonde American woman.",
    "A portrait of a young white European girl.",
    "A portrait of an elderly white man with pronounced facial signs of aging.",
    "A portrait of an older white man with visible signs of aging.",
    "A portrait of an older white American woman with visible facial aging.",
    "A portrait of an adult African woman with dark skin.",
    "A portrait of a white European girl.",
    "A portrait of an elderly white woman with clear visible signs of aging.",
    "A portrait of a young Asian woman with dark hair.",
    "A portrait of a young woman with dark hair.",
    "A portrait of a white blonde American woman.",
    "A portrait of a very elderly white American man with strong visible signs of aging.",
    "A portrait of an elderly white woman with clear visible signs of aging.",
    "A portrait of an adult Indian woman with a brown complexion.",
    "A portrait of an elderly white American woman with pronounced signs of aging.",
    "A portrait of an elderly white woman with pronounced facial aging.",
    "A portrait of a white adult man.",
    "A portrait of an adult Asian woman with fair skin.",
    "A portrait of an older white man with noticeable signs of aging.",
    "A portrait of a middle-aged white man.",
    "A portrait of a young adult white blonde woman.",
    "A portrait of a young adult white blonde American woman.",
    "A portrait of an older white brunette woman with visible facial aging.",
    "A portrait of an adult Asian woman with dark hair.",
    "A portrait of a young adult white American man.",
    "A portrait of a young adult Asian woman.",
    "A portrait of a European adult man.",
    "A portrait of a European white adult woman.",
    "A portrait of a very young white American girl.",
    "A portrait of an adult Indian man with a brown complexion.",
    "A portrait of an elderly white man with visible signs of aging.",
    "A portrait of a young adult white American woman.",
    "A portrait of a young Indian boy with dark hair.",
    "A portrait of a young adult white American man.",
    "A portrait of a white adult man with dark hair.",
    "A portrait of a young adult Asian man.",
    "A portrait of a young adult African American woman.",
    "A portrait of a young Asian woman with fair skin.",
    "A portrait of an elderly white woman with clear visible signs of aging.",
    "A portrait of a young adult American woman with dark hair."
]



labels_zip = r"/content/results.zip"

ethnicities = []
json_filenames = []

with zipfile.ZipFile(labels_zip, "r") as z:
    json_files = sorted([
        name for name in z.namelist()
        if name.lower().endswith(".json")
    ])

    print(f"JSON files found: {len(json_files)}")

    for json_name in json_files:
        with z.open(json_name) as f:
            data = json.load(f)

        ethnicity = data["global"]["ethnicity"]

        ethnicities.append(ethnicity)
        json_filenames.append(json_name)

print("Number of ethnicities:", len(ethnicities))
print(ethnicities)


labels_zip = r"/content/results.zip"
output_dir = Path("/content/results_translated_json")
output_dir.mkdir(parents=True, exist_ok=True)


def to_region_ethnicity(global_description: str) -> str:
    """
    Convierte:
    'A portrait of an older white man...'
    en:
    'an older white man...'

    Esto sirve para luego concatenar:
    'Forehead of an older white man...'
    """
    text = global_description.strip()

    # quitamos solo el inicio
    patterns = [
        r"^A portrait of an\s+",
        r"^A portrait of a\s+",
        r"^A portrait of\s+",]

    for pattern in patterns:
        if re.match(pattern, text, flags=re.IGNORECASE):
            text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
            break

    # opcional: asegurar primera letra minúscula si quieres concatenar más natural
    # "Forehead of an older..." queda bien
    if text:
        text = text[0].lower() + text[1:]

    return text

# =========================
# LEER ZIP Y MODIFICAR JSONS
# =========================
with zipfile.ZipFile(labels_zip, "r") as z:
    json_files = sorted([
        name for name in z.namelist()
        if name.lower().endswith(".json")])

    print(f"Found {len(json_files)} JSON files.")

    if len(json_files) != len(translated_ethnicities):
        raise ValueError(
            f"Number of JSON files ({len(json_files)}) != "
            f"number of translated descriptions ({len(translated_ethnicities)}).")

    for json_name, global_ethnicity in zip(json_files, translated_ethnicities):
        region_ethnicity = to_region_ethnicity(global_ethnicity)

        with z.open(json_name) as f:
            data = json.load(f)

        # -------------------------
        # GLOBAL -> full sentence
        # -------------------------
        if "global" in data and isinstance(data["global"], dict):
            data["global"]["ethnicity"] = global_ethnicity

        # -------------------------
        # REGIONS -> slots -> no "A portrait..."
        # -------------------------
        if "regions" in data and isinstance(data["regions"], dict):
            for region_key, region_data in data["regions"].items():
                if isinstance(region_data, dict) and "slots" in region_data:
                    for slot in region_data["slots"]:
                        if isinstance(slot, dict) and "ethnicity" in slot:
                            slot["ethnicity"] = region_ethnicity

        # -------------------------
        # ALL_SLOTS -> no "A portrait..."
        # -------------------------
        if "all_slots" in data and isinstance(data["all_slots"], list):
            for slot in data["all_slots"]:
                if isinstance(slot, dict) and "ethnicity" in slot:
                    slot["ethnicity"] = region_ethnicity

        # -------------------------
        # GUARDAR JSON NUEVO
        # -------------------------
        output_path = output_dir / Path(json_name).name
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\nDone. Modified JSONs saved in: {output_dir}")