from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}

    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        if config_path.suffix.lower() == ".json":
            return json.load(f)

    raise ValueError(f"Unsupported config format: {config_path}")


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_config_summary(config: Dict[str, Any]) -> None:
    print("\n========== CONFIG SUMMARY ==========")
    print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
