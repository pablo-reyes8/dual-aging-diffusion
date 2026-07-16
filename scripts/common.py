from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(
    path: Optional[str | Path],
    _seen: Optional[set[Path]] = None,
) -> Dict[str, Any]:
    """Load YAML/JSON, with optional ``_base_`` inheritance for experiment files."""
    if path is None:
        return {}

    config_path = Path(path)
    if not config_path.is_absolute():
        cwd_candidate = config_path.resolve()
        repo_candidate = (REPO_ROOT / config_path).resolve()
        config_path = cwd_candidate if cwd_candidate.exists() else repo_candidate
    config_path = config_path.resolve()

    _seen = set() if _seen is None else set(_seen)
    if config_path in _seen:
        raise ValueError(f"Cyclic _base_ config reference involving {config_path}")
    _seen.add(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            config = yaml.safe_load(f) or {}
        elif config_path.suffix.lower() == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")

    base_ref = config.pop("_base_", None)
    if base_ref is None:
        return config

    base_refs = [base_ref] if isinstance(base_ref, (str, Path)) else list(base_ref)
    merged: Dict[str, Any] = {}
    for ref in base_refs:
        ref_path = Path(ref)
        if not ref_path.is_absolute():
            ref_path = config_path.parent / ref_path
        merged = deep_update(merged, load_config(str(ref_path), _seen=_seen))
    return deep_update(merged, config)


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
