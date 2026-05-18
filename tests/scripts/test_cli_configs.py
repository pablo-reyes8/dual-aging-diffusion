import json
import subprocess
import sys
from pathlib import Path

from scripts.common import deep_update, load_config
from scripts.inference_cli import DEFAULT_CONFIG as INFERENCE_DEFAULTS
from scripts.train_cli import DEFAULT_CONFIG as TRAIN_DEFAULTS


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_training_config_has_required_orchestration_sections():
    config = deep_update(TRAIN_DEFAULTS, load_config(REPO_ROOT / "configs/training/default_train.yaml"))

    assert config["data"]["batch_size"] > 0
    assert config["data"]["num_workers"] >= 0
    assert config["models"]["global_model_id"]
    assert config["models"]["local_model_id"]
    assert config["adapters"]["global"]["adapter_type"] in {"lora", "dora"}
    assert config["adapters"]["local"]["adapter_type"] in {"lora", "dora"}
    assert set(config["training"]["train_order"]) <= {"local", "global"}
    assert config["training"]["num_epochs"] >= 1


def test_inference_config_and_example_local_spec_are_complete():
    config = deep_update(INFERENCE_DEFAULTS, load_config(REPO_ROOT / "configs/inference/default_inference.yaml"))
    spec_path = REPO_ROOT / "configs/inference/local_spec.example.json"
    specs = json.loads(spec_path.read_text(encoding="utf-8"))["crops"]

    assert config["checkpoints"]["strict_adapter"] is True
    assert config["generation"]["global_num_inference_steps"] > 0
    assert config["fusion"]["color_match"] is True
    assert specs
    assert all(len(item["bbox"]) == 4 for item in specs)
    assert all(item["prompt"] for item in specs)


def test_training_cli_dry_run_does_not_load_models():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_cli",
            "--config",
            "configs/training/default_train.yaml",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Models/data were not loaded" in result.stdout


def test_inference_cli_dry_run_validates_args_without_checkpoint_files():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.inference_cli",
            "--config",
            "configs/inference/default_inference.yaml",
            "--image",
            "missing_input.png",
            "--global-prompt",
            "a portrait photo of an elderly person",
            "--local-spec",
            "configs/inference/local_spec.example.json",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Models were not loaded" in result.stdout
