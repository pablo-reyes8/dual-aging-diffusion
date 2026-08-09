import json
import subprocess
import sys
from pathlib import Path

from scripts.common import deep_update, load_config
from scripts.inference_cli import DEFAULT_CONFIG as INFERENCE_DEFAULTS
from scripts.train_cli import DEFAULT_CONFIG as TRAIN_DEFAULTS
from scripts.train_cli import load_training_config


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
    assert config["generation"]["local_generation_method"] == "img2img"
    assert config["generation"]["local_inversion"]["enabled"] is False
    assert config["fusion"]["color_match"] is True
    assert config["refiner"]["inversion"]["enabled"] is False
    assert specs
    assert all(len(item["bbox"]) == 4 for item in specs)
    assert all(item["prompt"] for item in specs)

    inversion = deep_update(
        INFERENCE_DEFAULTS,
        load_config(REPO_ROOT / "configs/inference/ddim_inversion.yaml"),
    )
    assert inversion["generation"]["local_generation_method"] == "ddim_inversion"
    assert inversion["generation"]["local_inversion"]["enabled"] is True
    assert inversion["refiner"]["inversion"]["enabled"] is False


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


def test_paired_training_yamls_inherit_defaults_and_select_dataset():
    fgnet = load_training_config(REPO_ROOT / "configs/training/paired_fgnet_train.yaml")
    agedb = load_training_config(REPO_ROOT / "configs/training/paired_agedb_train.yaml")

    assert fgnet["models"]["global_model_id"] == TRAIN_DEFAULTS["models"]["global_model_id"]
    assert fgnet["paired_supervision"]["enabled"] is True
    assert fgnet["paired_supervision"]["dataset"] == "fgnet"
    assert fgnet["paired_supervision"]["weight"] == 0.25
    assert agedb["paired_supervision"]["enabled"] is True
    assert agedb["paired_supervision"]["dataset"] == "agedb"
    assert agedb["paired_supervision"]["weight"] == 0.20


def test_explicit_high_level_paired_values_override_data_yaml():
    config = load_training_config(REPO_ROOT / "configs/training/paired_fgnet_train.yaml")
    config["paired_supervision"]["dataset"] = "agedb"
    config["paired_supervision"]["weight"] = 0.10

    from scripts.train_cli import resolve_paired_supervision_config

    resolved = resolve_paired_supervision_config(config)
    assert resolved["dataset"] == "agedb"
    assert resolved["weight"] == 0.10


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


def test_ddim_ablation_cli_dry_run_does_not_load_models():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ablate_ddim_inversion",
            "--config",
            "configs/inference/ddim_inversion.yaml",
            "--image",
            "missing_input.png",
            "--local-spec",
            "configs/inference/local_spec.example.json",
            "--local-checkpoint",
            "missing_local.pt",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Models were not loaded" in result.stdout
