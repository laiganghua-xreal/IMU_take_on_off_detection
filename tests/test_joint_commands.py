import json
from pathlib import Path

import torch
import yaml

from egocharm.checkpoint import TrainingState, save_checkpoint
from egocharm.config import load_config
from egocharm.evaluate_joint import main as evaluate_main
from egocharm.joint_cache import write_joint_manifest, write_recording_cache
from egocharm.joint_dataset import JOINT_CLASS_NAMES
from egocharm.model import build_model
from egocharm.train_joint import main, validate_joint_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "ego_xreal_joint_9class.yaml"


def test_joint_config_builds_five_second_nine_class_model():
    config = load_config(CONFIG_PATH)
    validate_joint_config(config)
    model = build_model(config["model"])

    logits = model(torch.zeros(2, 5, 6, 200))

    assert logits.shape == (2, 9)
    assert tuple(config["model"]["class_names"]) == JOINT_CLASS_NAMES
    assert config["data"]["normalization"] == "none"
    assert config["data"]["candidate_step_seconds"] == 0.025


def test_joint_training_dry_run_reports_isolated_output(capsys):
    exit_code = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--run-name",
            "dry-run-test",
            "--dry-run",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["training_started"] is False
    assert report["class_names"] == list(JOINT_CLASS_NAMES)
    assert report["input_shape"] == [5, 6, 200]
    assert report["run_directory"].endswith(
        "artifacts/ego_xreal_joint_9class/runs/dry-run-test"
    )


def test_joint_evaluation_writes_metrics_and_confusion_matrices(tmp_path):
    base_config = load_config(CONFIG_PATH)
    ego_recordings = []
    xreal_recordings = []
    for class_index, class_name in enumerate(JOINT_CLASS_NAMES):
        recording = write_recording_cache(
            cache_root=tmp_path / "cache",
            recording_uid=f"test-{class_index}",
            source="xreal" if class_index >= 7 else "ego",
            label=class_name,
            split="test",
            values=torch.zeros(1000, 6).numpy(),
            segment_bounds=((0, 1000),),
            sampling_rate_hz=200,
            coordinate_frame="cpf-left-up-forward",
        )
        if class_index >= 7:
            xreal_recordings.append(recording)
        else:
            ego_recordings.append(recording)

    ego_manifest = tmp_path / "ego" / "manifest.json"
    xreal_manifest = tmp_path / "xreal" / "manifest.json"
    write_joint_manifest(ego_manifest, ego_recordings)
    write_joint_manifest(xreal_manifest, xreal_recordings)
    base_config["data"]["ego_manifest"] = str(ego_manifest)
    base_config["data"]["xreal_event_manifest"] = str(xreal_manifest)
    base_config["training"]["device"] = "cpu"
    base_config["training"]["batch_size"] = 9
    base_config["training"]["number_of_workers"] = 0
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base_config), encoding="utf-8")

    model = build_model(base_config["model"])
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        None,
        TrainingState(),
        {"class_names": list(JOINT_CLASS_NAMES)},
    )
    output_directory = tmp_path / "results"

    exit_code = evaluate_main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "test",
            "--output",
            str(output_directory),
        ]
    )

    assert exit_code == 0
    assert (output_directory / "metrics.json").is_file()
    assert (output_directory / "per-class.csv").is_file()
    assert (output_directory / "confusion-matrix.png").is_file()
    assert (output_directory / "confusion-matrix-normalized.png").is_file()
