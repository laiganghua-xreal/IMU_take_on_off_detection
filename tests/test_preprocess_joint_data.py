import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.preprocess_joint_data import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_counts_only_xreal_event_recordings(tmp_path, capsys):
    source_root = tmp_path / "xreal"
    directory_names = (
        "DEVICE_ON_USER_free_record_only_imu_001",
        "DEVICE_OFF_USER_free_record_only_imu_002",
        "DEVICE_NOWEAR_USER_free_record_only_imu_003",
    )
    for directory_name in directory_names:
        recording_directory = source_root / directory_name
        recording_directory.mkdir(parents=True)
        (recording_directory / "imu_0.csv").touch()

    config = {
        "data": {
            "ego_selection": str(tmp_path / "ego-selection.json"),
            "ego_dataset_root": str(tmp_path / "ego"),
            "ego_manifest": str(tmp_path / "ego-cache" / "manifest.json"),
            "xreal_event_manifest": str(
                tmp_path / "xreal-cache" / "manifest.json"
            ),
            "xreal_source_roots": [str(source_root)],
            "sampling_rate_hz": 200,
            "window_seconds": 5,
            "maximum_gap_seconds": 0.1,
            "minimum_segment_seconds": 5,
        },
        "model": {},
        "training": {},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    exit_code = main(["--config", str(config_path), "--dry-run"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["xreal_event_recordings"] == {
        "PUT_ON": 1,
        "TAKE_OFF": 1,
    }


def test_script_can_be_executed_by_file_path():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/preprocess_joint_data.py",
            "--config",
            "configs/ego_xreal_joint_9class.yaml",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["xreal_event_recordings"]["PUT_ON"] > 0
    assert report["xreal_event_recordings"]["TAKE_OFF"] > 0
