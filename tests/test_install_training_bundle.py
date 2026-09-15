from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from tools.install_training_bundle import build_config


def test_build_config_connects_stage07_artifacts(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    pose = bundle / "deployment" / "yolov8n-pose-finetuned.pt"
    classifier = bundle / "temporal_openvino" / "model.xml"
    checkpoint = bundle / "temporal_attention.pt"
    pose.parent.mkdir(parents=True)
    classifier.parent.mkdir(parents=True)
    pose.write_bytes(b"pose")
    classifier.write_text("model", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    (bundle / "deployment_manifest.json").write_text(
        json.dumps(
            {
                "pose_sha256": hashlib.sha256(b"pose").hexdigest(),
                "pose_model": "deployment/yolov8n-pose-finetuned.pt",
                "classifier_model": "temporal_openvino/model.xml",
                "classifier_checkpoint": "temporal_attention.pt",
                "sequence_length": 32,
                "sample_fps": 10.0,
                "fall_threshold": 0.61,
                "confirm_frames": 3,
                "cooldown_seconds": 15.0,
            }
        ),
        encoding="utf-8",
    )
    base = tmp_path / "default.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "detector": {"model": "baseline.pt"},
                "sequence": {"length": 32, "minimum_frames": 12},
                "analysis": {"classifier_model": None, "fall_threshold": 0.72},
            }
        ),
        encoding="utf-8",
    )

    config = build_config(bundle, base)

    assert config["detector"]["model"] == str(pose.resolve())
    assert config["analysis"]["classifier_model"] == str(classifier.resolve())
    assert config["analysis"]["fall_threshold"] == 0.61
    assert config["analysis"]["confirm_frames"] == 3
    assert config["sequence"]["minimum_frames"] == 32
    assert config["sequence"]["sample_fps"] == 10.0

    torch_config = build_config(bundle, base, classifier_backend="pytorch")
    assert torch_config["analysis"]["classifier_model"] == str(checkpoint.resolve())
