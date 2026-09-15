from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_config(bundle: Path, base_config: Path, classifier_backend: str = "openvino") -> dict:
    manifest_path = bundle / "deployment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pose_model = (bundle / manifest["pose_model"]).resolve()
    classifier_key = "classifier_model" if classifier_backend == "openvino" else "classifier_checkpoint"
    if classifier_key not in manifest:
        raise KeyError(f"deployment_manifest.json has no {classifier_key!r}")
    classifier_model = (bundle / manifest[classifier_key]).resolve()
    for model in (pose_model, classifier_model):
        if not model.is_file():
            raise FileNotFoundError(f"Missing deployment artifact: {model}")
    if sha256_file(pose_model) != manifest["pose_sha256"]:
        raise ValueError("Pose checkpoint does not match deployment_manifest.json")

    config = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    config.setdefault("detector", {})["model"] = str(pose_model)
    config.setdefault("sequence", {})["length"] = int(manifest["sequence_length"])
    config["sequence"]["minimum_frames"] = int(manifest["sequence_length"])
    config["sequence"]["sample_fps"] = float(manifest["sample_fps"])
    config.setdefault("analysis", {})["classifier_model"] = str(classifier_model)
    config["analysis"]["fall_threshold"] = float(manifest["fall_threshold"])
    config["analysis"]["confirm_frames"] = int(manifest["confirm_frames"])
    config["analysis"]["cooldown_seconds"] = float(manifest["cooldown_seconds"])
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a runtime config from a downloaded stage-07 bundle."
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--base-config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("configs/trained.yaml"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--classifier-backend",
        choices=("openvino", "pytorch"),
        default="openvino",
        help="Use optimized OpenVINO XML or the PyTorch checkpoint from stage 07.",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force")
    config = build_config(
        args.bundle.resolve(), args.base_config.resolve(), args.classifier_backend
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Saved runtime config: {args.output.resolve()}")


if __name__ == "__main__":
    main()
