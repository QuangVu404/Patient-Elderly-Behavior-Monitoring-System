"""Reusable stages for the Eldercare Kaggle training notebooks.

The module is intentionally self-contained. Upload the ``training`` directory as a
private Kaggle Dataset and attach it to every stage notebook.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

# Some Kaggle mirrors contain damaged MPEG-4 macroblocks. FFmpeg otherwise floods
# stderr even when OpenCV can recover at a later frame.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("TQDM_DISABLE", "1")

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ultralytics import YOLO

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
FEATURE_DIM = 89
KEYPOINT_COUNT = 17
COCO_FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
KAGGLE_FIVE = {"FallVision", "CAUCAFall", "URFD", "MCFD", "UCF101"}
KAGGLE_INPUT_SLUGS = {
    # This Kaggle package cites FallVision's Harvard Dataverse DOI.  The adapter
    # deliberately selects only its raw Fall/No_Fall tree, not the other bundled data.
    "FallVision": "fall-video-dataset",
    "CAUCAFall": "caucafall",
    "URFD": "ur-fall-detection-dataset",
    "MCFD": "multiple-cameras-fall-dataset",
    "UCF101": "ucf101-action-recognition",
}
KAGGLE_ROOT_ALIASES = {
    "fall-video-dataset": {"fall-video-dataset"},
    "caucafall": {"caucafall", "cauca-fall"},
    "ur-fall-detection-dataset": {"ur-fall-detection-dataset"},
    "multiple-cameras-fall-dataset": {"multiple-cameras-fall-dataset", "mcfd"},
    "ucf101-action-recognition": {"ucf101-action-recognition"},
}
DATASET_ALIASES = {
    "fall vision": "FallVision",
    "fallvision": "FallVision",
    "cauca fall": "CAUCAFall",
    "caucafall": "CAUCAFall",
    "ur fall detection dataset": "URFD",
    "urfd": "URFD",
    "multiple cameras fall dataset": "MCFD",
    "mcfd": "MCFD",
    "ucf101": "UCF101",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_one(root: Path, filename: str) -> Path:
    candidates = sorted(root.rglob(filename))
    if not candidates:
        raise FileNotFoundError(f"Cannot find {filename} below {root}")
    if len(candidates) > 1:
        preview = "\n- ".join(str(path) for path in candidates[:10])
        raise RuntimeError(
            f"Found multiple {filename} below {root}; remove stale/duplicate Inputs or pass "
            f"an explicit path:\n- {preview}"
        )
    return candidates[0]


def dataset_family(name: str) -> str:
    return str(name)


def _input_roots(root: Path, slug: str) -> list[Path]:
    """Return mount roots by Kaggle slug/title or by their known top-level layout."""
    normalized = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")
    aliases = KAGGLE_ROOT_ALIASES.get(normalized, {normalized})
    matches = []
    for child in _kaggle_dataset_roots(root):
        name = re.sub(r"[^a-z0-9]+", "-", child.name.lower()).strip("-")
        name_match = any(alias == name or alias in name for alias in aliases)
        if child.is_dir() and (name_match or _matches_kaggle_layout(child, normalized)):
            matches.append(child)
    return matches


def _kaggle_dataset_roots(root: Path) -> list[Path]:
    """Support both /input/<slug> and /input/datasets/<owner>/<slug> mounts."""
    if not root.is_dir():
        return []
    candidates = [path for path in root.iterdir() if path.is_dir()]
    datasets_root = root / "datasets"
    if datasets_root.is_dir():
        for owner_root in datasets_root.iterdir():
            if owner_root.is_dir():
                candidates.extend(path for path in owner_root.iterdir() if path.is_dir())
    return candidates


def _matches_kaggle_layout(path: Path, slug: str) -> bool:
    names = {
        re.sub(r"[^a-z0-9]+", "-", child.name.lower()).strip("-")
        for child in path.iterdir()
    }
    if slug == "fall-video-dataset":
        return {"fall", "no-fall"} <= names
    if slug == "caucafall":
        return sum(name.startswith("subject") for name in names) >= 5
    if slug == "ur-fall-detection-dataset":
        return any(name.startswith("ur-fall-detection-dataset") for name in names)
    if slug == "multiple-cameras-fall-dataset":
        return sum(bool(re.fullmatch(r"chute-?\d+", name)) for name in names) >= 20
    if slug == "ucf101-action-recognition":
        return {"train", "test", "val"} <= names
    return False


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_kaggle_inputs(
    input_root: str | Path = "/kaggle/input",
    required_families: set[str] | None = None,
) -> dict[str, Path]:
    """Fail before processing unless every requested dataset is a mounted Kaggle Input."""
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Kaggle Input root does not exist: {root}. Run this notebook on Kaggle."
        )
    required = KAGGLE_FIVE if required_families is None else set(required_families)
    unknown = required - KAGGLE_FIVE
    if unknown:
        raise ValueError(f"Unknown kaggle_rgb_5 families: {sorted(unknown)}")
    mounted: dict[str, Path] = {}
    missing: list[str] = []
    for family in sorted(required):
        slug = KAGGLE_INPUT_SLUGS[family]
        matches = _input_roots(root, slug)
        if not matches:
            missing.append(f"{family}: Add Input containing '{slug}'")
        else:
            mounted[family] = matches[0]
    if missing:
        details = "\n- ".join(missing)
        raise FileNotFoundError(
            "Required Kaggle datasets are not mounted under /kaggle/input.\n- "
            f"{details}\nUse Notebook > Add Input before starting the session; "
            "this pipeline intentionally does not download missing data."
        )
    return mounted


def resolve_mounted_file(input_root: str | Path, filename: str) -> Path:
    """Resolve a model/config file exclusively from the read-only Kaggle Input tree."""
    root = Path(input_root)
    requested = Path(filename)
    if requested.is_absolute() and requested.is_file() and _is_below(requested, root):
        return requested
    candidates = sorted(root.rglob(requested.name)) if root.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find mounted file '{requested.name}' below {root}. Add it as a Kaggle "
            "Input; automatic network download is disabled by this pipeline."
        )
    if len(candidates) > 1:
        preview = "\n- ".join(str(path) for path in candidates[:10])
        raise RuntimeError(
            f"Found multiple mounted files named '{requested.name}'. Pass an explicit path "
            f"below {root}:\n- {preview}"
        )
    return candidates[0]


def stage01_prepare_pose_baseline(
    input_root: str | Path = "/kaggle/input",
    output_dir: str | Path = "/kaggle/working/eldercare_pose",
    base_model: str = "yolov8n-pose.pt",
) -> Path:
    """Prepare a mounted pose checkpoint for smoke tests without pseudo-label training."""
    input_root, output_dir = Path(input_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = resolve_mounted_file(input_root, base_model)
    portable = output_dir / "yolov8n-pose-finetuned.pt"
    shutil.copy2(source, portable)
    model_sha = sha256_file(portable)
    (output_dir / "pose_model.sha256").write_text(model_sha + "\n", encoding="utf-8")
    metadata = {
        "model": portable.name,
        "sha256": model_sha,
        "base_model": source.name,
        "label_source": "mounted-baseline-no-finetune",
        "epochs": 0,
    }
    (output_dir / "pose_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return portable


def _row(dataset: str, path: Path, label: int, group: str, **extra: Any) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "path": str(path),
        "source_type": extra.pop("source_type", "video"),
        "label": label,
        "group": group,
        "annotation": extra.pop("annotation", ""),
        "segments": extra.pop("segments", ""),
        "subject": extra.pop("subject", ""),
        "activity": extra.pop("activity", ""),
        "trial": extra.pop("trial", ""),
        "camera": extra.pop("camera", ""),
        **extra,
    }


def discover_fallvision(root: Path) -> list[dict[str, Any]]:
    """Select the raw FallVision-style Fall/No_Fall trees from its Kaggle mirror."""
    rows = []
    for mounted in _input_roots(root, "fall-video-dataset"):
        for path in mounted.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            relative = path.relative_to(mounted)
            lowered = str(relative).lower()
            if any(marker in lowered for marker in ("masked", "landmark", "processed")):
                continue
            tokens = [re.sub(r"[^a-z0-9]+", "", part.lower()) for part in relative.parts[:-1]]
            negative = any(token in {"nofall", "nonfall", "notfall", "adl", "normal"} for token in tokens)
            positive = any(token in {"fall", "falls", "falling"} for token in tokens)
            if not (positive or negative):
                continue
            label = int(positive and not negative)
            subject_match = re.search(
                r"(?:subject|volunteer|person|actor)[-_ ]?(\d+)",
                str(relative),
                re.IGNORECASE,
            )
            subject = subject_match.group(1) if subject_match else ""
            group = f"fallvision_subject_{subject}" if subject else f"fallvision_{path.stem}"
            rows.append(_row("FallVision", path, label, group, subject=subject))
    return rows


def discover_caucafall(root: Path) -> list[dict[str, Any]]:
    """Use CAUCAFall PNG sequences directly, avoiding its AVI/audio codecs."""
    rows = []
    for mounted in _input_roots(root, "caucafall"):
        for directory in mounted.rglob("*"):
            if not directory.is_dir():
                continue
            images = [
                path for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not images:
                continue
            relative = directory.relative_to(mounted)
            evidence = f"{relative} {' '.join(path.stem for path in images[:3])}"
            normalized = re.sub(r"[^a-z0-9]+", "", evidence.lower())
            fall_markers = ("fall", "backwards", "forward", "lateralleft", "lateralright")
            if any(marker in normalized for marker in fall_markers):
                label = 1
            elif any(
                activity in normalized
                for activity in ("adl", "walk", "hop", "pickup", "sitdown", "kneel")
            ):
                label = 0
            else:
                continue
            subject_match = re.search(
                r"(?:subject|person|participant|s)[^0-9]*(\d+)",
                evidence,
                re.IGNORECASE,
            )
            subject = subject_match.group(1) if subject_match else ""
            group = f"caucafall_subject_{subject}" if subject else f"caucafall_{directory.name}"
            rows.append(
                _row(
                    "CAUCAFall", directory, label, group, source_type="images",
                    subject=subject, activity=directory.name
                )
            )
    return rows


def discover_urfd(root: Path) -> list[dict[str, Any]]:
    rows = []
    for mounted in _input_roots(root, "ur-fall-detection-dataset"):
        for directory in mounted.rglob("*"):
            if not directory.is_dir():
                continue
            match = re.match(
                r"(fall|adl)-(\d+)-cam(\d+)-rgb$", directory.name, re.IGNORECASE
            )
            if not match or not any(p.suffix.lower() in IMAGE_EXTENSIONS for p in directory.iterdir()):
                continue
            activity, sequence, camera = match.groups()
            rows.append(_row("URFD", directory, int(activity.lower() == "fall"), f"urfd_{activity.lower()}_{sequence}", source_type="images", trial=sequence, camera=camera))
    return rows


def discover_mcfd(root: Path) -> list[dict[str, Any]]:
    """Discover the canonical MCFD ``chuteXX/camY.avi`` multi-view layout."""
    rows = []
    for mounted in _input_roots(root, "multiple-cameras-fall-dataset"):
        for video in mounted.rglob("*"):
            if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            relative = video.relative_to(mounted)
            scenario_match = re.search(r"chute[-_ ]?(\d+)", str(relative), re.IGNORECASE)
            camera_match = re.search(r"cam(?:era)?[-_ ]?(\d+)", video.stem, re.IGNORECASE)
            if not scenario_match or not camera_match:
                continue
            scenario, camera = scenario_match.group(1), camera_match.group(1)
            # Per the original technical report, scenarios 01-22 contain a fall;
            # 23-24 contain confounding events only. All camera views of one
            # scenario remain in the same split to prevent cross-view leakage.
            label = int(int(scenario) <= 22)
            rows.append(
                _row(
                    "MCFD", video, label, f"mcfd_scenario_{int(scenario):02d}",
                    activity="fall_sequence" if label else "confounding",
                    trial=scenario, camera=camera, source=str(relative).replace("\\", "/"),
                )
            )
    return rows


def discover_ucf101(root: Path) -> list[dict[str, Any]]:
    rows = []
    seen_clips: set[str] = set()
    for mounted in _input_roots(root, "ucf101-action-recognition"):
        for video in mounted.rglob("*"):
            if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            match = re.search(r"_g(\d+)_c(\d+)$", video.stem, re.IGNORECASE)
            if not match or not video.stem.lower().startswith("v_"):
                continue
            canonical_id = video.name.lower()
            if canonical_id in seen_clips:
                continue
            seen_clips.add(canonical_id)
            group, clip = match.groups()
            rows.append(_row("UCF101", video, 0, f"ucf101_group_{group}", activity=video.parent.name, trial=clip))
    return rows


def discover_kaggle_five(root: Path) -> list[dict[str, Any]]:
    return (
        discover_fallvision(root)
        + discover_caucafall(root)
        + discover_urfd(root)
        + discover_mcfd(root)
        + discover_ucf101(root)
    )


def discover_user_manifests(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {"dataset", "path", "label", "group"}
    for csv_path in root.rglob("eldercare_manifest.csv"):
        frame = pd.read_csv(csv_path)
        if not required <= set(frame.columns):
            raise ValueError(f"{csv_path} is missing {sorted(required - set(frame.columns))}")
        for item in frame.to_dict("records"):
            source = Path(str(item["path"]))
            if not source.is_absolute():
                source = csv_path.parent / source
            if not source.exists():
                print(f"Skipping missing source: {source}")
                continue
            name = str(item["dataset"]).strip()
            name = DATASET_ALIASES.get(name.lower(), name)

            optional = {}
            for key, default in {
                "source_type": "video",
                "annotation": "",
                "segments": "",
                "subject": "",
                "activity": "",
                "trial": "",
                "camera": "",
            }.items():
                value = item.get(key, default)
                optional[key] = default if pd.isna(value) else value

            rows.append(
                {
                    **item,
                    "dataset": name,
                    "path": str(source),
                    **optional,
                }
            )
    return rows


def assign_group_splits(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    chunks = []
    for dataset_name, chunk in frame.groupby("dataset", sort=True):
        chunk = chunk.copy().reset_index(drop=True)
        group_labels = (
            chunk.assign(_group=chunk["group"].astype(str))
            .groupby("_group")["label"]
            .agg(lambda values: tuple(sorted({int(value) for value in values})))
        )
        groups = sorted(group_labels.index)
        if len(groups) < 3:
            print(f"Dropping {dataset_name}: fewer than 3 independent groups")
            continue
        salt = int(hashlib.sha1(dataset_name.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed + salt)
        buckets: dict[tuple[int, ...], list[str]] = defaultdict(list)
        for group in groups:
            buckets[group_labels[group]].append(group)
        val_groups: set[str] = set()
        test_groups: set[str] = set()
        for bucket in buckets.values():
            rng.shuffle(bucket)
            if len(bucket) == 1:
                continue
            if len(bucket) == 2:
                # Tiny strata such as MCFD's two non-fall scenarios cannot cover
                # all three splits. Keep one for domain learning and one untouched
                # for test; validation still receives negatives from other families.
                test_groups.add(bucket[0])
                continue
            held_count = min(max(2, round(len(bucket) * 0.20)), len(bucket) - 1)
            val_count = max(1, held_count // 2)
            val_groups.update(bucket[:val_count])
            test_groups.update(bucket[val_count:held_count])
        chunk["split"] = "train"
        chunk.loc[chunk["group"].astype(str).isin(val_groups), "split"] = "validation"
        chunk.loc[chunk["group"].astype(str).isin(test_groups), "split"] = "test"
        chunks.append(chunk)
    if not chunks:
        raise ValueError("No dataset has enough groups for train/validation/test")
    result = pd.concat(chunks, ignore_index=True)
    overlap = result.groupby(["dataset", "group"])["split"].nunique().max()
    if overlap != 1:
        raise AssertionError("Group leakage detected")
    for split in ("train", "validation", "test"):
        labels = set(result.loc[result["split"] == split, "label"].astype(int))
        if not {0, 1} <= labels:
            raise ValueError(f"Unable to create a class-balanced {split} group split")
    return result


def _stratified_source_limit(rows: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Cap a cache shard while retaining every available split/label stratum."""
    if len(rows) <= limit:
        return rows
    stratum_count = rows.groupby(["split", "label"]).ngroups
    if limit < stratum_count:
        raise ValueError("max_sources is too small to retain every split/label stratum")
    rng = random.Random(seed)
    buckets: list[list[int]] = []
    for _, bucket in rows.groupby(["split", "label"], sort=True):
        indices = list(bucket.index)
        rng.shuffle(indices)
        buckets.append(indices)
    selected: list[int] = []
    minimum_per_stratum = max(1, min(3, limit // stratum_count))
    for _ in range(minimum_per_stratum):
        for indices in buckets:
            if indices:
                selected.append(indices.pop())
    remaining = [index for indices in buckets for index in indices]
    rng.shuffle(remaining)
    selected.extend(remaining[: limit - len(selected)])
    return rows.loc[selected].sort_values(["split", "label", "group"])


def validate_canonical_inventory(manifest: pd.DataFrame) -> list[str]:
    """Reject incomplete mirrors; report harmless extras/repackaging as warnings."""
    canonical_minimums = {
        "FallVision": 5866,  # raw half of 11,732 raw + landmarked videos
        "CAUCAFall": 100,
        "MCFD": 192,  # 24 scenarios x 8 cameras
        "UCF101": 13320,
    }
    errors = []
    warnings = []
    for family, expected in canonical_minimums.items():
        actual = int((manifest["dataset"] == family).sum())
        if actual < expected:
            errors.append(
                f"{family}: expected at least {expected:,} canonical sources, "
                f"discovered {actual:,}"
            )
        elif actual > expected:
            warnings.append(
                f"{family}: canonical count is {expected:,}, mirror exposed {actual:,}; "
                "extra files are uploader additions and are not proof of extra independent data"
            )
    urfd = manifest[manifest["dataset"] == "URFD"]
    urfd_sequences = urfd["group"].nunique()
    if urfd_sequences != 70:
        errors.append(f"URFD: expected 70 independent sequences, discovered {urfd_sequences}")
    expected_groups = {"CAUCAFall": 10, "MCFD": 24, "UCF101": 25}
    for family, expected in expected_groups.items():
        actual = int(manifest.loc[manifest["dataset"] == family, "group"].nunique())
        if actual != expected:
            errors.append(
                f"{family}: expected {expected} canonical groups, discovered {actual}; "
                "refusing a split that may leak subjects/scenarios"
            )
    if errors:
        raise ValueError(
            "Dataset provenance/layout audit failed; a matching Kaggle title is not enough:\n- "
            + "\n- ".join(errors)
            + "\nCompare the mounted files with training/datasets_catalog.md."
        )
    for warning in warnings:
        print(f"[PROVENANCE WARNING] {warning}")
    return warnings


def stage00_build_manifest(
    input_root: str | Path = "/kaggle/input",
    output_dir: str | Path = "/kaggle/working/eldercare_manifest",
    seed: int = 42,
    require_all_five: bool = True,
) -> Path:
    input_root, output_dir = Path(input_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if require_all_five:
        validate_kaggle_inputs(input_root)
    rows = discover_user_manifests(input_root) + discover_kaggle_five(input_root)
    if not rows:
        raise FileNotFoundError(
            "No kaggle_rgb_5 source was discovered. Attach the five documented Kaggle inputs "
            "or provide eldercare_manifest.csv."
        )
    manifest = pd.DataFrame(rows).drop_duplicates(subset=["path"], keep="first")
    selected = manifest["dataset"].isin(KAGGLE_FIVE)
    manifest = manifest[selected].reset_index(drop=True)
    if manifest.empty:
        raise ValueError("Inputs contain no dataset from kaggle_rgb_5")
    outside_input = [path for path in manifest["path"] if not _is_below(Path(path), input_root)]
    if outside_input:
        raise ValueError(
            "Every raw source must be mounted below the Kaggle Input root; found: "
            f"{outside_input[0]}"
        )
    found_families = set(manifest["dataset"].map(dataset_family))
    expected_families = KAGGLE_FIVE
    missing_families = sorted(expected_families - found_families)
    if require_all_five and missing_families:
        raise ValueError(f"Full-data run is missing dataset families: {missing_families}")
    provenance_warnings = validate_canonical_inventory(manifest) if require_all_five else []
    manifest = assign_group_splits(manifest, seed)
    manifest["source_id"] = [
        hashlib.sha1(f"{dataset}:{path}".encode()).hexdigest()
        for dataset, path in zip(manifest["dataset"], manifest["path"])
    ]
    master_path = output_dir / "master_manifest.csv"
    manifest.to_csv(master_path, index=False)
    report = (
        manifest.groupby(["dataset", "split"])
        .agg(sources=("source_id", "nunique"), groups=("group", "nunique"))
        .reset_index()
    )
    (output_dir / "split_report.json").write_text(
        report.to_json(orient="records", indent=2), encoding="utf-8"
    )
    (output_dir / "provenance_audit.json").write_text(
        json.dumps(
            {"status": "passed_with_warnings" if provenance_warnings else "passed",
             "warnings": provenance_warnings},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report.to_string(index=False))
    print(f"Saved {master_path}")
    return master_path


def iter_source_frames(
    path: Path,
    source_type: str,
    dataset: str = "",
    target_fps: float = 10.0,
):
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")

    def sample_due(index: int, next_index: float, source_fps: float) -> tuple[bool, float]:
        """Select nearest source frames while preserving the requested average sample rate."""
        if index < round(next_index):
            return False, next_index
        increment = source_fps / min(target_fps, source_fps)
        while round(next_index) <= index:
            next_index += increment
        return True, next_index

    if source_type == "video":
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            print(f"[VIDEO SKIP] Cannot open {path}")
            return
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        index = 0
        next_sample_index = 0.0
        decode_failures = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    if frame_count <= 0 or index >= frame_count - 1:
                        break
                    decode_failures += 1
                    if decode_failures > 32:
                        print(f"[VIDEO TRUNCATED] Too many damaged frames in {path}")
                        break
                    index += 1
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    continue
                due, next_sample_index = sample_due(
                    index, next_sample_index, source_fps
                )
                if due:
                    yield index, index / source_fps, source_fps, frame
                index += 1
        finally:
            capture.release()
        return

    assumed_fps = 23.0 if dataset == "CAUCAFall" else 25.0
    next_sample_index = 0.0
    if source_type == "images":
        images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        for index, image_path in enumerate(images):
            due, next_sample_index = sample_due(index, next_sample_index, assumed_fps)
            if due:
                frame = cv2.imread(str(image_path))
                if frame is not None:
                    yield index, index / assumed_fps, assumed_fps, frame
        return
    if source_type != "zip":
        raise ValueError(f"Unsupported source_type={source_type}")
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if Path(name).suffix.lower() in IMAGE_EXTENSIONS)
        for index, name in enumerate(names):
            due, next_sample_index = sample_due(index, next_sample_index, assumed_fps)
            if not due:
                continue
            frame = cv2.imdecode(np.frombuffer(archive.read(name), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield index, index / assumed_fps, assumed_fps, frame


def yolo_pose_line(
    result: Any,
    frame_shape: tuple[int, ...],
    min_keypoint_confidence: float,
    min_visible_keypoints: int = 8,
) -> str | None:
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xywh.detach().cpu().numpy()
    points = result.keypoints.data.detach().cpu().numpy()
    selected = int(np.argmax(boxes[:, 2] * boxes[:, 3]))
    height, width = frame_shape[:2]
    box = boxes[selected] / np.array([width, height, width, height], dtype=np.float32)
    keypoints = points[selected].copy()
    if keypoints.shape != (17, 3):
        return None
    visible = keypoints[:, 2] >= min_keypoint_confidence
    if int(visible.sum()) < min_visible_keypoints:
        return None
    keypoints[:, 0] = np.clip(keypoints[:, 0] / width, 0.0, 1.0)
    keypoints[:, 1] = np.clip(keypoints[:, 1] / height, 0.0, 1.0)
    keypoints[~visible, :2] = 0.0
    keypoints[:, 2] = np.where(visible, 2.0, 0.0)
    values = np.concatenate([[0.0], np.clip(box, 0.0, 1.0), keypoints.reshape(-1)])
    return " ".join(f"{float(value):.6f}" for value in values)


def build_pseudo_pose_dataset(
    teacher: YOLO,
    manifest: pd.DataFrame,
    root: Path,
    frames_per_family: int,
    pseudo_confidence: float,
    sample_fps: float,
    image_size: int,
    seed: int,
    device: str,
) -> Path:
    if root.exists():
        shutil.rmtree(root)
    for split_name in ("train", "val"):
        (root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split_name).mkdir(parents=True, exist_ok=True)
    rows = manifest[manifest["split"] == "train"].copy()
    pose_val_groups = pose_validation_group_keys(rows, seed)
    train_cap = round(frames_per_family * 0.90)
    split_caps = {"train": train_cap, "val": frames_per_family - train_cap}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    saved = {"train": 0, "val": 0}
    for row in rows.sample(frac=1.0, random_state=seed).itertuples(index=False):
        family = dataset_family(row.dataset)
        split_name = "val" if f"{row.dataset}::{row.group}" in pose_val_groups else "train"
        count_key = (family, split_name)
        if counts[count_key] >= split_caps[split_name]:
            continue
        for frame_index, _, _, frame in iter_source_frames(
            Path(row.path), row.source_type, row.dataset, target_fps=sample_fps
        ):
            if counts[count_key] >= split_caps[split_name]:
                break
            result = teacher.predict(
                frame, imgsz=image_size, conf=pseudo_confidence, iou=0.5, classes=[0],
                device=device, verbose=False
            )[0]
            label = yolo_pose_line(result, frame.shape, 0.35, min_visible_keypoints=10)
            if label is None:
                continue
            stem = hashlib.sha1(f"{row.path}:{frame_index}".encode()).hexdigest()[:20]
            image_path = root / "images" / split_name / f"{stem}.jpg"
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                continue
            (root / "labels" / split_name / f"{stem}.txt").write_text(label + "\n", encoding="utf-8")
            counts[count_key] += 1
            saved[split_name] += 1
    missing_strata = []
    for family in sorted(KAGGLE_FIVE):
        for split_name, cap in split_caps.items():
            required = min(20, cap)
            actual = counts[(family, split_name)]
            if actual < required:
                missing_strata.append(
                    f"{family}/{split_name}: {actual} labels, require at least {required}"
                )
    if missing_strata:
        raise RuntimeError(
            "Pseudo pose coverage is insufficient for one or more family/split strata:\n- "
            + "\n- ".join(missing_strata)
        )
    data_yaml = root / "pose_dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "kpt_shape": [17, 3],
                "flip_idx": COCO_FLIP_INDEX,
                "names": {0: "person"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"Pseudo pose labels: {saved}; family/split={dict(counts)}")
    return data_yaml


def pose_validation_group_keys(rows: pd.DataFrame, seed: int) -> set[str]:
    """Hold out groups within every family for representative pseudo-pose validation."""
    selected: set[str] = set()
    for family, chunk in rows.groupby("dataset", sort=True):
        groups = sorted(chunk["group"].astype(str).unique())
        if len(groups) < 2:
            print(f"[POSE VAL WARNING] {family} has fewer than two training groups")
            continue
        salt = int(hashlib.sha1(str(family).encode("utf-8")).hexdigest()[:8], 16)
        random.Random(seed + 991 + salt).shuffle(groups)
        count = min(max(1, round(len(groups) * 0.10)), len(groups) - 1)
        selected.update(f"{family}::{group}" for group in groups[:count])
    return selected


def resolved_pose_splits(data_yaml: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if config.get("kpt_shape") != [17, 3]:
        raise ValueError(f"{data_yaml} must use kpt_shape: [17, 3]")
    base = Path(config.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = data_yaml.parent / base

    def resolve(value: str | list[str]) -> list[str]:
        values = value if isinstance(value, list) else [value]
        return [str(Path(item) if Path(item).is_absolute() else base / item) for item in values]

    return resolve(config["train"]), resolve(config["val"]), config


def combine_pose_datasets(gold_yaml: Path, pseudo_yaml: Path, output: Path, gold_repeat: int) -> Path:
    gold_train, gold_val, gold_config = resolved_pose_splits(gold_yaml)
    pseudo_train, _, _ = resolved_pose_splits(pseudo_yaml)
    output.write_text(
        yaml.safe_dump(
            {
                "train": gold_train * gold_repeat + pseudo_train,
                # Once real labels exist, model selection must not be dominated by
                # teacher-generated labels.
                "val": gold_val,
                "kpt_shape": [17, 3],
                "flip_idx": COCO_FLIP_INDEX,
                "names": gold_config.get("names", {0: "person"}),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return output


def _metric_values(metrics: Any) -> dict[str, float]:
    return {str(key): float(value) for key, value in metrics.results_dict.items()}


def _training_fitness(metrics: Any) -> float:
    """Extract Ultralytics validation fitness without depending on one release's key names."""
    fitness = getattr(metrics, "fitness", None)
    if fitness is not None:
        return float(fitness)
    values = getattr(metrics, "results_dict", {})
    for key in ("fitness", "metrics/mAP50-95(P)", "metrics/mAP50(P)"):
        if key in values:
            return float(values[key])
    raise ValueError("Ultralytics training result does not expose pose fitness")


def stage01_finetune_pose(
    input_root: str | Path = "/kaggle/input",
    output_dir: str | Path = "/kaggle/working/eldercare_pose",
    manifest_path: str | Path | None = None,
    base_model: str = "yolov8n-pose.pt",
    mode: str = "auto",
    frames_per_family: int = 500,
    pseudo_confidence: float = 0.70,
    pseudo_fps: float = 1.0,
    gold_repeat: int = 2,
    epochs: int = 15,
    unfreeze_epochs: int = 0,
    image_size: int = 416,
    batch_size: int = 8,
    workers: int = 0,
    plots: bool = False,
    export_openvino: bool = False,
    seed: int = 42,
) -> Path:
    if mode not in {"auto", "mixed", "gold", "pseudo"}:
        raise ValueError("mode must be auto, mixed, gold, or pseudo")
    seed_everything(seed)
    input_root, output_dir = Path(input_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_source = (
        Path(manifest_path) if manifest_path is not None else find_one(input_root, "master_manifest.csv")
    )
    if not manifest_source.is_file():
        raise FileNotFoundError(f"Master manifest does not exist: {manifest_source}")
    manifest = pd.read_csv(manifest_source).fillna("")
    base_model_path = resolve_mounted_file(input_root, base_model)
    teacher = YOLO(str(base_model_path))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gold_candidates = [p for p in sorted(input_root.rglob("pose_dataset.yaml")) if "pseudo" not in str(p)]
    if len(gold_candidates) > 1:
        preview = "\n- ".join(str(path) for path in gold_candidates[:10])
        raise RuntimeError(
            "Found multiple gold pose_dataset.yaml files; attach only the intended gold "
            f"Dataset:\n- {preview}"
        )
    gold_yaml = gold_candidates[0] if gold_candidates else None
    if mode in {"gold", "mixed"} and gold_yaml is None:
        raise FileNotFoundError(f"mode={mode} requires pose_dataset.yaml")
    pseudo_yaml = None
    if mode in {"pseudo", "mixed"} or (mode == "auto"):
        validate_kaggle_inputs(input_root)
        missing_sources = [path for path in manifest["path"] if not Path(path).exists()]
        if missing_sources:
            raise FileNotFoundError(
                "Stage 01 needs all five raw Inputs used by the manifest. Missing source: "
                f"{missing_sources[0]}"
            )
        pseudo_yaml = build_pseudo_pose_dataset(
            teacher, manifest, output_dir / "pseudo_dataset", frames_per_family,
            pseudo_confidence, pseudo_fps, image_size, seed, device
        )
    if mode == "gold":
        training_yaml, label_source = gold_yaml, "gold"
    elif mode == "mixed" or (mode == "auto" and gold_yaml is not None):
        training_yaml = combine_pose_datasets(
            gold_yaml, pseudo_yaml, output_dir / "mixed_pose_dataset.yaml", gold_repeat
        )
        label_source = "gold+pseudo"
    else:
        training_yaml, label_source = pseudo_yaml, "pseudo"

    gold_metrics: dict[str, Any] = {}
    train_device: int | str = 0 if torch.cuda.is_available() else "cpu"
    if gold_yaml is not None and "gold" in label_source:
        gold_metrics["before"] = _metric_values(
            teacher.val(
                data=str(gold_yaml), split="val", imgsz=image_size,
                device=train_device, verbose=False
            )
        )
    warmup_metrics = teacher.train(
        data=str(training_yaml), epochs=epochs, imgsz=image_size, batch=batch_size,
        device=train_device, workers=workers, seed=seed, project=str(output_dir),
        name="yolov8n_pose_warmup",
        exist_ok=True, pretrained=True, freeze=10, patience=5, close_mosaic=5, plots=plots,
        optimizer="AdamW", lr0=2e-3, lrf=0.1, weight_decay=5e-4,
        degrees=8.0, translate=0.10, scale=0.35, fliplr=0.5, mosaic=0.25
    )
    best = output_dir / "yolov8n_pose_warmup" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"YOLO training did not create {best}")
    warmup_best = best
    warmup_fitness = _training_fitness(warmup_metrics)
    unfreeze_fitness: float | None = None
    selected_phase = "warmup"
    if unfreeze_epochs > 0:
        student = YOLO(str(best))
        unfreeze_metrics = student.train(
            data=str(training_yaml), epochs=unfreeze_epochs, imgsz=image_size,
            batch=batch_size, device=train_device, workers=workers, seed=seed + 1,
            project=str(output_dir), name="yolov8n_pose_unfrozen", exist_ok=True,
            pretrained=True, freeze=0, patience=5, close_mosaic=3, plots=plots,
            optimizer="AdamW", lr0=3e-4, lrf=0.05, weight_decay=5e-4,
            degrees=5.0, translate=0.08, scale=0.25, fliplr=0.5, mosaic=0.10
        )
        deep_best = output_dir / "yolov8n_pose_unfrozen" / "weights" / "best.pt"
        if not deep_best.exists():
            raise FileNotFoundError(f"YOLO unfreeze phase did not create {deep_best}")
        unfreeze_fitness = _training_fitness(unfreeze_metrics)
        if unfreeze_fitness >= warmup_fitness:
            best = deep_best
            selected_phase = "unfrozen"
        else:
            best = warmup_best
            print(
                "[POSE ROLLBACK] Unfreeze fitness "
                f"{unfreeze_fitness:.6f} < warm-up {warmup_fitness:.6f}; keeping warm-up best"
            )
        del student
    portable = output_dir / "yolov8n-pose-finetuned.pt"
    shutil.copy2(best, portable)
    tuned = YOLO(str(portable))
    if gold_yaml is not None and "gold" in label_source:
        gold_metrics["after"] = _metric_values(
            tuned.val(
                data=str(gold_yaml), split="val", imgsz=image_size,
                device=train_device, verbose=False
            )
        )
        (output_dir / "pose_gold_metrics.json").write_text(
            json.dumps(gold_metrics, indent=2), encoding="utf-8"
        )
    model_sha = sha256_file(portable)
    (output_dir / "pose_model.sha256").write_text(model_sha + "\n", encoding="utf-8")
    metadata = {
        "model": portable.name,
        "sha256": model_sha,
        "base_model": base_model,
        "label_source": label_source,
        "epochs": epochs + unfreeze_epochs,
        "strategy": {
            "warmup_epochs": epochs,
            "warmup_freeze": 10,
            "warmup_lr": 2e-3,
            "unfreeze_epochs": unfreeze_epochs,
            "unfreeze_layers": "all",
            "unfreeze_lr": 3e-4 if unfreeze_epochs else None,
            "pseudo_keypoint_confidence": 0.35,
            "pseudo_min_visible_keypoints": 10,
            "warmup_fitness": warmup_fitness,
            "unfreeze_fitness": unfreeze_fitness,
            "selected_phase": selected_phase,
        },
        "seed": seed,
    }
    (output_dir / "pose_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if export_openvino:
        tuned.export(format="openvino", imgsz=416, half=True, device="cpu")
    print(json.dumps(metadata, indent=2))
    del teacher, tuned
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return portable


def pose_to_feature(box: np.ndarray, keypoints: np.ndarray, frame_shape: tuple[int, int], previous=None) -> np.ndarray:
    box = np.asarray(box, dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32).copy()
    width = max(float(box[2] - box[0]), 1.0)
    height = max(float(box[3] - box[1]), 1.0)
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)
    visible = keypoints[:, 2] >= 0.25
    normalized = (keypoints[:, :2] - center) / height
    normalized[~visible] = 0.0
    velocity = np.zeros((KEYPOINT_COUNT, 2), dtype=np.float32)
    if previous is not None:
        previous_box, previous_keypoints = previous
        previous_height = max(float(previous_box[3] - previous_box[1]), 1.0)
        both_visible = visible & (previous_keypoints[:, 2] >= 0.25)
        velocity[both_visible] = (
            keypoints[both_visible, :2] - previous_keypoints[both_visible, :2]
        ) / previous_height
    frame_height, frame_width = frame_shape
    geometry = np.array(
        [width / height, center[0] / frame_width, center[1] / frame_height,
         (width * height) / (frame_width * frame_height)],
        dtype=np.float32,
    )
    return np.concatenate(
        [normalized.reshape(-1), keypoints[:, 2], velocity.reshape(-1), geometry]
    ).astype(np.float32)


def frame_label(timestamp: float, clip_label: int, segments_json: str) -> int:
    if segments_json:
        segments = json.loads(segments_json)
        active = [int(label) for label, start, end in segments if float(start) <= timestamp <= float(end)]
        if not active:
            return -1
        return int(any(label in {1, 2} for label in active))
    return clip_label if clip_label in {0, 1} else -1


def caucafall_frame_labels(path: Path) -> dict[int, int]:
    """Read CAUCAFall's per-image class: 0=nofall, 1=fall."""
    labels: dict[int, int] = {}
    images = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)
    for index, image_path in enumerate(images):
        annotation = image_path.with_suffix(".txt")
        if not annotation.is_file():
            continue
        match = re.match(r"\s*([01])(?:\s|$)", annotation.read_text(errors="ignore"))
        if match:
            labels[index] = int(match.group(1))
    return labels


def stage_cache_dataset(
    family: str,
    input_root: str | Path = "/kaggle/input",
    output_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    pose_path: str | Path | None = None,
    pose_sha_path: str | Path | None = None,
    sample_fps: float = 10.0,
    image_size: int = 416,
    shard_index: int = 0,
    shard_count: int = 1,
    max_sources: int | None = None,
    minimum_cached_frames: int = 32,
    minimum_detection_coverage: float = 0.10,
    seed: int = 42,
) -> Path:
    if minimum_cached_frames < 1:
        raise ValueError("minimum_cached_frames must be positive")
    if not 0.0 <= minimum_detection_coverage <= 1.0:
        raise ValueError("minimum_detection_coverage must be between 0 and 1")
    seed_everything(seed)
    input_root = Path(input_root)
    validate_kaggle_inputs(input_root, {family})
    output_dir = Path(output_dir or f"/kaggle/working/eldercare_cache_{family.lower()}")
    cache_dir = output_dir / "pose_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_source = (
        Path(manifest_path) if manifest_path is not None else find_one(input_root, "master_manifest.csv")
    )
    if not manifest_source.is_file():
        raise FileNotFoundError(f"Master manifest does not exist: {manifest_source}")
    manifest = pd.read_csv(manifest_source).fillna("")
    rows = manifest[manifest["dataset"].map(dataset_family) == family].copy()
    if rows.empty:
        raise ValueError(f"Master manifest contains no family={family}")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard_index < shard_count")
    shard_mask = rows.apply(
        lambda row: int(hashlib.sha1(f"{row.dataset}:{row.group}".encode()).hexdigest(), 16)
        % shard_count
        == shard_index,
        axis=1,
    )
    rows = rows[shard_mask].sort_values(["dataset", "group", "path"]).reset_index(drop=True)
    if max_sources is not None and len(rows) > max_sources:
        rows = _stratified_source_limit(rows, max_sources, seed)
    pose_path = (
        Path(pose_path)
        if pose_path is not None
        else find_one(input_root, "yolov8n-pose-finetuned.pt")
    )
    if not pose_path.is_file():
        raise FileNotFoundError(f"Fine-tuned pose checkpoint does not exist: {pose_path}")
    pose_sha = sha256_file(pose_path)
    expected_sha_path = (
        Path(pose_sha_path)
        if pose_sha_path is not None
        else find_one(input_root, "pose_model.sha256")
    )
    if not expected_sha_path.is_file():
        raise FileNotFoundError(f"Pose SHA file does not exist: {expected_sha_path}")
    expected_sha = expected_sha_path.read_text(encoding="utf-8").strip()
    if pose_sha != expected_sha:
        raise AssertionError("Fine-tuned pose checkpoint does not match pose_model.sha256")
    pose_model = YOLO(str(pose_path))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_rows = []
    skipped_rows = []
    for number, row in enumerate(rows.itertuples(index=False), start=1):
        source = Path(row.path)
        if not source.exists():
            raise FileNotFoundError(f"Raw source is not attached at {source}")
        cache_path = cache_dir / f"{row.source_id}.npz"
        features, indices, timestamps, labels = [], [], [], []
        previous = None
        dense_labels = caucafall_frame_labels(source) if row.dataset == "CAUCAFall" else {}
        seen = 0
        source_fps = 25.0
        for index, timestamp, source_fps, frame in iter_source_frames(
            source, row.source_type, row.dataset, target_fps=sample_fps
        ):
            seen += 1
            result = pose_model.predict(
                frame, imgsz=image_size, conf=0.25, iou=0.5, classes=[0],
                device=device, verbose=False
            )[0]
            if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
                previous = None
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            points = result.keypoints.data.detach().cpu().numpy()
            selected = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
            box, keypoints = boxes[selected], points[selected]
            features.append(pose_to_feature(box, keypoints, frame.shape[:2], previous))
            indices.append(index)
            timestamps.append(timestamp)
            labels.append(
                dense_labels.get(index, frame_label(timestamp, int(row.label), str(row.segments)))
            )
            previous = (box.copy(), keypoints.copy())
        detection_coverage = len(features) / max(seen, 1)
        quality_reasons = []
        if len(features) < minimum_cached_frames:
            quality_reasons.append(
                f"detected_frames<{minimum_cached_frames}"
            )
        if detection_coverage < minimum_detection_coverage:
            quality_reasons.append(
                f"detection_coverage<{minimum_detection_coverage:.3f}"
            )
        if source_fps + 1e-6 < sample_fps:
            quality_reasons.append(f"source_fps<{sample_fps:.3f}")
        if features and not np.isfinite(np.asarray(features)).all():
            quality_reasons.append("non_finite_features")
        if quality_reasons:
            skipped_rows.append(
                {
                    "dataset": row.dataset,
                    "source_id": row.source_id,
                    "path": str(source),
                    "split": row.split,
                    "label": row.label,
                    "seen_frames": seen,
                    "detected_frames": len(features),
                    "detection_coverage": detection_coverage,
                    "reason": ";".join(quality_reasons),
                }
            )
            cache_path.unlink(missing_ok=True)
            print(
                f"[CACHE QUALITY SKIP] {source}: {quality_reasons}; "
                f"detected={len(features)}/{seen}"
            )
            continue
        np.savez_compressed(
            cache_path,
            features=np.asarray(features, dtype=np.float32),
            frame_indices=np.asarray(indices, dtype=np.int32),
            timestamps=np.asarray(timestamps, dtype=np.float64),
            frame_labels=np.asarray(labels, dtype=np.int8),
            source_fps=float(source_fps),
        )
        item = row._asdict()
        item.update(
            {
                "cache_relpath": str(cache_path.relative_to(output_dir)).replace("\\", "/"),
                "pose_sha256": pose_sha,
                "feature_dim": FEATURE_DIM,
                "sample_fps": sample_fps,
                "seen_frames": seen,
                "detected_frames": len(features),
                "detection_coverage": detection_coverage,
                "quality_status": "passed",
            }
        )
        output_rows.append(item)
        if number % 10 == 0 or number == len(rows):
            print(f"{family} shard {shard_index + 1}/{shard_count}: {number}/{len(rows)}")
    shard_manifest = output_dir / "shard_manifest.csv"
    if not output_rows:
        raise RuntimeError(f"No {family} source passed cache quality checks")
    pd.DataFrame(output_rows).to_csv(shard_manifest, index=False)
    quality_report = output_dir / "cache_quality_skips.csv"
    pd.DataFrame(
        skipped_rows,
        columns=[
            "dataset", "source_id", "path", "split", "label", "seen_frames",
            "detected_frames", "detection_coverage", "reason",
        ],
    ).to_csv(quality_report, index=False)
    metadata = {
        "family": family,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "sources": len(output_rows),
        "skipped_sources": len(skipped_rows),
        "minimum_cached_frames": minimum_cached_frames,
        "minimum_detection_coverage": minimum_detection_coverage,
        "pose_sha256": pose_sha,
        "feature_dim": FEATURE_DIM,
        "sample_fps": sample_fps,
        "image_size": image_size,
    }
    (output_dir / "shard_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    del pose_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return shard_manifest


class TemporalAttention1D(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 64, classes: int = 2):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(0.15),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 1), nn.GELU(),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(0.15), nn.Linear(hidden_dim, classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.projection(features)
        encoded = encoded + self.temporal(encoded.transpose(1, 2)).transpose(1, 2)
        weights = torch.softmax(self.attention(encoded), dim=1)
        return self.classifier(torch.sum(encoded * weights, dim=1))


class MultiScaleTemporalGRU(nn.Module):
    """Project-owned multi-scale temporal encoder with bidirectional context."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 96, classes: int = 2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    hidden_dim, hidden_dim, 3, padding=dilation,
                    dilation=dilation, groups=hidden_dim
                ),
                nn.Conv1d(hidden_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(0.20),
            )
            for dilation in (1, 2, 4)
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim, 1), nn.GELU()
        )
        self.recurrent = nn.GRU(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4), nn.Dropout(0.20),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.projection(features)
        channels = encoded.transpose(1, 2)
        multi_scale = self.fusion(
            torch.cat([branch(channels) for branch in self.branches], dim=1)
        )
        recurrent, _ = self.recurrent(encoded + multi_scale.transpose(1, 2))
        weights = torch.softmax(self.attention(recurrent), dim=1)
        attended = torch.sum(recurrent * weights, dim=1)
        maximum = torch.amax(recurrent, dim=1)
        return self.classifier(torch.cat([attended, maximum], dim=1))


class WindowDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.features, self.labels, self.augment = features, labels, augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        value = self.features[index].copy()
        if self.augment and random.random() < 0.5:
            value[:, 0:34:2] *= -1
            value[:, 51:85:2] *= -1
            value[:, -3] = 1.0 - value[:, -3]
        if self.augment:
            value += np.random.normal(0, 0.003, value.shape).astype(np.float32)
            if random.random() < 0.35:
                start = random.randrange(max(1, len(value) - 3))
                value[start : start + random.randint(1, 3)] = 0.0
            if random.random() < 0.35:
                value *= (np.random.random(value.shape) > 0.02).astype(np.float32)
        return torch.from_numpy(value), torch.tensor(self.labels[index], dtype=torch.long)


def metric_dict(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    accuracy = (true_positive + true_negative) / max(len(labels), 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
    specificity = true_negative / max(true_negative + false_positive, 1)
    false_positive_rate = false_positive / max(true_negative + false_positive, 1)
    classes = set(np.unique(labels))
    balanced_accuracy = (recall + specificity) / 2 if classes == {0, 1} else (
        recall if classes == {1} else specificity
    )
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "samples": float(len(labels)),
        "positive_samples": float(np.sum(labels == 1)),
        "negative_samples": float(np.sum(labels == 0)),
    }


def average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute binary average precision without adding a scikit-learn dependency."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return 0.0
    order = np.argsort(-probabilities, kind="stable")
    ranked_labels = labels[order]
    true_positives = np.cumsum(ranked_labels == 1)
    precision_at_rank = true_positives / np.arange(1, len(labels) + 1)
    return float(np.sum(precision_at_rank[ranked_labels == 1]) / positives)


def aggregate_prediction_metrics(
    predictions: pd.DataFrame, threshold: float
) -> dict[str, dict[str, float]]:
    """Report less-correlated source and independent group/label metrics."""
    source_rows = (
        predictions.groupby(["dataset", "group", "source"], as_index=False)
        .agg(label=("label", "max"), probability=("probability", "max"))
    )
    group_label_rows = (
        predictions.groupby(["dataset", "group", "label"], as_index=False)
        .agg(probability=("probability", "max"))
    )

    def evaluate(rows: pd.DataFrame) -> dict[str, float]:
        return metric_dict(
            rows["label"].to_numpy(dtype=np.int64),
            rows["probability"].to_numpy(dtype=np.float64),
            threshold,
        )

    return {
        "source_level": evaluate(source_rows),
        "group_label_level": evaluate(group_label_rows),
    }


def confirmed_alert_metrics(
    predictions: pd.DataFrame,
    threshold: float,
    confirm_frames: int = 3,
    cooldown_seconds: float = 15.0,
) -> dict[str, float]:
    """Evaluate source alerts using the same confirmation/cooldown policy as runtime."""
    if confirm_frames < 1:
        raise ValueError("confirm_frames must be positive")
    source_labels: list[int] = []
    source_alerts: list[int] = []
    false_alerts = 0
    true_alerts = 0
    positive_event_sources = 0
    detected_positive_event_sources = 0
    negative_seconds = 0.0
    source_keys = ["source"]
    if "dataset" in predictions.columns:
        source_keys.insert(0, "dataset")
    for _, source_rows in predictions.groupby(source_keys, sort=False):
        source_rows = source_rows.sort_values("window_time")
        labels = source_rows["label"].to_numpy(dtype=np.int64)
        probabilities = source_rows["probability"].to_numpy(dtype=np.float64)
        times = source_rows["window_time"].to_numpy(dtype=np.float64)
        source_label = int(np.any(labels == 1))
        consecutive = 0
        last_alert_at = float("-inf")
        alerts = []
        for label, probability, timestamp in zip(labels, probabilities, times):
            consecutive = consecutive + 1 if probability >= threshold else 0
            if (
                consecutive >= confirm_frames
                and timestamp - last_alert_at >= cooldown_seconds
            ):
                alerts.append((float(timestamp), int(label)))
                last_alert_at = float(timestamp)
                consecutive = 0
        source_labels.append(source_label)
        # A source prediction must depend only on emitted alerts, never on the
        # ground-truth label at the alert timestamp.
        source_alerts.append(int(bool(alerts)))
        if source_label == 1:
            positive_event_sources += 1
            detected_positive_event_sources += int(
                any(label == 1 for _, label in alerts)
            )
        true_alerts += sum(label == 1 for _, label in alerts)
        if len(times):
            typical_step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
            negative_seconds += float(np.sum(labels == 0)) * max(typical_step, 0.0)
            false_alerts += sum(label == 0 for _, label in alerts)

    labels_array = np.asarray(source_labels, dtype=np.int64)
    alerts_array = np.asarray(source_alerts, dtype=np.float64)
    metrics = metric_dict(labels_array, alerts_array, threshold=0.5)
    negative_hours = negative_seconds / 3600.0
    metrics.update(
        {
            "false_alerts": float(false_alerts),
            "true_alerts": float(true_alerts),
            "alert_precision": float(
                true_alerts / max(true_alerts + false_alerts, 1)
            ),
            "labeled_event_recall": float(
                detected_positive_event_sources / max(positive_event_sources, 1)
            ),
            "negative_monitoring_hours": float(negative_hours),
            "false_alarms_per_hour": float(false_alerts / max(negative_hours, 1e-12)),
            "confirm_frames": float(confirm_frames),
            "cooldown_seconds": float(cooldown_seconds),
        }
    )
    return metrics


def select_operational_policy(
    predictions: pd.DataFrame,
    thresholds: np.ndarray,
    confirm_frame_candidates: tuple[int, ...] = (3, 4, 5),
    cooldown_seconds: float = 15.0,
    target_false_alarms_per_hour: float = 5.0,
    minimum_labeled_event_recall: float = 0.85,
    recall_safety_margin: float = 0.05,
    minimum_cohort_positive_sources: int = 5,
) -> tuple[float, int, dict[str, Any], pd.DataFrame]:
    """Select a runtime policy with explicit false-alarm and recall constraints."""
    if predictions.empty:
        raise ValueError("Cannot select an operational policy from empty predictions")
    if len(thresholds) == 0:
        raise ValueError("thresholds cannot be empty")
    if not confirm_frame_candidates or any(value < 1 for value in confirm_frame_candidates):
        raise ValueError("confirm_frame_candidates must contain positive integers")
    if target_false_alarms_per_hour < 0:
        raise ValueError("target_false_alarms_per_hour cannot be negative")
    if not 0.0 <= minimum_labeled_event_recall <= 1.0:
        raise ValueError("minimum_labeled_event_recall must be in [0, 1]")
    if not 0.0 <= recall_safety_margin <= 1.0:
        raise ValueError("recall_safety_margin must be in [0, 1]")
    if minimum_cohort_positive_sources < 1:
        raise ValueError("minimum_cohort_positive_sources must be positive")

    # Validation-only stress checks, not independent cross-validation training.
    # Keep every group together, and exclude tiny positive cohorts from hard
    # constraints: one missed event should not decide the global policy.
    cohorts: dict[str, pd.DataFrame] = {}
    diagnostic_cohorts: dict[str, pd.DataFrame] = {}
    if "dataset" in predictions and "group" in predictions:
        fold_parts: dict[int, list[pd.DataFrame]] = defaultdict(list)
        for dataset, dataset_rows in predictions.groupby("dataset", sort=True):
            cohorts[f"dataset:{dataset}"] = dataset_rows
            group_names = sorted(dataset_rows["group"].astype(str).unique())
            for index, group in enumerate(group_names):
                fold_parts[index % 3].append(
                    dataset_rows[dataset_rows["group"].astype(str) == group]
                )
        for fold, parts in fold_parts.items():
            cohorts[f"group_fold:{fold}"] = pd.concat(parts, ignore_index=True)
        diagnostic_cohorts = cohorts.copy()
        cohorts = {
            name: cohort
            for name, cohort in cohorts.items()
            if int(
                (cohort.groupby(["dataset", "source"])["label"].max() == 1).sum()
            ) >= minimum_cohort_positive_sources
        }
    required_pooled_recall = min(
        1.0, minimum_labeled_event_recall + recall_safety_margin
    )

    rows: list[dict[str, Any]] = []
    for candidate_confirm_frames in sorted(set(confirm_frame_candidates)):
        for threshold in sorted({float(value) for value in thresholds}):
            metrics = confirmed_alert_metrics(
                predictions,
                threshold,
                candidate_confirm_frames,
                cooldown_seconds,
            )
            cohort_recalls = [
                confirmed_alert_metrics(
                    cohort, threshold, candidate_confirm_frames, cooldown_seconds
                )["labeled_event_recall"]
                for cohort in cohorts.values()
            ]
            rows.append(
                {
                    "threshold": threshold,
                    "confirm_frames": int(candidate_confirm_frames),
                    **metrics,
                    "worst_cohort_event_recall": float(
                        min(cohort_recalls, default=metrics["labeled_event_recall"])
                    ),
                }
            )
    audit = pd.DataFrame(rows)
    meets_recall = (
        (audit["labeled_event_recall"] >= required_pooled_recall)
        & (audit["worst_cohort_event_recall"] >= minimum_labeled_event_recall)
    )
    meets_false_alarm_target = (
        audit["false_alarms_per_hour"] <= target_false_alarms_per_hour
    )
    feasible = audit[meets_recall & meets_false_alarm_target]
    if not feasible.empty:
        candidates = feasible
        selection_status = "constraints_met"
        sort_columns = [
            "f1", "alert_precision", "labeled_event_recall",
            "false_alarms_per_hour", "confirm_frames", "threshold",
        ]
        ascending = [False, False, False, True, True, False]
    elif meets_recall.any():
        # Preserve event recall, then choose the least unsafe false-alarm rate.
        candidates = audit[meets_recall]
        selection_status = "false_alarm_target_not_met"
        sort_columns = [
            "false_alarms_per_hour", "f1", "alert_precision",
            "confirm_frames", "threshold",
        ]
        ascending = [True, False, False, True, False]
    else:
        # Do not silently deploy a zero-alert policy when the recall floor fails.
        candidates = audit
        selection_status = "recall_target_not_met"
        sort_columns = [
            "worst_cohort_event_recall", "labeled_event_recall", "false_alarms_per_hour", "f1",
            "alert_precision", "confirm_frames", "threshold",
        ]
        ascending = [False, False, True, False, False, True, False]
    selected = candidates.sort_values(sort_columns, ascending=ascending).iloc[0]
    policy = {
        "selection_status": selection_status,
        "target_false_alarms_per_hour": float(target_false_alarms_per_hour),
        "minimum_labeled_event_recall": float(minimum_labeled_event_recall),
        "required_pooled_recall": float(required_pooled_recall),
        "recall_safety_margin": float(recall_safety_margin),
        "validation_stress_cohorts": list(cohorts),
        "minimum_cohort_positive_sources": minimum_cohort_positive_sources,
        "validation_cohort_metrics": {
            name: {
                "used_as_recall_constraint": name in cohorts,
                **confirmed_alert_metrics(
                    cohort,
                    float(selected["threshold"]),
                    int(selected["confirm_frames"]),
                    cooldown_seconds,
                ),
            }
            for name, cohort in diagnostic_cohorts.items()
        },
        "validation_metrics": {
            key: float(selected[key])
            for key in (
                "f1", "specificity", "false_positive_rate",
                "false_alarms_per_hour", "alert_precision",
                "labeled_event_recall", "negative_monitoring_hours",
                "worst_cohort_event_recall",
            )
        },
    }
    return (
        float(selected["threshold"]),
        int(selected["confirm_frames"]),
        policy,
        audit,
    )


def collect_probabilities(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for features, targets in loader:
            logits = model(features.to(device))
            labels.append(targets.numpy())
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def collect_dense_split_predictions(
    model: nn.Module,
    manifest: pd.DataFrame,
    split: str,
    sequence_length: int,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    """Score every consecutive window for runtime-equivalent alert evaluation."""
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for source in manifest[manifest["split"] == split].itertuples(index=False):
            cached = np.load(Path(source.resolved_cache), allow_pickle=False)
            values = cached["features"]
            labels = cached["frame_labels"]
            times = cached["timestamps"]
            if len(values) < sequence_length:
                continue
            starts = list(range(len(values) - sequence_length + 1))
            for offset in range(0, len(starts), batch_size):
                batch_starts = starts[offset : offset + batch_size]
                features = np.stack(
                    [values[start : start + sequence_length] for start in batch_starts]
                ).astype(np.float32)
                logits = model(torch.from_numpy(features).to(device))
                probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                for start, probability in zip(batch_starts, probabilities):
                    center = start + sequence_length // 2
                    label = int(labels[center])
                    if label not in {0, 1}:
                        continue
                    rows.append(
                        {
                            "dataset": str(source.dataset),
                            "group": str(source.group),
                            "source": str(source.path),
                            "window_time": float(times[center]),
                            "label": label,
                            "probability": float(probability),
                        }
                    )
    if not rows:
        raise ValueError(f"No dense predictions were produced for split={split}")
    return pd.DataFrame(rows)


def validate_cache_manifest(manifest: pd.DataFrame) -> None:
    required_columns = {
        "dataset",
        "group",
        "split",
        "source_id",
        "pose_sha256",
        "feature_dim",
        "sample_fps",
        "seen_frames",
        "detected_frames",
        "detection_coverage",
        "quality_status",
    }
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Cache manifest is missing columns: {sorted(missing_columns)}")
    found_families = set(manifest["dataset"].map(dataset_family))
    missing_families = sorted(KAGGLE_FIVE - found_families)
    if missing_families:
        raise ValueError(f"Stage 07 is missing cache families: {missing_families}")
    if manifest["source_id"].duplicated().any():
        duplicates = manifest.loc[manifest["source_id"].duplicated(), "source_id"].tolist()[:5]
        raise AssertionError(f"Duplicate source_id across shards: {duplicates}")
    for column in ("pose_sha256", "feature_dim", "sample_fps"):
        if manifest[column].nunique() != 1:
            raise AssertionError(
                f"Cache shards disagree on {column}: {manifest[column].unique()}"
            )
    if int(manifest["feature_dim"].iloc[0]) != FEATURE_DIM:
        raise ValueError("Cache feature dimension is not 89")
    if set(manifest["quality_status"].astype(str)) != {"passed"}:
        raise ValueError("Cache manifest contains sources that failed quality checks")
    if (manifest["detected_frames"].astype(int) < 1).any():
        raise ValueError("Cache manifest contains a source without detected pose frames")
    if manifest.groupby(["dataset", "group"])["split"].nunique().max() != 1:
        raise AssertionError("Group leakage across cache shards")


def window_stratum_limits(
    manifest: pd.DataFrame, maximum_per_family_split: int | None
) -> dict[tuple[str, str, int], int | None]:
    """Allocate each family/split window budget across its available labels."""
    limits: dict[tuple[str, str, int], int | None] = {}
    working = manifest.assign(_family=manifest["dataset"].map(dataset_family))
    for (family, split), chunk in working.groupby(["_family", "split"], sort=True):
        labels = sorted({int(label) for label in chunk["label"] if int(label) in {0, 1}})
        if not labels:
            continue
        if maximum_per_family_split is None:
            for label in labels:
                limits[(str(family), str(split), label)] = None
            continue
        base, remainder = divmod(maximum_per_family_split, len(labels))
        if base < 1:
            raise ValueError(
                "max_windows_per_family_split is too small for every available label"
            )
        for index, label in enumerate(labels):
            limits[(str(family), str(split), label)] = base + int(index < remainder)
    return limits


def stage07_train_temporal(
    input_root: str | Path = "/kaggle/input",
    output_dir: str | Path = "/kaggle/working/eldercare_training",
    cache_manifest_paths: list[str | Path] | None = None,
    pose_model_path: str | Path | None = None,
    sequence_length: int = 32,
    window_stride: int = 8,
    max_windows_per_family_split: int | None = 10_000,
    hidden_dim: int = 96,
    batch_size: int = 128,
    epochs: int = 25,
    learning_rate: float = 2e-3,
    hard_negative_epochs: int = 3,
    hard_negative_fraction: float = 0.20,
    hard_negative_boost: float = 3.0,
    confirm_frames: int = 3,
    confirm_frame_candidates: tuple[int, ...] | None = None,
    cooldown_seconds: float = 15.0,
    target_false_alarms_per_hour: float = 5.0,
    minimum_labeled_event_recall: float = 0.85,
    recall_safety_margin: float = 0.05,
    minimum_cohort_positive_sources: int = 5,
    workers: int = 0,
    export_models: bool = True,
    seed: int = 42,
) -> Path:
    from sklearn.metrics import confusion_matrix

    if hard_negative_epochs < 0:
        raise ValueError("hard_negative_epochs cannot be negative")
    if not 0.0 < hard_negative_fraction <= 1.0:
        raise ValueError("hard_negative_fraction must be in (0, 1]")
    if hard_negative_boost < 1.0:
        raise ValueError("hard_negative_boost must be at least 1")
    if confirm_frames < 1:
        raise ValueError("confirm_frames must be positive")
    if confirm_frame_candidates is None:
        confirm_frame_candidates = tuple(range(confirm_frames, confirm_frames + 3))
    if not confirm_frame_candidates or any(value < 1 for value in confirm_frame_candidates):
        raise ValueError("confirm_frame_candidates must contain positive integers")
    if cooldown_seconds < 0:
        raise ValueError("cooldown_seconds cannot be negative")
    if target_false_alarms_per_hour < 0:
        raise ValueError("target_false_alarms_per_hour cannot be negative")
    if not 0.0 <= minimum_labeled_event_recall <= 1.0:
        raise ValueError("minimum_labeled_event_recall must be in [0, 1]")
    if not 0.0 <= recall_safety_margin <= 1.0:
        raise ValueError("recall_safety_margin must be in [0, 1]")
    if minimum_cohort_positive_sources < 1:
        raise ValueError("minimum_cohort_positive_sources must be positive")
    seed_everything(seed)
    input_root, output_dir = Path(input_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = (
        sorted(Path(path) for path in cache_manifest_paths)
        if cache_manifest_paths is not None
        else sorted(input_root.rglob("shard_manifest.csv"))
    )
    if not manifest_paths:
        raise FileNotFoundError("Attach cache notebook outputs containing shard_manifest.csv")
    chunks = []
    for manifest_path in manifest_paths:
        chunk = pd.read_csv(manifest_path).fillna("")
        parent = manifest_path.parent
        chunk["resolved_cache"] = chunk["cache_relpath"].map(
            lambda value, base=parent: str(base / value)
        )
        chunks.append(chunk)
    manifest = pd.concat(chunks, ignore_index=True)
    validate_cache_manifest(manifest)
    pose_source = (
        Path(pose_model_path)
        if pose_model_path is not None
        else find_one(input_root, "yolov8n-pose-finetuned.pt")
    )
    if not pose_source.is_file():
        raise FileNotFoundError(f"Deployment pose model does not exist: {pose_source}")
    if sha256_file(pose_source) != str(manifest["pose_sha256"].iloc[0]):
        raise AssertionError("Deployment pose model does not match the cache pose SHA")

    # Per-label reservoir sampling caps RAM without letting a dominant label consume
    # the complete family/split budget.
    stratum_limits = window_stratum_limits(manifest, max_windows_per_family_split)
    records: dict[
        tuple[str, str, int],
        list[tuple[str, np.ndarray, int, str, str, str, float]],
    ] = defaultdict(list)
    seen_windows: dict[tuple[str, str, int], int] = defaultdict(int)
    reservoir_rng = random.Random(seed + 1701)
    short_cache_sources: list[dict[str, Any]] = []
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        cache_path = Path(row.resolved_cache)
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        cached = np.load(cache_path, allow_pickle=False)
        values = cached["features"]
        frame_labels = cached["frame_labels"]
        timestamps = cached["timestamps"]
        if len(values) < sequence_length:
            short_cache_sources.append(
                {
                    "dataset": str(row.dataset),
                    "source": str(row.path),
                    "detected_frames": len(values),
                    "required_frames": sequence_length,
                }
            )
            continue
        for start in range(0, len(values) - sequence_length + 1, window_stride):
            end = start + sequence_length
            label = int(frame_labels[(start + end) // 2])
            if label not in {0, 1}:
                continue
            bucket_key = (dataset_family(str(row.dataset)), str(row.split), label)
            record = (
                str(row.dataset), values[start:end].copy(), label,
                str(row.group), str(row.split), str(row.path),
                float(timestamps[(start + end) // 2]),
            )
            seen_windows[bucket_key] += 1
            bucket = records[bucket_key]
            stratum_limit = stratum_limits.get(bucket_key)
            if stratum_limit is None or len(bucket) < stratum_limit:
                bucket.append(record)
            else:
                replacement = reservoir_rng.randrange(seen_windows[bucket_key])
                if replacement < stratum_limit:
                    bucket[replacement] = record
        if number % 100 == 0 or number == len(manifest):
            print(f"Windowing sources: {number}/{len(manifest)}")

    (output_dir / "windowing_audit.json").write_text(
        json.dumps(
            {
                "cached_sources": len(manifest),
                "short_sources_skipped": len(short_cache_sources),
                "short_sources": short_cache_sources,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if short_cache_sources:
        print(
            f"[WINDOWING WARNING] Skipped {len(short_cache_sources)} cache sources "
            f"shorter than sequence_length={sequence_length}; see windowing_audit.json"
        )

    selected_records = [record for key in sorted(records) for record in records[key]]
    if not selected_records:
        raise ValueError("No valid windows were produced")
    datasets = np.asarray([item[0] for item in selected_records])
    x = np.asarray([item[1] for item in selected_records], dtype=np.float32)
    y = np.asarray([item[2] for item in selected_records], dtype=np.int64)
    groups = np.asarray([item[3] for item in selected_records])
    splits = np.asarray([item[4] for item in selected_records])
    sources = np.asarray([item[5] for item in selected_records])
    window_times = np.asarray([item[6] for item in selected_records], dtype=np.float64)
    if x.shape[1:] != (sequence_length, FEATURE_DIM):
        raise ValueError(f"Unexpected X shape {x.shape}")
    for split in ("train", "validation", "test"):
        mask = splits == split
        if set(np.unique(y[mask])) != {0, 1}:
            raise ValueError(f"{split} does not contain both classes")
    summary = pd.DataFrame({"split": splits, "dataset": datasets, "label": y})
    print(summary.groupby(["split", "dataset", "label"]).size())

    train_mask, val_mask, test_mask = splits == "train", splits == "validation", splits == "test"
    balance = pd.DataFrame({"dataset": datasets[train_mask], "label": y[train_mask]})
    counts = balance.groupby(["dataset", "label"]).size().to_dict()
    weights = np.asarray([1.0 / counts[(dataset, int(label))] for dataset, label in zip(datasets[train_mask], y[train_mask])])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(
        WindowDataset(x[train_mask], y[train_mask], augment=True), batch_size=batch_size,
        sampler=sampler, num_workers=workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        WindowDataset(x[val_mask], y[val_mask]), batch_size=batch_size * 2,
        num_workers=workers
    )
    test_loader = DataLoader(
        WindowDataset(x[test_mask], y[test_mask]), batch_size=batch_size * 2,
        num_workers=workers
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiScaleTemporalGRU(FEATURE_DIM, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    best_average_precision, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features.to(device)), targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_labels, val_probabilities = collect_probabilities(model, val_loader, device)
        fixed_metrics = metric_dict(val_labels, val_probabilities, 0.5)
        current_average_precision = average_precision(val_labels, val_probabilities)
        history.append(
            {
                "epoch": epoch,
                "phase": "main",
                "loss": float(np.mean(losses)),
                "average_precision": current_average_precision,
                **fixed_metrics,
            }
        )
        if current_average_precision > best_average_precision:
            best_average_precision = current_average_precision
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch:03d} loss={np.mean(losses):.4f} "
            f"val_ap={current_average_precision:.4f} val_f1@0.5={fixed_metrics['f1']:.4f}"
        )
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    if hard_negative_epochs:
        train_eval_loader = DataLoader(
            WindowDataset(x[train_mask], y[train_mask]),
            batch_size=batch_size * 2,
            num_workers=workers,
        )
        _, train_probabilities = collect_probabilities(model, train_eval_loader, device)
        train_labels = y[train_mask]
        train_sources = sources[train_mask]
        entirely_negative_sources = {
            str(source)
            for source in np.unique(train_sources)
            if np.all(train_labels[train_sources == source] == 0)
        }
        if entirely_negative_sources:
            source_scores = sorted(
                (
                    float(np.max(train_probabilities[train_sources == source])),
                    str(source),
                )
                for source in entirely_negative_sources
            )
            selected_source_count = max(
                1, int(np.ceil(len(source_scores) * hard_negative_fraction))
            )
            selected_hard_sources = {
                source for _, source in source_scores[-selected_source_count:]
            }
            hard_negative_mask = np.asarray(
                [
                    label == 0 and str(source) in selected_hard_sources
                    for label, source in zip(train_labels, train_sources)
                ],
                dtype=bool,
            )
            hard_weights = weights.copy()
            hard_weights[hard_negative_mask] *= hard_negative_boost
            hard_sampler = WeightedRandomSampler(
                hard_weights, len(hard_weights), replacement=True
            )
            hard_loader = DataLoader(
                WindowDataset(x[train_mask], y[train_mask], augment=True),
                batch_size=batch_size,
                sampler=hard_sampler,
                num_workers=workers,
                pin_memory=torch.cuda.is_available(),
            )
            hard_optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate * 0.20, weight_decay=1e-3
            )
            for hard_epoch in range(1, hard_negative_epochs + 1):
                model.train()
                hard_losses = []
                for features, targets in hard_loader:
                    hard_optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(features.to(device)), targets.to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    hard_optimizer.step()
                    hard_losses.append(float(loss.detach().cpu()))
                val_labels, val_probabilities = collect_probabilities(
                    model, val_loader, device
                )
                current_average_precision = average_precision(
                    val_labels, val_probabilities
                )
                fixed_metrics = metric_dict(val_labels, val_probabilities, 0.5)
                history.append(
                    {
                        "epoch": epochs + hard_epoch,
                        "phase": "hard_negative",
                        "loss": float(np.mean(hard_losses)),
                        "average_precision": current_average_precision,
                        **fixed_metrics,
                    }
                )
                if current_average_precision > best_average_precision:
                    best_average_precision = current_average_precision
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                print(
                    f"hard_epoch={hard_epoch:02d} loss={np.mean(hard_losses):.4f} "
                    f"val_ap={current_average_precision:.4f} "
                    f"hard_negative_sources={len(selected_hard_sources)} "
                    f"hard_negative_windows={int(hard_negative_mask.sum())}"
                )
            model.load_state_dict(best_state)
    validation_predictions = collect_dense_split_predictions(
        model,
        manifest,
        "validation",
        sequence_length,
        batch_size * 2,
        device,
    )
    dense_val_probabilities = validation_predictions["probability"].to_numpy(
        dtype=np.float64
    )
    thresholds = np.unique(
        np.clip(
            np.concatenate(
                [
                    np.quantile(
                        dense_val_probabilities, np.linspace(0.05, 0.99, 48)
                    ),
                    np.asarray([0.5]),
                ]
            ),
            0.01,
            0.99,
        )
    )
    best_threshold, selected_confirm_frames, operational_policy, policy_audit = (
        select_operational_policy(
            validation_predictions,
            thresholds,
            confirm_frame_candidates,
            cooldown_seconds,
            target_false_alarms_per_hour,
            minimum_labeled_event_recall,
            recall_safety_margin,
            minimum_cohort_positive_sources,
        )
    )
    if operational_policy["selection_status"] != "constraints_met":
        print(
            "[POLICY WARNING] Validation constraints were not all met: "
            f"{operational_policy['selection_status']}. "
            "This policy is an evaluation candidate, not a production approval."
        )
    test_labels, test_probabilities = collect_probabilities(model, test_loader, device)
    test_metrics = metric_dict(test_labels, test_probabilities, float(best_threshold))
    test_datasets = datasets[test_mask]
    test_prediction_frame = pd.DataFrame(
        {
            "dataset": test_datasets,
            "group": groups[test_mask],
            "source": sources[test_mask],
            "window_time": window_times[test_mask],
            "label": test_labels,
            "probability": test_probabilities,
            "prediction": (test_probabilities >= best_threshold).astype(np.int64),
        }
    )
    aggregate_metrics = aggregate_prediction_metrics(
        test_prediction_frame, float(best_threshold)
    )
    operational_test_predictions = collect_dense_split_predictions(
        model,
        manifest,
        "test",
        sequence_length,
        batch_size * 2,
        device,
    )
    operational_metrics = confirmed_alert_metrics(
        operational_test_predictions,
        float(best_threshold),
        selected_confirm_frames,
        cooldown_seconds,
    )
    per_dataset = {}
    per_dataset_aggregates = {}
    per_dataset_operational = {}
    for dataset in sorted(set(test_datasets)):
        mask = test_datasets == dataset
        per_dataset[dataset] = metric_dict(test_labels[mask], test_probabilities[mask], float(best_threshold))
        per_dataset_aggregates[dataset] = aggregate_prediction_metrics(
            test_prediction_frame[test_prediction_frame["dataset"] == dataset],
            float(best_threshold),
        )
        per_dataset_operational[dataset] = confirmed_alert_metrics(
            operational_test_predictions[
                operational_test_predictions["dataset"] == dataset
            ],
            float(best_threshold),
            selected_confirm_frames,
            cooldown_seconds,
        )
    results = {
        "seed": seed,
        "model": "MultiScaleTemporalGRU-from-scratch",
        "pose_sha256": str(manifest["pose_sha256"].iloc[0]),
        "sequence_length": sequence_length,
        "feature_dim": FEATURE_DIM,
        "recommended_threshold": float(best_threshold),
        "training_strategy": {
            "checkpoint_metric": "validation_average_precision",
            "hard_negative_epochs": hard_negative_epochs,
            "hard_negative_fraction": hard_negative_fraction,
            "hard_negative_boost": hard_negative_boost,
            "threshold_metric": "constrained_operational_policy",
            "confirm_frame_candidates": list(confirm_frame_candidates),
            "confirm_frames": selected_confirm_frames,
            "cooldown_seconds": cooldown_seconds,
            "operational_policy": operational_policy,
        },
        "test": test_metrics,
        "test_aggregates": aggregate_metrics,
        "operational_test": operational_metrics,
        "per_dataset": per_dataset,
        "per_dataset_aggregates": per_dataset_aggregates,
        "per_dataset_operational": per_dataset_operational,
        "cache_shards": [str(path) for path in manifest_paths],
    }
    checkpoint = output_dir / "temporal_attention.pt"
    torch.save(
        {
            "state_dict": best_state,
            "model_name": "MultiScaleTemporalGRU",
            "input_dim": FEATURE_DIM,
            "hidden_dim": hidden_dim,
            "sequence_length": sequence_length,
            "metrics": {**test_metrics, "recommended_threshold": float(best_threshold)},
            "training": results,
        },
        checkpoint,
    )
    deployment_dir = output_dir / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pose_source, deployment_dir / "yolov8n-pose-finetuned.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    test_prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    operational_test_predictions.to_csv(
        output_dir / "operational_test_predictions.csv", index=False
    )
    policy_audit.to_csv(output_dir / "operational_policy_audit.csv", index=False)
    np.savetxt(output_dir / "confusion_matrix.csv", confusion_matrix(test_labels, test_probabilities >= best_threshold), fmt="%d", delimiter=",")

    if export_models:
        cpu_model = MultiScaleTemporalGRU(FEATURE_DIM, hidden_dim).cpu().eval()
        cpu_model.load_state_dict(best_state)
        example = torch.zeros(1, sequence_length, FEATURE_DIM)
        onnx_path = output_dir / "temporal_attention.onnx"
        torch.onnx.export(
            cpu_model, example, onnx_path, input_names=["features"], output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17, dynamo=False
        )
        import openvino as ov

        ov_model = ov.convert_model(cpu_model, example_input=example)
        openvino_path = output_dir / "temporal_openvino" / "model.xml"
        openvino_path.parent.mkdir(parents=True, exist_ok=True)
        ov.save_model(ov_model, openvino_path, compress_to_fp16=True)
    deployment = {
        "pose_sha256": str(manifest["pose_sha256"].iloc[0]),
        "pose_model": "deployment/yolov8n-pose-finetuned.pt",
        "classifier_model": "temporal_openvino/model.xml" if export_models else None,
        "classifier_checkpoint": checkpoint.name,
        "model_name": "MultiScaleTemporalGRU",
        "sequence_length": sequence_length,
        "sample_fps": float(manifest["sample_fps"].iloc[0]),
        "fall_threshold": float(best_threshold),
        "required_runtime_policy": "full_window_before_classification",
        "confirm_frames": selected_confirm_frames,
        "cooldown_seconds": cooldown_seconds,
    }
    (output_dir / "deployment_manifest.json").write_text(
        json.dumps(deployment, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    return checkpoint
