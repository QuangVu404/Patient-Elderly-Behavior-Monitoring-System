from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from training import kaggle_staged as ks


def test_validate_kaggle_inputs_requires_all_five_mounts(tmp_path: Path) -> None:
    for slug in ks.KAGGLE_INPUT_SLUGS.values():
        (tmp_path / slug).mkdir()

    mounted = ks.validate_kaggle_inputs(tmp_path)

    assert set(mounted) == ks.KAGGLE_FIVE
    assert all(path.parent == tmp_path for path in mounted.values())


def test_validate_kaggle_inputs_accepts_titles_and_visible_layout(tmp_path: Path) -> None:
    layouts = {
        "Fall Video Dataset": ("Fall", "No_Fall"),
        "ur fall detection dataset": ("UR_fall_detection_dataset_cam0_rgb",),
        "CAUCAFall": tuple(f"Subject{i}" for i in range(1, 6)),
        "Multiple Cameras Fall Dataset": tuple(f"chute{i:02d}" for i in range(1, 25)),
        "UCF101 - Action Recognition": ("train", "test", "val"),
    }
    for title, children in layouts.items():
        for child in children:
            (tmp_path / title / child).mkdir(parents=True, exist_ok=True)

    mounted = ks.validate_kaggle_inputs(tmp_path)

    assert set(mounted) == ks.KAGGLE_FIVE
    assert mounted["CAUCAFall"].name == "CAUCAFall"
    assert mounted["MCFD"].name == "Multiple Cameras Fall Dataset"


def test_validate_kaggle_inputs_accepts_owner_scoped_mounts(tmp_path: Path) -> None:
    owners = {
        "payutch": "fall-video-dataset",
        "tuyenldvn": "caucafall",
        "shahliza27": "ur-fall-detection-dataset",
        "soumicksarker": "multiple-cameras-fall-dataset",
        "matthewjansen": "ucf101-action-recognition",
    }
    for owner, slug in owners.items():
        (tmp_path / "datasets" / owner / slug).mkdir(parents=True)

    mounted = ks.validate_kaggle_inputs(tmp_path)

    assert set(mounted) == ks.KAGGLE_FIVE
    assert mounted["CAUCAFall"] == tmp_path / "datasets" / "tuyenldvn" / "caucafall"


def test_validate_kaggle_inputs_reports_missing_slug(tmp_path: Path) -> None:
    (tmp_path / "fall-video-dataset").mkdir()

    try:
        ks.validate_kaggle_inputs(tmp_path, {"FallVision", "CAUCAFall"})
    except FileNotFoundError as error:
        assert "caucafall" in str(error)
        assert "does not download missing data" in str(error)
    else:
        raise AssertionError("Missing Kaggle Input must stop the stage")


def test_resolve_mounted_file_never_falls_back_to_network(tmp_path: Path) -> None:
    checkpoint = tmp_path / "code-input" / "yolov8n-pose.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    assert ks.resolve_mounted_file(tmp_path, checkpoint.name) == checkpoint

    try:
        ks.resolve_mounted_file(tmp_path, "missing.pt")
    except FileNotFoundError as error:
        assert "automatic network download is disabled" in str(error)
    else:
        raise AssertionError("Missing checkpoint must not trigger an implicit download")


def test_find_one_rejects_ambiguous_artifacts(tmp_path: Path) -> None:
    for folder in ("old", "new"):
        path = tmp_path / folder / "master_manifest.csv"
        path.parent.mkdir()
        path.touch()

    try:
        ks.find_one(tmp_path, "master_manifest.csv")
    except RuntimeError as error:
        assert "multiple" in str(error)
        assert "explicit path" in str(error)
    else:
        raise AssertionError("Ambiguous stage artifacts must not be selected silently")


def test_prepare_pose_baseline_creates_cache_compatible_artifacts(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    checkpoint = input_root / "code" / "yolov8n-pose.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"local checkpoint")

    portable = ks.stage01_prepare_pose_baseline(input_root, tmp_path / "output")

    assert portable.read_bytes() == checkpoint.read_bytes()
    assert (portable.parent / "pose_model.sha256").read_text(encoding="utf-8").strip()
    assert '"epochs": 0' in (portable.parent / "pose_metadata.json").read_text(
        encoding="utf-8"
    )


def _cache_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": sorted(ks.KAGGLE_FIVE),
            "group": [f"group_{index}" for index in range(5)],
            "split": ["train"] * 5,
            "source_id": [f"source_{index}" for index in range(5)],
            "pose_sha256": ["same-checkpoint"] * 5,
            "feature_dim": [ks.FEATURE_DIM] * 5,
            "sample_fps": [10.0] * 5,
            "seen_frames": [100] * 5,
            "detected_frames": [80] * 5,
            "detection_coverage": [0.8] * 5,
            "quality_status": ["passed"] * 5,
        }
    )


def test_cache_manifest_requires_all_five_families() -> None:
    manifest = _cache_manifest()
    manifest = manifest[manifest["dataset"] != "UCF101"]

    try:
        ks.validate_cache_manifest(manifest)
    except ValueError as error:
        assert "UCF101" in str(error)
    else:
        raise AssertionError("Stage 07 must reject an incomplete cache set")


def test_canonical_inventory_rejects_same_name_incomplete_mirror() -> None:
    manifest = pd.DataFrame(
        {"dataset": ["FallVision", "CAUCAFall", "URFD", "MCFD", "UCF101"],
         "group": ["fv", "cauca", "urfd_fall_01", "mcfd_scenario_01", "ucf"]}
    )

    try:
        ks.validate_canonical_inventory(manifest)
    except ValueError as error:
        assert "matching Kaggle title is not enough" in str(error)
        assert "FallVision" in str(error)
    else:
        raise AssertionError("An incomplete same-name mirror must be rejected")


def test_cache_manifest_requires_one_sample_rate() -> None:
    manifest = _cache_manifest()
    manifest.loc[manifest["dataset"] == "UCF101", "sample_fps"] = 6.0

    try:
        ks.validate_cache_manifest(manifest)
    except AssertionError as error:
        assert "sample_fps" in str(error)
    else:
        raise AssertionError("Stage 07 must reject incompatible sample rates")


def test_stage00_builds_stable_group_splits(tmp_path: Path) -> None:
    rows = []
    for dataset in ("FallVision", "MCFD"):
        for subject in range(1, 5):
            for label in (0, 1):
                video = tmp_path / dataset / f"subject{subject}_{label}.mp4"
                video.parent.mkdir(parents=True, exist_ok=True)
                video.touch()
                rows.append(
                    {
                        "dataset": dataset,
                        "path": str(video),
                        "label": label,
                        "group": f"{dataset.lower()}_subject{subject}",
                        "source_type": "video",
                    }
                )
    pd.DataFrame(rows).to_csv(tmp_path / "eldercare_manifest.csv", index=False)

    output = tmp_path / "output"
    master_path = ks.stage00_build_manifest(tmp_path, output, seed=42, require_all_five=False)
    manifest = pd.read_csv(master_path)

    assert set(manifest["dataset"]) == {"FallVision", "MCFD"}
    assert set(manifest["split"]) == {"train", "validation", "test"}
    assert manifest["source_id"].is_unique
    assert manifest.groupby(["dataset", "group"])["split"].nunique().max() == 1
    assert (output / "split_report.json").exists()


def test_two_group_label_stratum_keeps_train_and_test_examples() -> None:
    rows = []
    for label, count in ((0, 2), (1, 6)):
        for index in range(count):
            rows.append(
                {
                    "dataset": "MCFD",
                    "path": f"source_{label}_{index}",
                    "label": label,
                    "group": f"group_{label}_{index}",
                }
            )
    for label in (0, 1):
        for index in range(6):
            rows.append(
                {
                    "dataset": "Auxiliary",
                    "path": f"aux_{label}_{index}",
                    "label": label,
                    "group": f"aux_group_{label}_{index}",
                }
            )

    split = ks.assign_group_splits(pd.DataFrame(rows), seed=42)
    negative_splits = set(
        split.loc[(split["dataset"] == "MCFD") & (split["label"] == 0), "split"]
    )

    assert negative_splits == {"train", "test"}


def test_source_limit_preserves_split_label_strata() -> None:
    rows = pd.DataFrame(
        [
            {"split": split, "label": label, "group": f"{split}_{label}_{index}"}
            for split in ("train", "validation", "test")
            for label in (0, 1)
            for index in range(10)
        ]
    )

    limited = ks._stratified_source_limit(rows, 12, seed=42)

    assert len(limited) == 12
    assert limited.groupby(["split", "label"]).ngroups == 6


def test_window_budget_is_balanced_only_when_both_labels_exist() -> None:
    manifest = pd.DataFrame(
        {
            "dataset": ["CAUCAFall", "CAUCAFall", "UCF101"],
            "split": ["train", "train", "train"],
            "label": [0, 1, 0],
        }
    )

    limits = ks.window_stratum_limits(manifest, 10_000)

    assert limits[("CAUCAFall", "train", 0)] == 5_000
    assert limits[("CAUCAFall", "train", 1)] == 5_000
    assert limits[("UCF101", "train", 0)] == 10_000


def test_pose_validation_selects_groups_from_every_family() -> None:
    rows = pd.DataFrame(
        [
            {"dataset": family, "group": f"{family}_{index}"}
            for family in ("FallVision", "CAUCAFall", "URFD")
            for index in range(10)
        ]
    )

    selected = ks.pose_validation_group_keys(rows, seed=42)

    assert {key.split("::", 1)[0] for key in selected} == {
        "FallVision", "CAUCAFall", "URFD"
    }
    assert len(selected) == 3


def test_metric_dict_handles_a_nonfall_only_dataset() -> None:
    labels = np.asarray([0, 0, 0])
    probabilities = np.asarray([0.1, 0.2, 0.9])

    metrics = ks.metric_dict(labels, probabilities, threshold=0.5)

    assert metrics["specificity"] == 2 / 3
    assert metrics["false_positive_rate"] == 1 / 3
    assert metrics["balanced_accuracy"] == 2 / 3


def test_average_precision_rewards_correct_probability_ranking() -> None:
    labels = np.asarray([1, 0, 1, 0])

    good = ks.average_precision(labels, np.asarray([0.9, 0.2, 0.8, 0.1]))
    bad = ks.average_precision(labels, np.asarray([0.2, 0.9, 0.1, 0.8]))

    assert good == 1.0
    assert good > bad


def test_prediction_metrics_include_source_and_group_label_levels() -> None:
    predictions = pd.DataFrame(
        {
            "dataset": ["A", "A", "A", "A"],
            "group": ["g1", "g1", "g2", "g2"],
            "source": ["fall", "fall", "normal", "normal"],
            "label": [1, 1, 0, 0],
            "probability": [0.4, 0.9, 0.1, 0.2],
        }
    )

    metrics = ks.aggregate_prediction_metrics(predictions, threshold=0.5)

    assert metrics["source_level"]["accuracy"] == 1.0
    assert metrics["source_level"]["samples"] == 2.0
    assert metrics["group_label_level"]["accuracy"] == 1.0


def test_confirmed_alert_metrics_ignore_isolated_probability_spikes() -> None:
    predictions = pd.DataFrame(
        {
            "source": ["fall"] * 4 + ["normal"] * 5,
            "window_time": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "label": [0, 1, 1, 1, 0, 0, 0, 0, 0],
            "probability": [0.1, 0.8, 0.9, 0.85, 0.1, 0.9, 0.1, 0.8, 0.1],
        }
    )

    metrics = ks.confirmed_alert_metrics(
        predictions, threshold=0.5, confirm_frames=3, cooldown_seconds=15.0
    )

    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["false_alerts"] == 0.0


def test_confirmed_alert_metrics_count_consecutive_negative_alerts_as_false() -> None:
    predictions = pd.DataFrame(
        {
            "source": ["normal"] * 4,
            "window_time": [0.0, 1.0, 2.0, 3.0],
            "label": [0, 0, 0, 0],
            "probability": [0.8, 0.9, 0.85, 0.1],
        }
    )

    metrics = ks.confirmed_alert_metrics(
        predictions, threshold=0.5, confirm_frames=3, cooldown_seconds=15.0
    )

    assert metrics["specificity"] == 0.0
    assert metrics["false_alerts"] == 1.0
    assert metrics["alert_precision"] == 0.0


def test_frame_label_uses_dense_segments() -> None:
    segments = "[[0, 0.0, 1.0], [1, 1.0, 2.0], [2, 2.0, 3.0]]"
    assert ks.frame_label(0.5, -2, segments) == 0
    assert ks.frame_label(1.5, -2, segments) == 1
    assert ks.frame_label(2.5, -2, segments) == 1
    assert ks.frame_label(4.0, -2, segments) == -1
    assert ks.frame_label(0.0, 1, "") == 1


def test_image_sampling_preserves_requested_average_fps(tmp_path: Path) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    for index in range(100):
        cv2.imwrite(str(tmp_path / f"frame_{index:03d}.png"), frame)

    sampled = list(
        ks.iter_source_frames(
            tmp_path, source_type="images", dataset="URFD", target_fps=10.0
        )
    )

    assert len(sampled) == 40
    assert sampled[0][0] == 0
    assert sampled[-1][0] in {97, 98}


def test_mixed_pose_dataset_uses_only_gold_validation(tmp_path: Path) -> None:
    gold = tmp_path / "gold.yaml"
    pseudo = tmp_path / "pseudo.yaml"
    gold.write_text(
        "path: .\ntrain: gold_train\nval: gold_val\nkpt_shape: [17, 3]\n",
        encoding="utf-8",
    )
    pseudo.write_text(
        "path: .\ntrain: pseudo_train\nval: pseudo_val\nkpt_shape: [17, 3]\n",
        encoding="utf-8",
    )

    output = ks.combine_pose_datasets(gold, pseudo, tmp_path / "mixed.yaml", 2)
    config = ks.yaml.safe_load(output.read_text(encoding="utf-8"))

    assert config["val"] == [str(tmp_path / "gold_val")]
    assert all("pseudo_val" not in path for path in config["val"])


def test_urfd_discovery_reads_image_sequences(tmp_path: Path) -> None:
    root = tmp_path / "ur-fall-detection-dataset" / "UR_fall_detection_dataset_cam0_rgb"
    fall = root / "fall-01-cam0-rgb"
    adl = root / "adl-02-cam0-rgb"
    fall.mkdir(parents=True)
    adl.mkdir(parents=True)
    (fall / "fall-01-cam0-rgb-001.png").touch()
    (adl / "adl-02-cam0-rgb-001.png").touch()

    rows = ks.discover_urfd(tmp_path)

    assert {row["label"] for row in rows} == {0, 1}
    assert {row["source_type"] for row in rows} == {"images"}


def test_caucafall_discovery_uses_png_sequences_and_subject_groups(tmp_path: Path) -> None:
    mounted = tmp_path / "caucafall"
    fall = mounted / "Subject.01" / "FallBackwards" / "FallBackwardsS1"
    adl = mounted / "Subject.01" / "ADL" / "WalkS1"
    for directory in (fall, adl):
        directory.mkdir(parents=True)
        (directory / "frame001.png").touch()
        (directory / "frame001.txt").write_text(
            "1 0.5 0.5 0.4 0.8\n" if directory == fall else "0 0.5 0.5 0.4 0.8\n",
            encoding="utf-8",
        )
        (directory / "recording.avi").touch()

    rows = ks.discover_caucafall(tmp_path)

    assert len(rows) == 2
    assert {row["label"] for row in rows} == {0, 1}
    assert {row["source_type"] for row in rows} == {"images"}
    assert {row["group"] for row in rows} == {"caucafall_subject_01"}
    assert all(not row["path"].endswith(".avi") for row in rows)
    assert ks.caucafall_frame_labels(fall) == {0: 1}
    assert ks.caucafall_frame_labels(adl) == {0: 0}


def test_ucf101_discovery_groups_related_clips(tmp_path: Path) -> None:
    action = tmp_path / "ucf101-action-recognition" / "train" / "WalkingWithDog"
    action.mkdir(parents=True)
    (action / "v_WalkingWithDog_g01_c01.avi").touch()
    (action / "v_WalkingWithDog_g01_c02.avi").touch()

    rows = ks.discover_ucf101(tmp_path)

    assert len(rows) == 2
    assert {row["label"] for row in rows} == {0}
    assert len({row["group"] for row in rows}) == 1


def test_ucf101_discovery_ignores_noncanonical_and_duplicate_clips(tmp_path: Path) -> None:
    mounted = tmp_path / "ucf101-action-recognition"
    train = mounted / "train" / "WalkingWithDog"
    duplicate = mounted / "val" / "WalkingWithDog"
    train.mkdir(parents=True)
    duplicate.mkdir(parents=True)
    (train / "v_WalkingWithDog_g01_c01.avi").touch()
    (duplicate / "v_WalkingWithDog_g01_c01.avi").touch()
    (train / "preview.avi").touch()

    rows = ks.discover_ucf101(tmp_path)

    assert len(rows) == 1


def test_fallvision_discovery_uses_class_directories(tmp_path: Path) -> None:
    mounted = tmp_path / "fall-video-dataset"
    fall = mounted / "Fall" / "Raw_Video" / "subject01_event.mp4"
    normal = mounted / "No_Fall" / "Raw_Video" / "subject02_event.mp4"
    masked = mounted / "Fall" / "Masked_Video" / "subject01_event.mp4"
    for video in (fall, normal, masked):
        video.parent.mkdir(parents=True, exist_ok=True)
        video.touch()

    rows = ks.discover_fallvision(tmp_path)

    assert len(rows) == 2
    assert {row["label"] for row in rows} == {0, 1}
    assert all("Masked_Video" not in row["path"] for row in rows)


def test_mcfd_discovery_groups_camera_views_by_scenario(tmp_path: Path) -> None:
    mounted = tmp_path / "multiple-cameras-fall-dataset"
    videos = [
        mounted / "dataset" / "chute01" / "cam1.avi",
        mounted / "dataset" / "chute01" / "cam8.avi",
        mounted / "dataset" / "chute24" / "cam1.avi",
    ]
    for video in videos:
        video.parent.mkdir(parents=True, exist_ok=True)
        video.touch()

    rows = ks.discover_mcfd(tmp_path)

    assert {row["label"] for row in rows} == {0, 1}
    assert len({row["group"] for row in rows[:2]}) == 1
    assert rows[0]["camera"] in {"1", "8"}
