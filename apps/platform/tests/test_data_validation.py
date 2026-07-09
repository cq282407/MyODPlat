#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_data_validation.py
# @Project   : ODPlatform
# @Function  : data_validation smoke and regression tests.

from pathlib import Path

from od_platform.cli.validate_data import main as validate_main
from od_platform.data_validation.registry import CheckSeverity
from od_platform.data_validation.service import validate_dataset
from od_platform.data_validation.snapshot import build_snapshot


def test_snapshot_builds_data_dictionary(tmp_path: Path) -> None:
    yaml_path = _make_yolo_dataset(tmp_path)

    snapshot = build_snapshot(yaml_path)
    dictionary = snapshot.as_data_dictionary()

    assert dictionary["nc"] == 1
    assert dictionary["class_names"] == ["head"]
    assert dictionary["totals"] == {"images": 3, "labels": 3, "instances": 3}
    assert dictionary["splits"]["train"]["instances"] == 1


def test_validate_dataset_clean_case_writes_artifacts(tmp_path: Path) -> None:
    yaml_path = _make_yolo_dataset(tmp_path / "dataset")
    run_dir = tmp_path / "run"

    report = validate_dataset(yaml_path, options=_options(run_dir))

    assert report.exit_code == 0
    assert report.overall_severity == CheckSeverity.PASS
    assert (run_dir / "report.json").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "report.html").exists()
    assert (run_dir / "report.docx").exists()
    assert (run_dir / "data_dictionary.json").exists()
    assert (run_dir / "audit.json").exists()
    assert (run_dir / "recommendations.json").exists()
    assert (run_dir / "repair_items.csv").exists()
    assert (run_dir / "repair_items.xlsx").exists()
    assert (run_dir / "charts" / "class_distribution.svg").exists()
    assert any(result.name == "phash_duplicates" and result.severity == CheckSeverity.INFO for result in report.results)


def test_validate_dataset_catches_bad_label_format(tmp_path: Path) -> None:
    yaml_path = _make_yolo_dataset(tmp_path / "dataset")
    (tmp_path / "dataset" / "labels" / "train" / "train_0.txt").write_text(
        "0 0.5 0.5\n",
        encoding="utf-8",
    )

    report = validate_dataset(yaml_path, options=_options(tmp_path / "run"))

    assert report.exit_code == 2
    assert any(result.name == "label_format" and result.severity == "ERROR" for result in report.results)


def test_validate_dataset_catches_real_bbox_overflow(tmp_path: Path) -> None:
    yaml_path = _make_yolo_dataset(tmp_path / "dataset")
    (tmp_path / "dataset" / "labels" / "train" / "train_0.txt").write_text(
        "0 0.990000 0.500000 0.100000 0.250000\n",
        encoding="utf-8",
    )

    report = validate_dataset(yaml_path, options=_options(tmp_path / "run"))

    assert report.exit_code == 2
    assert any(
        result.name == "bbox_within_image" and result.severity == "ERROR"
        for result in report.results
    )


def test_validate_dataset_phash_finds_near_duplicate_images(tmp_path: Path) -> None:
    yaml_path = _make_yolo_dataset(tmp_path / "dataset")
    run_dir = tmp_path / "run"

    report = validate_dataset(
        yaml_path,
        options=_options(run_dir, check_phash=True, phash_threshold=0),
    )

    phash_result = next(result for result in report.results if result.name == "phash_duplicates")
    assert report.exit_code == 1
    assert phash_result.severity == CheckSeverity.WARNING
    assert phash_result.details["pairs_count"] >= 1

    repair_csv = (run_dir / "repair_items.csv").read_text(encoding="utf-8-sig")
    assert "phash_duplicates" in repair_csv
    assert "paired_file" in repair_csv.splitlines()[0]
    assert "distance" in repair_csv.splitlines()[0]


def test_validate_cli_uses_exit_codes(tmp_path: Path) -> None:
    yaml_path = _make_yolo_dataset(tmp_path / "dataset")

    exit_code = validate_main(
        [
            "--yaml",
            str(yaml_path),
            "--run-id",
            "pytest",
            "--output-dir",
            str(tmp_path / "run"),
            "--operator",
            "pytest",
        ]
    )

    assert exit_code == 0


def _options(run_dir: Path, **overrides):
    from od_platform.data_validation.registry import ValidationOptions

    return ValidationOptions(run_id="pytest", output_dir=run_dir, operator="pytest", **overrides)


def _make_yolo_dataset(root: Path) -> Path:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        _write_test_image(root / "images" / split / f"{split}_0.jpg")
        (root / "labels" / split / f"{split}_0.txt").write_text(
            "0 0.500000 0.500000 0.250000 0.250000\n",
            encoding="utf-8",
        )

    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        (
            f"path: {root.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "nc: 1\n"
            "names:\n"
            "  0: head\n"
        ),
        encoding="utf-8",
    )
    return yaml_path


def _write_test_image(path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path, format="JPEG")
