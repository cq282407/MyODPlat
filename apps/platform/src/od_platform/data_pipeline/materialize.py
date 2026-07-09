#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : materialize.py
# @Project   : ODPlatform
# @Function  : Materialize YOLO train/val/test directory structure.

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

from od_platform.common.constants import DATASET_SPLITS, IMAGE_EXTENSIONS
from od_platform.data_pipeline.split.registry import DatasetItem, SplitResult


def collect_yolo_items(
    images_dir: Path,
    labels_dir: Path,
    include_unlabeled: bool = False,
) -> list[DatasetItem]:
    """Collect image files and their matching YOLO label files."""

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Label directory does not exist: {labels_dir}")

    label_by_stem = {path.stem: path for path in sorted(labels_dir.glob("*.txt"))}
    items: list[DatasetItem] = []
    for image_path in _iter_images(images_dir):
        label_path = label_by_stem.get(image_path.stem)
        if label_path is None:
            if not include_unlabeled:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            label_path.write_text("", encoding="utf-8")
        items.append(DatasetItem(stem=image_path.stem, image_path=image_path, label_path=label_path))

    if not items:
        raise ValueError(f"No image/label pairs found in {images_dir} and {labels_dir}")
    return items


def materialize_yolo_dataset(
    split_result: SplitResult,
    output_dir: Path,
    config_yaml_path: Path,
    class_names: list[str],
    dataset_name: str,
    split_strategy: str,
    random_state: int,
    train_rate: float,
    val_rate: float,
    test_rate: float,
) -> dict:
    """Copy/hardlink split items into YOLO train/val/test folders."""

    _clear_root_label_files(output_dir / "labels")
    for split in DATASET_SPLITS:
        _clear_split_dir(output_dir / "images" / split)
        _clear_split_dir(output_dir / "labels" / split)

    split_counts: dict[str, int] = {}
    object_counts: dict[str, int] = {}
    class_counts: dict[str, dict[str, int]] = {}

    for split in DATASET_SPLITS:
        items = split_result.get(split, [])
        split_counts[split] = len(items)
        counter: Counter[int] = Counter()
        object_total = 0
        for item in items:
            _copy_or_link(item.image_path, output_dir / "images" / split / item.image_path.name)
            _copy_or_link(item.label_path, output_dir / "labels" / split / item.label_path.name)
            label_counter = _count_label_classes(item.label_path)
            counter.update(label_counter)
            object_total += sum(label_counter.values())
        object_counts[split] = object_total
        class_counts[split] = {
            class_names[class_id]: counter.get(class_id, 0)
            for class_id in range(len(class_names))
        }

    _write_dataset_yaml(output_dir, config_yaml_path, class_names)
    _write_split_report(
        output_dir=output_dir,
        dataset_name=dataset_name,
        split_strategy=split_strategy,
        random_state=random_state,
        train_rate=train_rate,
        val_rate=val_rate,
        test_rate=test_rate,
        split_counts=split_counts,
        object_counts=object_counts,
        class_counts=class_counts,
    )

    return {
        "split_counts": split_counts,
        "object_counts": object_counts,
        "class_counts": class_counts,
        "dataset_yaml": str(output_dir / "dataset.yaml"),
        "config_yaml": str(config_yaml_path),
    }


def _iter_images(images_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _clear_split_dir(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    path.mkdir(parents=True, exist_ok=True)


def _clear_root_label_files(labels_dir: Path) -> None:
    """Remove old unsplit label files so labels/ only contains split folders."""

    if not labels_dir.exists():
        return
    for child in labels_dir.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _count_label_classes(label_path: Path) -> Counter[int]:
    counter: Counter[int] = Counter()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        counter[int(parts[0])] += 1
    return counter


def _write_dataset_yaml(output_dir: Path, config_yaml_path: Path, class_names: list[str]) -> None:
    try:
        config_root = Path(os.path.relpath(output_dir, start=config_yaml_path.parent)).as_posix()
    except ValueError:
        config_root = output_dir.resolve().as_posix()

    (output_dir / "dataset.yaml").write_text(
        _dataset_yaml_content(None, class_names),
        encoding="utf-8",
    )
    config_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    config_yaml_path.write_text(
        _dataset_yaml_content(config_root, class_names),
        encoding="utf-8",
    )


def _dataset_yaml_content(root_ref: str | None, class_names: list[str]) -> str:
    names_yaml = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    path_line = "" if not root_ref else f"path: {root_ref}\n"
    return (
        f"{path_line}"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        f"{names_yaml}\n"
    )


def _write_split_report(
    output_dir: Path,
    dataset_name: str,
    split_strategy: str,
    random_state: int,
    train_rate: float,
    val_rate: float,
    test_rate: float,
    split_counts: dict[str, int],
    object_counts: dict[str, int],
    class_counts: dict[str, dict[str, int]],
) -> None:
    report = {
        "dataset": dataset_name,
        "split_strategy": split_strategy,
        "random_state": random_state,
        "rates": {"train": train_rate, "val": val_rate, "test": test_rate},
        "image_counts": split_counts,
        "object_counts": object_counts,
        "class_counts": class_counts,
    }
    (output_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
