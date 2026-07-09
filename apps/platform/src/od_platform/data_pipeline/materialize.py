#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : materialize.py
# @Project   : ODPlatform
# @Function  : Materialize YOLO train/val/test directory structure.

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

from od_platform.common.constants import DATASET_SPLITS, IMAGE_EXTENSIONS, MaterializeMode
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
    materialize_mode: str = MaterializeMode.HARDLINK,
) -> dict:
    """Materialize split items into YOLO folders or txt file lists."""

    if materialize_mode not in MaterializeMode.all():
        raise ValueError(f"Unsupported materialize mode: {materialize_mode}")

    _clear_root_label_files(output_dir / "labels")
    for split in DATASET_SPLITS:
        _clear_txt_list(output_dir / f"{split}.txt")
        _clear_split_dir(output_dir / "images" / split)
        _clear_split_dir(output_dir / "labels" / split)

    split_counts: dict[str, int] = {}
    object_counts: dict[str, int] = {}
    class_counts: dict[str, dict[str, int]] = {}
    fingerprint_rows: list[dict[str, str | int]] = []

    for split in DATASET_SPLITS:
        items = split_result.get(split, [])
        split_counts[split] = len(items)
        counter: Counter[int] = Counter()
        object_total = 0
        txt_lines: list[str] = []
        for item in items:
            image_path = output_dir / "images" / split / item.image_path.name
            label_path = output_dir / "labels" / split / item.label_path.name
            _materialize_file(item.image_path, image_path, materialize_mode)
            _materialize_file(item.label_path, label_path, materialize_mode)
            if materialize_mode == MaterializeMode.TXT:
                txt_lines.append(image_path.resolve().as_posix())

            label_counter = _count_label_classes(item.label_path)
            counter.update(label_counter)
            object_total += sum(label_counter.values())
            fingerprint_rows.append(_build_fingerprint_row(split, image_path, label_path))

        if materialize_mode == MaterializeMode.TXT:
            _write_txt_list(output_dir / f"{split}.txt", txt_lines)
        object_counts[split] = object_total
        class_counts[split] = {
            class_names[class_id]: counter.get(class_id, 0)
            for class_id in range(len(class_names))
        }

    _write_dataset_yaml(output_dir, config_yaml_path, class_names, materialize_mode)
    _write_split_report(
        output_dir=output_dir,
        dataset_name=dataset_name,
        split_strategy=split_strategy,
        materialize_mode=materialize_mode,
        random_state=random_state,
        train_rate=train_rate,
        val_rate=val_rate,
        test_rate=test_rate,
        split_counts=split_counts,
        object_counts=object_counts,
        class_counts=class_counts,
    )
    fingerprint = _write_fingerprint_files(
        output_dir=output_dir,
        dataset_name=dataset_name,
        split_strategy=split_strategy,
        materialize_mode=materialize_mode,
        random_state=random_state,
        split_counts=split_counts,
        rows=fingerprint_rows,
    )

    return {
        "split_counts": split_counts,
        "object_counts": object_counts,
        "class_counts": class_counts,
        "dataset_yaml": str(output_dir / "dataset.yaml"),
        "config_yaml": str(config_yaml_path),
        "materialize_mode": materialize_mode,
        **fingerprint,
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


def _clear_txt_list(path: Path) -> None:
    if path.exists():
        path.unlink()


def _materialize_file(src: Path, dst: Path, materialize_mode: str) -> None:
    if materialize_mode == MaterializeMode.COPY:
        _copy_file(src, dst)
    else:
        _link_or_copy(src, dst)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def _link_or_copy(src: Path, dst: Path) -> None:
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


def _write_dataset_yaml(
    output_dir: Path,
    config_yaml_path: Path,
    class_names: list[str],
    materialize_mode: str,
) -> None:
    names_yaml = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    train_path = "train.txt" if materialize_mode == MaterializeMode.TXT else "images/train"
    val_path = "val.txt" if materialize_mode == MaterializeMode.TXT else "images/val"
    test_path = "test.txt" if materialize_mode == MaterializeMode.TXT else "images/test"
    content = (
        f"path: {output_dir.as_posix()}\n"
        f"train: {train_path}\n"
        f"val: {val_path}\n"
        f"test: {test_path}\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        f"{names_yaml}\n"
    )
    (output_dir / "dataset.yaml").write_text(content, encoding="utf-8")
    config_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    config_yaml_path.write_text(content, encoding="utf-8")


def _write_split_report(
    output_dir: Path,
    dataset_name: str,
    split_strategy: str,
    materialize_mode: str,
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
        "materialize_mode": materialize_mode,
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


def _write_txt_list(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _build_fingerprint_row(split: str, image_path: Path, label_path: Path) -> dict[str, str | int]:
    image_size = _file_size(image_path)
    label_size = _file_size(label_path)
    image_sha256 = _file_sha256(image_path)
    label_sha256 = _file_sha256(label_path)
    combined_sha256 = hashlib.sha256(f"{image_sha256}:{label_sha256}".encode("utf-8")).hexdigest()
    return {
        "split": split,
        "image": image_path.as_posix(),
        "label": label_path.as_posix(),
        "size_bytes": image_size + label_size,
        "image_size_bytes": image_size,
        "label_size_bytes": label_size,
        "sha256": combined_sha256,
        "image_sha256": image_sha256,
        "label_sha256": label_sha256,
    }


def _write_fingerprint_files(
    output_dir: Path,
    dataset_name: str,
    split_strategy: str,
    materialize_mode: str,
    random_state: int,
    split_counts: dict[str, int],
    rows: list[dict[str, str | int]],
) -> dict[str, str]:
    csv_path = output_dir / "dataset_fingerprint.csv"
    json_path = output_dir / "dataset_fingerprint.json"
    fieldnames = [
        "split",
        "image",
        "label",
        "size_bytes",
        "image_size_bytes",
        "label_size_bytes",
        "sha256",
        "image_sha256",
        "label_sha256",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dataset_fingerprint = _dataset_fingerprint(rows)
    payload = {
        "schema_version": 1,
        "dataset": dataset_name,
        "split_strategy": split_strategy,
        "materialize_mode": materialize_mode,
        "random_state": random_state,
        "image_counts": split_counts,
        "item_count": len(rows),
        "fingerprint": dataset_fingerprint,
        "csv": csv_path.name,
        "items": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "fingerprint": dataset_fingerprint,
        "fingerprint_csv": str(csv_path),
        "fingerprint_json": str(json_path),
    }


def _dataset_fingerprint(rows: list[dict[str, str | int]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (str(item["split"]), str(item["image"]))):
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_size(path: Path) -> int:
    return path.stat().st_size


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
