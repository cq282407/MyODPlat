#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : nwpu_vhr10.py
# @Project   : ODPlatform
# @Function  : NWPU VHR-10 txt annotations to YOLO txt converter.

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import List

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, register

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "airplane",
    "ship",
    "storage_tank",
    "baseball_diamond",
    "tennis_court",
    "basketball_court",
    "ground_track_field",
    "harbor",
    "bridge",
    "vehicle",
]

_BOX_PATTERN = re.compile(
    r"\(\s*(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*\)"
    r"\s*,\s*"
    r"\(\s*(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)\s*\)"
    r"\s*,\s*(?P<class_id>\d+)"
)


@register(AnnotationFormat.NWPU_VHR10, supported_tasks=(Task.DETECT,))
def convert_nwpu_vhr10(
    input_dir: Path,
    output_labels_dir: Path,
    options: ConvertOptions,
) -> List[str]:
    """Convert the original NWPU VHR-10 folder layout into YOLO labels."""

    dataset_root = _resolve_dataset_root(input_dir)
    positive_dir = dataset_root / "positive image set"
    negative_dir = dataset_root / "negative image set"
    gt_dir = dataset_root / "ground truth"

    if not positive_dir.exists():
        raise FileNotFoundError(f"Missing positive image set: {positive_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Missing ground truth directory: {gt_dir}")

    staged_images_dir = output_labels_dir.parent / "_source_images"
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    staged_images_dir.mkdir(parents=True, exist_ok=True)
    classes = list(options.classes) if options.classes else list(CLASS_NAMES)
    allowed = set(classes)

    summary = {
        "annotation_files": 0,
        "label_files": 0,
        "objects": 0,
        "skipped_unknown_classes": 0,
        "skipped_invalid_boxes": 0,
        "negative_images": 0,
    }

    for gt_path in sorted(gt_dir.glob("*.txt")):
        summary["annotation_files"] += 1
        image_path = _matching_image(positive_dir, gt_path.stem)
        if image_path is None:
            logger.warning("Missing positive image for annotation: %s", gt_path.name)
            summary["skipped_invalid_boxes"] += 1
            continue
        _link_or_copy(image_path, staged_images_dir / image_path.name)

        width, height = _read_image_size(image_path)
        lines: list[str] = []
        for raw_line in gt_path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_line(raw_line)
            if parsed is None:
                if raw_line.strip():
                    summary["skipped_invalid_boxes"] += 1
                continue

            x1, y1, x2, y2, source_class_id = parsed
            class_name = CLASS_NAMES[source_class_id - 1]
            if class_name not in allowed:
                summary["skipped_unknown_classes"] += 1
                continue

            converted = _convert_box(x1, y1, x2, y2, width, height, options.clip_boxes)
            if converted is None:
                summary["skipped_invalid_boxes"] += 1
                continue

            class_id = classes.index(class_name)
            lines.append(_format_line(class_id, converted))

        if lines or options.write_empty:
            (output_labels_dir / f"{gt_path.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
            summary["label_files"] += 1
            summary["objects"] += len(lines)

    if options.write_empty and negative_dir.exists():
        for image_path in _iter_images(negative_dir):
            staged_name = f"neg_{image_path.name}"
            _link_or_copy(image_path, staged_images_dir / staged_name)
            (output_labels_dir / f"neg_{image_path.stem}.txt").write_text("", encoding="utf-8")
            summary["negative_images"] += 1
            summary["label_files"] += 1

    _write_metadata(output_labels_dir, classes, summary)
    logger.info(
        "NWPU VHR-10 conversion finished: %d annotations, %d labels, %d objects.",
        summary["annotation_files"],
        summary["label_files"],
        summary["objects"],
    )
    return classes


def _resolve_dataset_root(input_dir: Path) -> Path:
    if (input_dir / "ground truth").exists():
        return input_dir
    if input_dir.name.lower() == "ground truth":
        return input_dir.parent
    return input_dir


def _matching_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _iter_images(images_dir: Path) -> list[Path]:
    images: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(images_dir.glob(suffix))
    return sorted(images)


def _read_image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NWPU VHR-10 conversion requires Pillow to read image sizes.") from exc

    with Image.open(image_path) as image:
        return image.size


def _parse_line(raw_line: str) -> tuple[float, float, float, float, int] | None:
    match = _BOX_PATTERN.search(raw_line.strip())
    if match is None:
        return None

    class_id = int(match.group("class_id"))
    if not 1 <= class_id <= len(CLASS_NAMES):
        return None

    return (
        float(match.group("x1")),
        float(match.group("y1")),
        float(match.group("x2")),
        float(match.group("y2")),
        class_id,
    )


def _convert_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
    clip_boxes: bool,
) -> tuple[float, float, float, float] | None:
    xmin = min(x1, x2)
    xmax = max(x1, x2)
    ymin = min(y1, y2)
    ymax = max(y1, y2)

    if clip_boxes:
        xmin = min(max(xmin, 0.0), float(image_width))
        xmax = min(max(xmax, 0.0), float(image_width))
        ymin = min(max(ymin, 0.0), float(image_height))
        ymax = min(max(ymax, 0.0), float(image_height))

    if xmin >= xmax or ymin >= ymax:
        return None

    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    values = (x_center, y_center, width, height)
    if not all(0.0 <= value <= 1.0 for value in values):
        return None
    return values


def _format_line(class_id: int, box: tuple[float, float, float, float]) -> str:
    x_center, y_center, width, height = box
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def _write_metadata(output_labels_dir: Path, classes: List[str], summary: dict[str, int]) -> None:
    output_dir = output_labels_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    classes_txt = "\n".join(f"{idx} {name}" for idx, name in enumerate(classes))
    (output_dir / "classes.txt").write_text(classes_txt + "\n", encoding="utf-8")

    report_lines = [f"{key}: {value}" for key, value in summary.items()]
    report_lines.append(f"classes: {classes}")
    (output_dir / "conversion_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
