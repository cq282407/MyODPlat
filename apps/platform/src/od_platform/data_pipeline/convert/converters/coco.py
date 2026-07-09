#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : coco.py
# @Project   : ODPlatform
# @Function  : COCO JSON annotations to YOLO txt converter.

from __future__ import annotations

from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Any, DefaultDict, List

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, register

logger = logging.getLogger(__name__)


@register(AnnotationFormat.COCO, supported_tasks=(Task.DETECT,))
def convert_coco(
    input_dir: Path,
    output_labels_dir: Path,
    options: ConvertOptions,
) -> List[str]:
    """Convert one COCO detection JSON file into YOLO label txt files."""

    annotation_path = _resolve_annotation_file(input_dir)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = _read_images(payload)
    categories = _read_categories(payload)
    classes = _resolve_classes(categories, options)
    category_to_class_id = _build_category_mapping(categories, classes)

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    annotations_by_image: DefaultDict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        image_id = annotation.get("image_id")
        if isinstance(image_id, int):
            annotations_by_image[image_id].append(annotation)

    summary = {
        "annotation_files": 1,
        "image_records": len(images),
        "label_files": 0,
        "objects": 0,
        "clipped_boxes": 0,
        "skipped_unknown_classes": 0,
        "skipped_invalid_boxes": 0,
        "skipped_missing_images": 0,
    }

    for image_id, image in sorted(images.items(), key=lambda item: item[1]["file_name"]):
        lines: list[str] = []
        width = image["width"]
        height = image["height"]
        for annotation in annotations_by_image.get(image_id, []):
            category_id = annotation.get("category_id")
            if not isinstance(category_id, int) or category_id not in category_to_class_id:
                summary["skipped_unknown_classes"] += 1
                continue

            converted = _convert_bbox(annotation.get("bbox"), width, height, options.clip_boxes)
            if converted is None:
                summary["skipped_invalid_boxes"] += 1
                continue

            yolo_box, clipped = converted
            if clipped:
                summary["clipped_boxes"] += 1
            lines.append(_format_line(category_to_class_id[category_id], yolo_box))

        if lines or options.write_empty:
            label_name = f"{Path(image['file_name']).stem}.txt"
            (output_labels_dir / label_name).write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
            summary["label_files"] += 1
            summary["objects"] += len(lines)

    _write_metadata(output_labels_dir, classes, summary, annotation_path)
    logger.info(
        "COCO conversion finished: %d images, %d labels, %d objects.",
        summary["image_records"],
        summary["label_files"],
        summary["objects"],
    )
    return classes


def _resolve_annotation_file(input_dir: Path) -> Path:
    if input_dir.is_file() and input_dir.suffix.lower() == ".json":
        return input_dir

    candidates = sorted(input_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No COCO JSON file found in {input_dir}")

    preferred_names = (
        "annotations.json",
        "instances_default.json",
        "instances_train.json",
        "instances_train2017.json",
    )
    by_name = {path.name.lower(): path for path in candidates}
    for name in preferred_names:
        if name in by_name:
            return by_name[name]

    instances = [path for path in candidates if path.name.lower().startswith("instances")]
    return instances[0] if instances else candidates[0]


def _read_images(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    images: dict[int, dict[str, Any]] = {}
    for image in payload.get("images", []):
        image_id = image.get("id")
        file_name = image.get("file_name")
        width = image.get("width")
        height = image.get("height")
        if (
            isinstance(image_id, int)
            and isinstance(file_name, str)
            and _is_positive_number(width)
            and _is_positive_number(height)
        ):
            images[image_id] = {
                "file_name": file_name,
                "width": float(width),
                "height": float(height),
            }
    if not images:
        raise ValueError("COCO JSON does not contain valid images with width/height.")
    return images


def _read_categories(payload: dict[str, Any]) -> dict[int, str]:
    categories: dict[int, str] = {}
    for category in payload.get("categories", []):
        category_id = category.get("id")
        name = category.get("name")
        if isinstance(category_id, int) and isinstance(name, str) and name.strip():
            categories[category_id] = name.strip()
    if not categories:
        raise ValueError("COCO JSON does not contain valid categories.")
    return categories


def _resolve_classes(categories: dict[int, str], options: ConvertOptions) -> list[str]:
    if options.classes is not None:
        return list(options.classes)
    return [name for _, name in sorted(categories.items(), key=lambda item: item[0])]


def _build_category_mapping(categories: dict[int, str], classes: list[str]) -> dict[int, int]:
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    return {
        category_id: class_to_id[name]
        for category_id, name in categories.items()
        if name in class_to_id
    }


def _convert_bbox(
    bbox: Any,
    image_width: float,
    image_height: float,
    clip_boxes: bool,
) -> tuple[tuple[float, float, float, float], bool] | None:
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return None
    if not all(_is_number(value) for value in bbox):
        return None

    x, y, width, height = (float(value) for value in bbox)
    xmin = x
    ymin = y
    xmax = x + width
    ymax = y + height

    clipped = False
    if clip_boxes:
        old_box = (xmin, ymin, xmax, ymax)
        xmin = min(max(xmin, 0.0), image_width)
        ymin = min(max(ymin, 0.0), image_height)
        xmax = min(max(xmax, 0.0), image_width)
        ymax = min(max(ymax, 0.0), image_height)
        clipped = old_box != (xmin, ymin, xmax, ymax)

    if xmin >= xmax or ymin >= ymax:
        return None

    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height
    box_width = (xmax - xmin) / image_width
    box_height = (ymax - ymin) / image_height
    values = (x_center, y_center, box_width, box_height)
    if not all(0.0 <= value <= 1.0 for value in values):
        return None
    return values, clipped


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float)


def _is_positive_number(value: Any) -> bool:
    return _is_number(value) and float(value) > 0


def _format_line(class_id: int, box: tuple[float, float, float, float]) -> str:
    x_center, y_center, width, height = box
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def _write_metadata(
    output_labels_dir: Path,
    classes: List[str],
    summary: dict[str, int],
    annotation_path: Path,
) -> None:
    output_dir = output_labels_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    classes_txt = "\n".join(f"{idx} {name}" for idx, name in enumerate(classes))
    (output_dir / "classes.txt").write_text(classes_txt + "\n", encoding="utf-8")

    report = dict(summary)
    report["annotation_file"] = str(annotation_path)
    report_lines = [f"{key}: {value}" for key, value in report.items()]
    report_lines.append(f"classes: {classes}")
    (output_dir / "conversion_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
