#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : pascal_voc.py
# @Project   : ODPlatform
# @Function  : Pascal VOC XML to YOLO txt converter.

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, register

logger = logging.getLogger(__name__)


@register(AnnotationFormat.PASCAL_VOC, supported_tasks=(Task.DETECT,))
def convert_voc(
    input_dir: Path,
    output_labels_dir: Path,
    options: ConvertOptions,
) -> List[str]:
    """Convert Pascal VOC XML files under input_dir into YOLO txt labels."""

    xml_files = sorted(input_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {input_dir}")

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    classes: List[str] = list(options.classes) if options.classes else []
    discovering = options.classes is None

    summary = {
        "xml_files": 0,
        "label_files": 0,
        "objects": 0,
        "clipped_boxes": 0,
        "skipped_unknown_classes": 0,
        "skipped_invalid_boxes": 0,
        "skipped_missing_size": 0,
    }

    for xml_path in xml_files:
        summary["xml_files"] += 1
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        if size is None:
            logger.warning("%s misses <size>, skipped.", xml_path.name)
            summary["skipped_missing_size"] += 1
            continue

        image_width = _read_float(size, "width")
        image_height = _read_float(size, "height")
        if image_width <= 0 or image_height <= 0:
            logger.warning("%s has invalid image size, skipped.", xml_path.name)
            summary["skipped_missing_size"] += 1
            continue

        lines: List[str] = []
        for obj in root.findall("object"):
            class_name = (obj.findtext("name") or "").strip()
            if not class_name:
                summary["skipped_invalid_boxes"] += 1
                continue

            if class_name not in classes:
                if discovering:
                    classes.append(class_name)
                else:
                    logger.debug("%s is not in class whitelist, skipped.", class_name)
                    summary["skipped_unknown_classes"] += 1
                    continue

            bbox = obj.find("bndbox")
            converted = _convert_box(bbox, image_width, image_height, options.clip_boxes)
            if converted is None:
                summary["skipped_invalid_boxes"] += 1
                continue

            class_id = classes.index(class_name)
            yolo_box, clipped = converted
            if clipped:
                summary["clipped_boxes"] += 1
            lines.append(_format_line(class_id, yolo_box))

        if lines or options.write_empty:
            (output_labels_dir / f"{xml_path.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
            summary["label_files"] += 1
            summary["objects"] += len(lines)

    _write_metadata(output_labels_dir, classes, summary)
    logger.info(
        "VOC conversion finished: %d XML files, %d label files, %d classes.",
        summary["xml_files"],
        summary["label_files"],
        len(classes),
    )
    return classes


def _read_float(parent: ET.Element, tag: str) -> float:
    try:
        return float(parent.findtext(tag, "0"))
    except (TypeError, ValueError):
        return 0.0


def _convert_box(
    bbox: ET.Element | None,
    image_width: float,
    image_height: float,
    clip_boxes: bool,
) -> tuple[tuple[float, float, float, float], bool] | None:
    if bbox is None:
        return None

    try:
        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))
    except (TypeError, ValueError):
        return None

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
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    values = (x_center, y_center, width, height)

    if not all(0.0 <= value <= 1.0 for value in values):
        return None

    return values, clipped


def _format_line(class_id: int, box: tuple[float, float, float, float]) -> str:
    x_center, y_center, width, height = box
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def _write_metadata(output_labels_dir: Path, classes: List[str], summary: dict) -> None:
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
