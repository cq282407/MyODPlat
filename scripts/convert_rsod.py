#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : convert_rsod.py
# @Project   : ODPlatform
# @Function  : RSOD conversion script, Pascal VOC -> YOLO.

"""Convert data/raw/rsod Pascal VOC annotations into YOLO txt labels.

Usage:
    python scripts/convert_rsod.py
    python scripts/convert_rsod.py --convert
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLATFORM_SRC = REPO_ROOT / "apps" / "platform" / "src"
sys.path.insert(0, str(PLATFORM_SRC))

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, list_capabilities
from od_platform.data_pipeline.convert.service import convert_data_to_yolo


def main(convert: bool = False) -> None:
    raw_dir = REPO_ROOT / "data" / "raw" / "rsod"
    xml_dir = raw_dir / "annotations"
    images_dir = raw_dir / "images"
    output_dir = REPO_ROOT / "data" / "processed" / "rsod_yolo"
    output_labels_dir = output_dir / "labels"

    print("=" * 60)
    print("RSOD dataset conversion: Pascal VOC -> YOLO")
    print("=" * 60)

    xml_count = len(list(xml_dir.glob("*.xml"))) if xml_dir.exists() else 0
    img_count = 0
    if images_dir.exists():
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            img_count += len(list(images_dir.glob(pattern)))

    print("\n[Dataset Overview]")
    print(f"   Annotation dir: {xml_dir}")
    print(f"   Image dir:      {images_dir}")
    print(f"   XML files:      {xml_count}")
    print(f"   Image files:    {img_count}")
    print(f"   Backgrounds:    {img_count - xml_count}")

    print("\n[Registered Converters]")
    for fmt, tasks in list_capabilities().items():
        enabled = "[V]" if Task.DETECT in tasks else "[!]"
        print(f"   {enabled} {fmt}: supports {', '.join(tasks)}")

    if not convert:
        print("\n[Tip] This is dry-run mode. Add --convert to write YOLO labels.")
        return

    print("\n[Start Conversion]")
    print(f"   Input:  {xml_dir}")
    print(f"   Output: {output_labels_dir}")

    classes = convert_data_to_yolo(
        input_dir=xml_dir,
        output_labels_dir=output_labels_dir,
        annotation_format=AnnotationFormat.PASCAL_VOC,
        options=ConvertOptions(task=Task.DETECT),
    )

    txt_count = len(list(output_labels_dir.glob("*.txt"))) if output_labels_dir.exists() else 0
    print("\n[Done]")
    print(f"   YOLO labels: {txt_count}")
    print(f"   Classes ({len(classes)}): {', '.join(classes)}")
    print(f"   Output dir: {output_dir}")

    print("\n[Preview: first 3 labels]")
    for index, txt_file in enumerate(sorted(output_labels_dir.glob("*.txt"))[:3], 1):
        lines = txt_file.read_text(encoding="utf-8").strip().splitlines()
        print(f"   [{index}] {txt_file.name}: {len(lines)} objects")
        for line in lines[:3]:
            print(f"       {line}")


if __name__ == "__main__":
    main(convert="--convert" in sys.argv)
