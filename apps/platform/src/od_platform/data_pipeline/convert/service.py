#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : service.py
# @Project   : ODPlatform
# @Function  : Unified service entrance for annotation conversion.

from __future__ import annotations

from pathlib import Path
from typing import List

from od_platform.common.constants import AnnotationFormat
from od_platform.data_pipeline.convert.registry import ConvertOptions, get_converter


def convert_data_to_yolo(
    input_dir: Path,
    output_labels_dir: Path,
    annotation_format: str,
    options: ConvertOptions,
) -> List[str]:
    """Convert source annotations to YOLO label txt files."""

    normalized_format = AnnotationFormat.normalize(annotation_format)
    entry = get_converter(normalized_format)
    if not entry.supports(options.task):
        raise ValueError(
            f"Format {annotation_format!r} does not support task {options.task!r}. "
            f"Supported tasks: {entry.supported_tasks}"
        )
    return entry.func(input_dir, output_labels_dir, options)
