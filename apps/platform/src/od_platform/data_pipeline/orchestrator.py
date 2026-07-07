#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : orchestrator.py
# @Project   : ODPlatform
# @Function  : Dataset pipeline orchestration.

from __future__ import annotations

import logging
from pathlib import Path

from od_platform.common.constants import AnnotationFormat, SplitStrategy, Task
from od_platform.common.paths import DATASET_CONFIGS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from od_platform.data_pipeline.convert.registry import ConvertOptions, get_converter
from od_platform.data_pipeline.convert.service import convert_data_to_yolo
from od_platform.data_pipeline.materialize import (
    collect_yolo_items,
    materialize_yolo_dataset,
)
from od_platform.data_pipeline.split.registry import SplitOptions
from od_platform.data_pipeline.split.service import split_dataset

logger = logging.getLogger(__name__)


class DatasetPipeline:
    """Coordinate conversion from raw dataset into YOLO label files."""

    def __init__(
        self,
        dataset: str,
        annotation_format: str,
        task: str = Task.DETECT,
        train_rate: float = 0.8,
        classes: list[str] | None = None,
        random_state: int = 42,
        split_strategy: str = SplitStrategy.NONE,
        output_name: str | None = None,
        val_rate: float = 0.1,
        test_rate: float = 0.1,
        include_unlabeled: bool = False,
    ) -> None:
        self.dataset = dataset
        self.annotation_format = AnnotationFormat.normalize(annotation_format)
        self.task = task
        self.train_rate = train_rate
        self.classes = classes
        self.random_state = random_state
        self.split_strategy = split_strategy
        self.output_name = output_name or f"{dataset}_yolo"
        self.val_rate = val_rate
        self.test_rate = test_rate
        self.include_unlabeled = include_unlabeled

    def run(self) -> dict:
        if self.task != Task.DETECT:
            raise ValueError(f"Current pipeline only supports task: {Task.DETECT}")
        if self.split_strategy not in SplitStrategy.all():
            raise ValueError(f"Unsupported split strategy: {self.split_strategy}")
        if not get_converter(self.annotation_format).supports(self.task):
            raise ValueError(f"Unsupported annotation format/task: {self.annotation_format}/{self.task}")

        raw_dir = RAW_DATA_DIR / self.dataset
        input_dir = _resolve_annotation_input(raw_dir, self.annotation_format)
        output_dir = PROCESSED_DATA_DIR / self.output_name
        output_labels_dir = output_dir / (
            "_converted_labels"
            if self.split_strategy != SplitStrategy.NONE
            else "labels"
        )

        logger.info(
            "Start dataset conversion: %s (%s -> yolo)",
            self.dataset,
            self.annotation_format,
        )
        classes = convert_data_to_yolo(
            input_dir=input_dir,
            output_labels_dir=output_labels_dir,
            annotation_format=self.annotation_format,
            options=ConvertOptions(task=self.task, classes=self.classes),
        )

        result = self._collect_result(
            raw_dir=raw_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            output_labels_dir=output_labels_dir,
            classes=classes,
        )

        if self.split_strategy != SplitStrategy.NONE:
            split_result = self._split_and_materialize(
                raw_dir=raw_dir,
                output_dir=output_dir,
                output_labels_dir=output_labels_dir,
                classes=classes,
            )
            result.update(split_result)

        logger.info(
            "Dataset conversion finished: dataset=%s output=%s labels=%s objects=%s",
            result["dataset"],
            result["output"],
            result["label_files"],
            result["objects"],
        )
        logger.debug("Dataset conversion result: %s", result)
        return result

    def _collect_result(
        self,
        raw_dir: Path,
        input_dir: Path,
        output_dir: Path,
        output_labels_dir: Path,
        classes: list[str],
    ) -> dict:
        label_files = sorted(output_labels_dir.glob("*.txt"))
        objects = 0
        for label_file in label_files:
            text = label_file.read_text(encoding="utf-8").strip()
            if text:
                objects += len(text.splitlines())

        report_values = self._read_report(output_dir / "conversion_report.txt")
        return {
            "dataset": self.dataset,
            "source": str(raw_dir),
            "input": str(input_dir),
            "output": str(output_dir),
            "classes": classes,
            "annotation_files": int(report_values.get("annotation_files", report_values.get("xml_files", 0))),
            "xml_files": int(report_values.get("xml_files", report_values.get("annotation_files", 0))),
            "label_files": len(label_files),
            "objects": objects,
            "clipped_boxes": int(report_values.get("clipped_boxes", 0)),
            "skipped_unknown_classes": int(report_values.get("skipped_unknown_classes", 0)),
            "skipped_invalid_boxes": int(report_values.get("skipped_invalid_boxes", 0)),
        }

    def _read_report(self, report_path: Path) -> dict[str, str]:
        if not report_path.exists():
            return {}

        values: dict[str, str] = {}
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if ": " not in line:
                continue
            key, value = line.split(": ", 1)
            values[key] = value
        return values

    def _split_and_materialize(
        self,
        raw_dir: Path,
        output_dir: Path,
        output_labels_dir: Path,
        classes: list[str],
    ) -> dict:
        images_dir = _resolve_images_dir(raw_dir, output_dir)
        items = collect_yolo_items(
            images_dir=images_dir,
            labels_dir=output_labels_dir,
            include_unlabeled=self.include_unlabeled,
        )
        split_options = SplitOptions(
            train_rate=self.train_rate,
            val_rate=self.val_rate,
            test_rate=self.test_rate,
            random_state=self.random_state,
        )
        split_result = split_dataset(
            items=items,
            strategy=self.split_strategy,
            options=split_options,
        )
        materialized = materialize_yolo_dataset(
            split_result=split_result,
            output_dir=output_dir,
            config_yaml_path=DATASET_CONFIGS_DIR / f"{self.dataset}.yaml",
            class_names=classes,
            dataset_name=self.dataset,
            split_strategy=self.split_strategy,
            random_state=self.random_state,
            train_rate=self.train_rate,
            val_rate=self.val_rate,
            test_rate=self.test_rate,
        )
        return {
            "paired_images": len(items),
            "split_strategy": self.split_strategy,
            **materialized,
        }


def _resolve_annotation_input(raw_dir: Path, annotation_format: str) -> Path:
    if annotation_format == AnnotationFormat.NWPU_VHR10:
        return raw_dir
    return raw_dir / "annotations"


def _resolve_images_dir(raw_dir: Path, output_dir: Path) -> Path:
    staged_images = output_dir / "_source_images"
    if staged_images.exists():
        return staged_images
    return raw_dir / "images"
