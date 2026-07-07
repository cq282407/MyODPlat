#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : render.py
# @Project   : ODPlatform
# @Function  : Human-friendly terminal report for odp-transform.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

H1_LINE = "=" * 72
H2_LINE = "-" * 72


def render_transform_report(
    result: dict[str, Any],
    log: logging.Logger | None = None,
    *,
    task: str,
    source_format: str,
    train_rate: float,
    val_rate: float,
    test_rate: float,
    seed: int,
) -> None:
    """Render a compact teacher-style report for a completed transform run."""

    log = log or logger
    _render_header(result, log, task=task, source_format=source_format)
    _render_conversion_summary(result, log)
    _render_class_balance(result, log)
    _render_split_summary(
        result,
        log,
        train_rate=train_rate,
        val_rate=val_rate,
        test_rate=test_rate,
        seed=seed,
    )
    _render_outputs(result, log)


def _render_header(
    result: dict[str, Any],
    log: logging.Logger,
    *,
    task: str,
    source_format: str,
) -> None:
    log.info(H1_LINE)
    log.info("                   ODPlatform Dataset Transform Report")
    log.info(H1_LINE)
    log.info("  dataset      %s", result.get("dataset", "N/A"))
    log.info("  source       %s", result.get("source", "N/A"))
    log.info("  format       %s -> yolo", source_format)
    log.info("  task         %s", task)
    log.info("  output       %s", result.get("output", "N/A"))


def _render_conversion_summary(result: dict[str, Any], log: logging.Logger) -> None:
    classes = result.get("classes") or []
    log.info(H2_LINE)
    log.info("  Conversion Summary")
    log.info("    annotation files       %s", _fmt_int(result.get("annotation_files", result.get("xml_files", 0))))
    log.info("    label files            %s", _fmt_int(result.get("label_files", 0)))
    log.info("    objects                %s", _fmt_int(result.get("objects", 0)))
    log.info("    classes                %s", _format_classes(classes))
    log.info("    clipped boxes          %s", _fmt_int(result.get("clipped_boxes", 0)))
    log.info(
        "    skipped                unknown_class=%s  invalid_box=%s",
        _fmt_int(result.get("skipped_unknown_classes", 0)),
        _fmt_int(result.get("skipped_invalid_boxes", 0)),
    )


def _render_class_balance(result: dict[str, Any], log: logging.Logger) -> None:
    classes = result.get("classes") or []
    split_class_counts = result.get("class_counts") or {}
    if not classes or not split_class_counts:
        return

    totals = {
        class_name: sum(split_counts.get(class_name, 0) for split_counts in split_class_counts.values())
        for class_name in classes
    }
    total_objects = sum(totals.values()) or 1

    log.info(H2_LINE)
    log.info("  Class Balance")
    log.info("    %-16s %10s %10s %10s %10s %8s", "class", "total", "train", "val", "test", "share")
    for class_name in classes:
        train = split_class_counts.get("train", {}).get(class_name, 0)
        val = split_class_counts.get("val", {}).get(class_name, 0)
        test = split_class_counts.get("test", {}).get(class_name, 0)
        total = totals[class_name]
        log.info(
            "    %-16s %10s %10s %10s %10s %7.1f%%",
            class_name,
            _fmt_int(total),
            _fmt_int(train),
            _fmt_int(val),
            _fmt_int(test),
            total / total_objects * 100,
        )


def _render_split_summary(
    result: dict[str, Any],
    log: logging.Logger,
    *,
    train_rate: float,
    val_rate: float,
    test_rate: float,
    seed: int,
) -> None:
    split_counts = result.get("split_counts")
    if not split_counts:
        log.info(H2_LINE)
        log.info("  Split Summary")
        log.info("    split disabled; converted labels were written without train/val/test materialization")
        return

    object_counts = result.get("object_counts") or {}
    log.info(H2_LINE)
    log.info("  Split Summary")
    log.info(
        "    strategy=%s  seed=%s  rates=train %.0f%% / val %.0f%% / test %.0f%%",
        result.get("split_strategy", "N/A"),
        seed,
        train_rate * 100,
        val_rate * 100,
        test_rate * 100,
    )
    log.info("    paired images          %s", _fmt_int(result.get("paired_images", 0)))
    log.info("    %-8s %10s %10s %10s", "split", "images", "labels", "objects")
    for split in ("train", "val", "test"):
        count = split_counts.get(split, 0)
        log.info(
            "    %-8s %10s %10s %10s",
            split,
            _fmt_int(count),
            _fmt_int(count),
            _fmt_int(object_counts.get(split, 0)),
        )


def _render_outputs(result: dict[str, Any], log: logging.Logger) -> None:
    output_dir = Path(result["output"]) if result.get("output") else None
    log.info(H2_LINE)
    log.info("  Outputs")
    if output_dir is not None:
        log.info("    processed root         %s", output_dir)
        log.info("    conversion report      %s", output_dir / "conversion_report.txt")
        log.info("    split report           %s", output_dir / "split_report.json")
        log.info("    classes file           %s", output_dir / "classes.txt")
    if result.get("dataset_yaml"):
        log.info("    dataset yaml           %s", result["dataset_yaml"])
    if result.get("config_yaml"):
        log.info("    config yaml            %s", result["config_yaml"])
    log.info(H1_LINE)


def _format_classes(classes: list[str]) -> str:
    if not classes:
        return "0"
    return f"{len(classes)} ({', '.join(classes)})"


def _fmt_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)
