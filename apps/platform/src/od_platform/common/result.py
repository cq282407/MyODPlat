#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : result.py
# @Project   : ODPlatform
# @Function  : Training/evaluation metric snapshot and logger rendering.
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from od_platform.common.constants import Task
from od_platform.common.string_utils import pad_to_width

logger = logging.getLogger(__name__)

_METRIC_FIELDS_BY_TASK: dict[str, list[tuple[str, str]]] = {
    Task.DETECT: [
        ("metrics/precision(B)", "Precision(B)"),
        ("metrics/recall(B)", "Recall(B)"),
        ("metrics/mAP50(B)", "mAP50(B)"),
        ("metrics/mAP50-95(B)", "mAP50-95(B)"),
    ],
    Task.SEGMENT: [
        ("metrics/precision(B)", "Precision(B)"),
        ("metrics/recall(B)", "Recall(B)"),
        ("metrics/mAP50(B)", "mAP50(B)"),
        ("metrics/mAP50-95(B)", "mAP50-95(B)"),
        ("metrics/precision(M)", "Precision(M)"),
        ("metrics/recall(M)", "Recall(M)"),
        ("metrics/mAP50(M)", "mAP50(M)"),
        ("metrics/mAP50-95(M)", "mAP50-95(M)"),
    ],
}


def _safe_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_nan(values: dict[str, float]) -> dict[str, float | None]:
    return {key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in values.items()}


@dataclass(frozen=True)
class TrainMetrics:
    """Structured snapshot of ultralytics train/val metrics."""

    task: str
    save_dir: Path
    timestamp: str
    speed_ms: dict[str, float]
    overall: dict[str, float]
    class_map_50_95: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yolo_results(cls, results: Any, model_trainer: Any = None) -> "TrainMetrics":
        task = getattr(results, "task", "unknown")
        save_dir_raw = getattr(results, "save_dir", None)
        if save_dir_raw is None and model_trainer is not None:
            save_dir_raw = getattr(model_trainer, "save_dir", None)
        save_dir = Path(save_dir_raw) if save_dir_raw is not None else Path("unknown")

        speed_raw = getattr(results, "speed", {}) or {}
        speed_ms = {
            "preprocess": _safe_float(speed_raw.get("preprocess")),
            "inference": _safe_float(speed_raw.get("inference")),
            "loss": _safe_float(speed_raw.get("loss")),
            "postprocess": _safe_float(speed_raw.get("postprocess")),
        }
        valid_speed = [value for value in speed_ms.values() if not math.isnan(value)]
        speed_ms["total"] = sum(valid_speed) if valid_speed else math.nan

        results_dict = getattr(results, "results_dict", {}) or {}
        overall = {"fitness": _safe_float(getattr(results, "fitness", None))}
        for key, value in results_dict.items():
            overall[key] = _safe_float(value)
        if math.isnan(overall["fitness"]):
            overall["fitness"] = _safe_float(results_dict.get("fitness"))

        class_map: dict[str, float] = {}
        names = getattr(results, "names", {}) or {}
        maps = getattr(results, "maps", None)
        if names and maps is not None and hasattr(maps, "__len__"):
            for idx, class_name in names.items():
                try:
                    if idx < len(maps):
                        class_map[str(class_name)] = _safe_float(maps[idx])
                except (TypeError, IndexError):
                    continue

        return cls(
            task=str(task),
            save_dir=save_dir,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            speed_ms=speed_ms,
            overall=overall,
            class_map_50_95=class_map,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "save_dir": str(self.save_dir),
            "timestamp": self.timestamp,
            "speed_ms": _clean_nan(self.speed_ms),
            "overall": _clean_nan(self.overall),
            "class_map_50_95": _clean_nan(self.class_map_50_95),
        }


def log_train_metrics(
    metrics: TrainMetrics,
    *,
    logger: logging.Logger | None = None,
    key_width: int = 20,
    section_width: int = 60,
) -> None:
    """Log a human-friendly metrics summary."""

    log = logger or logging.getLogger(__name__)
    line = "=" * section_width
    thin = "-" * section_width
    log.info(line)
    log.info(pad_to_width(f"训练结果 ({metrics.task} Task)", section_width, align="center"))
    log.info(line)
    log.info(pad_to_width("基本信息", section_width, align="center"))
    log.info(thin)
    log.info("%s: %s", pad_to_width("任务类型", key_width), metrics.task)
    log.info("%s: %s", pad_to_width("保存目录", key_width), metrics.save_dir)
    log.info("%s: %s", pad_to_width("时间戳", key_width), metrics.timestamp)

    log.info(pad_to_width("处理速度 (ms/image)", section_width, align="center"))
    log.info(thin)
    for display, key in (
        ("预处理", "preprocess"),
        ("推理", "inference"),
        ("损失计算", "loss"),
        ("后处理", "postprocess"),
        ("总计", "total"),
    ):
        log.info("%s: %.3f ms", pad_to_width(display, key_width), metrics.speed_ms.get(key, math.nan))

    log.info(pad_to_width("整体评估指标", section_width, align="center"))
    log.info(thin)
    log.info("%s: %.4f", pad_to_width("Fitness 分数", key_width), metrics.overall.get("fitness", math.nan))
    metric_fields = _METRIC_FIELDS_BY_TASK.get(metrics.task, [])
    if metric_fields:
        for raw_key, display in metric_fields:
            log.info("%s: %.4f", pad_to_width(display, key_width), metrics.overall.get(raw_key, math.nan))
    else:
        log.info("task=%r 不在指标表中, 打印 results_dict 全量", metrics.task)
        for key, value in metrics.overall.items():
            if key != "fitness":
                log.info("%s: %.4f", pad_to_width(key, key_width), value)

    if metrics.class_map_50_95:
        log.info(pad_to_width("类别级 mAP@0.5:0.95", section_width, align="center"))
        log.info(thin)
        valid = {key: value for key, value in metrics.class_map_50_95.items() if not math.isnan(value)}
        for class_name, map_value in sorted(valid.items(), key=lambda item: item[1], reverse=True):
            log.info("%s: %.4f", pad_to_width(class_name, key_width), map_value)
    log.info(line)
