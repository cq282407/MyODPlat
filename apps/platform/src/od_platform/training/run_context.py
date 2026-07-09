#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : run_context.py
# @Project   : ODPlatform
# @Function  : Stable naming metadata for one training run.
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingRunContext:
    """Stable identity shared by logs, output dirs, reports and archives."""

    run_id: str
    started_at: datetime
    task: str
    dataset_label: str
    model_label: str
    operator: str | None = None
    operator_role: str | None = None
    notes: str | None = None

    @property
    def archive_basename(self) -> str:
        return self.run_id


def build_training_run_context(
    *,
    config: Any,
    data_path: Path,
    model_path: Path,
    raw_data: str | Path | None,
    raw_model: str | Path | None,
    started_at: datetime,
    run_id: str | None = None,
    operator: str | None = None,
    operator_role: str | None = None,
    notes: str | None = None,
) -> TrainingRunContext:
    """Build a safe, readable run id before YOLO starts."""

    dataset_label = _label_from(raw_data, fallback=data_path.stem or "dataset")
    model_label = _label_from(raw_model, fallback=model_path.stem or "model")
    explicit_name = _text_or_none(getattr(config, "name", None))
    experiment_name = _text_or_none(getattr(config, "experiment_name", None))
    chosen_id = (
        _text_or_none(run_id)
        or explicit_name
        or experiment_name
        or _default_run_id(started_at, dataset_label, model_label)
    )
    return TrainingRunContext(
        run_id=_safe_name(chosen_id, default="train_run"),
        started_at=started_at,
        task=str(getattr(config, "task", "detect") or "detect"),
        dataset_label=_safe_name(dataset_label, default="dataset"),
        model_label=_safe_name(model_label, default="model"),
        operator=_text_or_none(operator),
        operator_role=_text_or_none(operator_role),
        notes=_text_or_none(notes),
    )


def _default_run_id(started_at: datetime, dataset_label: str, model_label: str) -> str:
    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    return f"train-{stamp}-{dataset_label}-{model_label}"


def _label_from(value: str | Path | None, *, fallback: str) -> str:
    text = _text_or_none(value)
    if not text:
        return fallback
    return Path(text).stem or text


def _text_or_none(value: Any) -> str | None:
    if isinstance(value, Path):
        text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        return None
    stripped = text.strip()
    return stripped or None


def _safe_name(value: str, *, default: str, max_len: int = 96) -> str:
    cleaned = "".join(char if char.isalnum() or char in "_-" else "_" for char in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return (cleaned[:max_len].strip("_-") or default)
