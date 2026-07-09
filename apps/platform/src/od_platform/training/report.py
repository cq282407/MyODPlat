#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : report.py
# @Project   : ODPlatform
# @Function  : Training report artifacts.
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from od_platform.common.result import TrainMetrics
from od_platform.training.run_context import TrainingRunContext


def write_training_report(
    *,
    output_dir: Path,
    run_context: TrainingRunContext,
    config: Any,
    merger: Any,
    metrics: TrainMetrics,
    train_time: float,
    raw_data: str | Path | None,
    data_path: Path,
    raw_model: str | Path | None,
    model_path: Path,
    validation_report: Any | None,
    archived: dict[str, Path],
    best_weight: Path | None,
    last_weight: Path | None,
    log_path: Path | None,
    visualization_path: Path | None,
    audit_path: Path | None,
) -> tuple[Path | None, Path | None]:
    """Write human-facing D6 training report artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "training_report.json"
    markdown_path = output_dir / "training_report.md"
    payload = build_training_report_payload(
        run_context=run_context,
        config=config,
        merger=merger,
        metrics=metrics,
        train_time=train_time,
        raw_data=raw_data,
        data_path=data_path,
        raw_model=raw_model,
        model_path=model_path,
        validation_report=validation_report,
        archived=archived,
        best_weight=best_weight,
        last_weight=last_weight,
        log_path=log_path,
        visualization_path=visualization_path,
        audit_path=audit_path,
        output_dir=output_dir,
    )
    try:
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(render_training_report_markdown(payload), encoding="utf-8")
    except OSError:
        return None, None
    return json_path, markdown_path


def build_training_report_payload(
    *,
    run_context: TrainingRunContext,
    config: Any,
    merger: Any,
    metrics: TrainMetrics,
    train_time: float,
    raw_data: str | Path | None,
    data_path: Path,
    raw_model: str | Path | None,
    model_path: Path,
    validation_report: Any | None,
    archived: dict[str, Path],
    best_weight: Path | None,
    last_weight: Path | None,
    log_path: Path | None,
    visualization_path: Path | None,
    audit_path: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Build a serializable report answering who/when/data/quality/result."""

    return {
        "schema_version": 1,
        "kind": "training_report",
        "run_id": run_context.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": run_context.started_at.isoformat(timespec="seconds"),
        "duration_seconds": round(train_time, 3),
        "operator": _plain(
            {
                "name": run_context.operator,
                "role": run_context.operator_role,
                "notes": run_context.notes,
            }
        ),
        "dataset": _plain(
            {
                "declared": str(raw_data) if raw_data is not None else None,
                "yaml_path": str(data_path),
                "label": run_context.dataset_label,
                "quality": _validation_summary(validation_report),
            }
        ),
        "training": _plain(
            {
                "task": run_context.task,
                "model_declared": str(raw_model) if raw_model is not None else None,
                "model_path": str(model_path),
                "model_label": run_context.model_label,
                "output_dir": str(output_dir),
                "config": _call_or_empty(config, "to_audit_snapshot"),
                "merger": _call_or_empty(merger, "to_audit_log"),
            }
        ),
        "results": _plain(
            {
                "metrics": metrics.to_dict(),
                "best_weight": str(best_weight) if best_weight else None,
                "last_weight": str(last_weight) if last_weight else None,
                "best_archive": str(archived.get("best")) if archived.get("best") else None,
                "last_archive": str(archived.get("last")) if archived.get("last") else None,
                "log_path": str(log_path) if log_path else None,
                "visualization_path": str(visualization_path) if visualization_path else None,
                "audit_path": str(audit_path) if audit_path else None,
            }
        ),
    }


def render_training_report_markdown(payload: dict[str, Any]) -> str:
    dataset = payload["dataset"]
    quality = dataset.get("quality") or {}
    training = payload["training"]
    operator = payload["operator"]
    results = payload["results"]
    overall = results.get("metrics", {}).get("overall", {})

    lines = [
        "# ODPlatform Training Report",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Started At: `{payload['started_at']}`",
        f"- Duration: `{payload['duration_seconds']} sec`",
        f"- Operator: `{operator.get('name') or 'N/A'}`",
        f"- Role: `{operator.get('role') or 'N/A'}`",
        "",
        "## Dataset",
        "",
        f"- Declared: `{dataset.get('declared')}`",
        f"- YAML: `{dataset.get('yaml_path')}`",
        f"- Quality: `{quality.get('overall_severity', 'N/A')}` "
        f"(exit_code={quality.get('exit_code', 'N/A')})",
        f"- Quality Report: `{quality.get('report_path') or 'N/A'}`",
        "",
        "## Training",
        "",
        f"- Task: `{training.get('task')}`",
        f"- Model: `{training.get('model_declared')}`",
        f"- Model Path: `{training.get('model_path')}`",
        f"- Output Dir: `{training.get('output_dir')}`",
        "",
        "## Results",
        "",
        f"- Fitness: `{overall.get('fitness')}`",
        f"- mAP50(B): `{overall.get('metrics/mAP50(B)')}`",
        f"- mAP50-95(B): `{overall.get('metrics/mAP50-95(B)')}`",
        f"- Best Weight: `{results.get('best_weight')}`",
        f"- Last Weight: `{results.get('last_weight')}`",
        f"- Training Chart: `{results.get('visualization_path') or 'N/A'}`",
        f"- Log: `{results.get('log_path') or 'N/A'}`",
    ]
    notes = operator.get("notes")
    if notes:
        lines.extend(["", "## Notes", "", notes])
    return "\n".join(lines) + "\n"


def _validation_summary(report: Any | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return _plain(
        {
            "run_id": getattr(report, "run_id", None),
            "operation": getattr(report, "operation", None),
            "overall_severity": getattr(report, "overall_severity", None),
            "exit_code": getattr(report, "exit_code", None),
            "severity_counts": _call_or_empty(report, "severity_counts"),
            "report_path": _path_str(getattr(report, "report_path", None)),
            "markdown_path": _path_str(getattr(report, "markdown_path", None)),
            "html_path": _path_str(getattr(report, "html_path", None)),
            "word_path": _path_str(getattr(report, "word_path", None)),
        }
    )


def _call_or_empty(obj: Any, method_name: str) -> Any:
    method = getattr(obj, method_name, None)
    if method is None:
        return {}
    try:
        return _plain(method())
    except Exception:
        return {}


def _path_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(child) for child in value]
    return str(value)
