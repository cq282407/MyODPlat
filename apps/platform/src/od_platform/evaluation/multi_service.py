#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : multi_service.py
# @Project   : ODPlatform
# @Function  : Multi-model evaluation orchestration service.
from __future__ import annotations

import csv
import json
import logging
import math
from argparse import Namespace
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from od_platform.common.paths import RUNS_DIR
from od_platform.common.refs import resolve_trained_model
from od_platform.evaluation.service import ValMetrics, ValResult, ValService
from od_platform.runtime_config import build_val_config

logger = logging.getLogger(__name__)

DEFAULT_SORT_BY = "metrics/mAP50-95(B)"
_FALLBACK_SORT_KEYS: tuple[str, ...] = (
    DEFAULT_SORT_BY,
    "metrics/mAP50-95(M)",
    "fitness",
)
_SUMMARY_COLUMNS: tuple[str, ...] = (
    "rank",
    "model",
    "model_path",
    "child_run_name",
    "success",
    "sort_by",
    "sort_value",
    "fitness",
    "precision_b",
    "recall_b",
    "map50_b",
    "map50_95_b",
    "precision_m",
    "recall_m",
    "map50_m",
    "map50_95_m",
    "output_dir",
    "audit_path",
    "log_path",
    "error",
)
_MULTI_CONTROL_FIELDS = {
    "model",
    "models",
    "name",
    "project",
    "run_name",
    "sort_by",
    "yaml",
    "academic_plots",
    "log_level",
    "rename_log",
}


@dataclass(frozen=True)
class MultiValModelResult:
    model: str
    model_path: Path
    child_run_name: str
    success: bool
    rank: int | None = None
    sort_value: float | None = None
    output_dir: Path | None = None
    metrics: ValMetrics | None = None
    error: str | None = None
    audit_path: Path | None = None
    log_path: Path | None = None

    def to_summary_row(self, *, sort_by: str) -> dict[str, Any]:
        metrics_summary = _metrics_summary(self.metrics)
        return {
            "rank": self.rank,
            "model": self.model,
            "model_path": str(self.model_path),
            "child_run_name": self.child_run_name,
            "success": self.success,
            "sort_by": sort_by,
            "sort_value": _clean_number(self.sort_value),
            "fitness": metrics_summary["fitness"],
            "precision_b": metrics_summary["precision_b"],
            "recall_b": metrics_summary["recall_b"],
            "map50_b": metrics_summary["map50_b"],
            "map50_95_b": metrics_summary["map50_95_b"],
            "precision_m": metrics_summary["precision_m"],
            "recall_m": metrics_summary["recall_m"],
            "map50_m": metrics_summary["map50_m"],
            "map50_95_m": metrics_summary["map50_95_m"],
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "audit_path": str(self.audit_path) if self.audit_path else None,
            "log_path": str(self.log_path) if self.log_path else None,
            "error": self.error,
        }

    def to_dict(self, *, sort_by: str) -> dict[str, Any]:
        payload = self.to_summary_row(sort_by=sort_by)
        payload["metrics"] = self.metrics.to_dict() if self.metrics is not None else None
        return payload


@dataclass(frozen=True)
class MultiValResult:
    success: bool
    output_dir: Path | None
    results: list[MultiValModelResult] = field(default_factory=list)
    best_model: str | None = None
    best_score: float | None = None
    summary_csv_path: Path | None = None
    summary_json_path: Path | None = None
    summary_md_path: Path | None = None
    error: str | None = None
    log_path: Path | None = None


class MultiValService:
    """Sequential multi-model evaluation that reuses ValService."""

    def __init__(self, val_service: ValService | None = None) -> None:
        self._val_service = val_service or ValService()

    def evaluate_many(
        self,
        models: list[str | Path] | tuple[str | Path, ...],
        yaml_path: str | Path | None = None,
        data: str | Path | None = None,
        cli_args: dict[str, Any] | Namespace | None = None,
        *,
        run_name: str | None = None,
        sort_by: str = DEFAULT_SORT_BY,
        rename_inner_logs: bool = False,
    ) -> MultiValResult:
        model_refs = [str(model).strip() for model in models if str(model).strip()]
        if not model_refs:
            return MultiValResult(
                success=False,
                output_dir=None,
                error="At least one model is required for multi evaluation.",
                log_path=_find_project_log_path(),
            )

        output_dir: Path | None = None
        try:
            base_cli_args = _merge_cli_args(cli_args, data=data)
            config, _ = build_val_config(
                yaml_path=yaml_path or "val.yaml",
                cli_args=base_cli_args,
            )
            assert config is not None

            task = getattr(config, "task", "detect")
            output_dir = _resolve_output_dir(
                RUNS_DIR / f"{task}_val_compare",
                run_name or "compare",
                exist_ok=False,
            )
            logger.info("=" * 60)
            logger.info("Starting multi-model evaluation (task=%s)", task)
            logger.info("Output directory: %s", output_dir)
            logger.info("Model count: %s", len(model_refs))
            logger.info("=" * 60)

            raw_results: list[MultiValModelResult] = []
            used_child_names: set[str] = set()
            for index, model_ref in enumerate(model_refs, start=1):
                child_run_name = _make_child_run_name(model_ref, index=index, used=used_child_names)
                per_model_cli_args = dict(base_cli_args or {})
                per_model_cli_args["model"] = model_ref
                per_model_cli_args["project"] = str(output_dir)
                per_model_cli_args["name"] = child_run_name

                logger.info("[%s/%s] Evaluating model: %s", index, len(model_refs), model_ref)
                try:
                    val_result = self._val_service.evaluate(
                        yaml_path=yaml_path,
                        cli_args=per_model_cli_args,
                        rename_log=rename_inner_logs,
                    )
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.error("Unexpected multi-eval inner failure: %s", exc, exc_info=True)
                    val_result = ValResult(
                        success=False,
                        output_dir=None,
                        error=str(exc),
                        log_path=_find_project_log_path(),
                    )

                model_result = MultiValModelResult(
                    model=model_ref,
                    model_path=resolve_trained_model(model_ref),
                    child_run_name=child_run_name,
                    success=val_result.success,
                    sort_value=_extract_sort_value(val_result.metrics, sort_by),
                    output_dir=val_result.output_dir,
                    metrics=val_result.metrics,
                    error=val_result.error,
                    audit_path=val_result.audit_path,
                    log_path=val_result.log_path,
                )
                raw_results.append(model_result)
                if model_result.success:
                    logger.info(
                        "Model finished: %s | score=%s | output=%s",
                        model_result.model,
                        _format_float(model_result.sort_value),
                        model_result.output_dir,
                    )
                else:
                    logger.warning("Model failed but batch continues: %s | %s", model_result.model, model_result.error)

            ranked_results = _rank_results(raw_results)
            summary_csv_path, summary_json_path, summary_md_path = _write_summary_files(
                output_dir=output_dir,
                results=ranked_results,
                sort_by=sort_by,
            )
            best_result = next((result for result in ranked_results if result.success), None)
            if best_result is not None:
                logger.info(
                    "Best model: %s | %s=%s",
                    best_result.model,
                    sort_by,
                    _format_float(best_result.sort_value),
                )
            else:
                logger.warning("No successful model evaluation was produced in this batch.")

            return MultiValResult(
                success=best_result is not None,
                output_dir=output_dir,
                results=ranked_results,
                best_model=best_result.model if best_result else None,
                best_score=best_result.sort_value if best_result else None,
                summary_csv_path=summary_csv_path,
                summary_json_path=summary_json_path,
                summary_md_path=summary_md_path,
                log_path=_find_project_log_path(),
            )
        except Exception as exc:
            logger.error("Multi-model evaluation failed: %s", exc, exc_info=True)
            return MultiValResult(
                success=False,
                output_dir=output_dir,
                error=str(exc),
                log_path=_find_project_log_path(),
            )


def evaluate_yolo_multi(
    models: list[str | Path] | tuple[str | Path, ...],
    yaml_path: str | Path | None = None,
    data: str | Path | None = None,
    cli_args: dict[str, Any] | Namespace | None = None,
    *,
    run_name: str | None = None,
    sort_by: str = DEFAULT_SORT_BY,
    rename_inner_logs: bool = False,
) -> MultiValResult:
    return MultiValService().evaluate_many(
        models=models,
        yaml_path=yaml_path,
        data=data,
        cli_args=cli_args,
        run_name=run_name,
        sort_by=sort_by,
        rename_inner_logs=rename_inner_logs,
    )


def _merge_cli_args(
    cli_args: dict[str, Any] | Namespace | None,
    *,
    data: str | Path | None,
) -> dict[str, Any] | None:
    if cli_args is None and data is None:
        return None
    if cli_args is None:
        raw: dict[str, Any] = {}
    elif isinstance(cli_args, Namespace):
        raw = vars(cli_args).copy()
    else:
        raw = dict(cli_args)

    payload = {
        key: value
        for key, value in raw.items()
        if key not in _MULTI_CONTROL_FIELDS and value is not None
    }
    if data is not None:
        payload["data"] = str(data)
    return payload or None


def _resolve_output_dir(base: Path, name: str, *, exist_ok: bool) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    candidate = base / name
    if exist_ok or not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    index = 2
    while (base / f"{name}{index}").exists():
        index += 1
    output_dir = base / f"{name}{index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _make_child_run_name(model_ref: str, *, index: int, used: set[str]) -> str:
    stem = Path(model_ref).stem or Path(model_ref).name or f"model_{index}"
    base = f"{index:03d}_{_safe_name(stem)}"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "_-" else "_" for char in value)
    return safe.strip("_") or "model"


def _extract_sort_value(metrics: ValMetrics | None, sort_by: str) -> float | None:
    if metrics is None:
        return None
    candidates = [sort_by]
    candidates.extend(key for key in _FALLBACK_SORT_KEYS if key not in candidates)
    for key in candidates:
        value = _clean_number(metrics.overall.get(key))
        if value is not None:
            return value
    return None


def _clean_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _sortable_score(value: float | None) -> float:
    return value if value is not None else float("-inf")


def _rank_results(results: list[MultiValModelResult]) -> list[MultiValModelResult]:
    indexed = list(enumerate(results))
    indexed.sort(
        key=lambda pair: (
            0 if pair[1].success else 1,
            -_sortable_score(pair[1].sort_value),
            pair[0],
        )
    )

    ranked: list[MultiValModelResult] = []
    rank = 1
    for _, result in indexed:
        current_rank = rank if result.success else None
        ranked.append(replace(result, rank=current_rank))
        if result.success:
            rank += 1
    return ranked


def _metrics_summary(metrics: ValMetrics | None) -> dict[str, float | None]:
    overall = metrics.overall if metrics is not None else {}
    return {
        "fitness": _clean_number(overall.get("fitness")),
        "precision_b": _clean_number(overall.get("metrics/precision(B)")),
        "recall_b": _clean_number(overall.get("metrics/recall(B)")),
        "map50_b": _clean_number(overall.get("metrics/mAP50(B)")),
        "map50_95_b": _clean_number(overall.get("metrics/mAP50-95(B)")),
        "precision_m": _clean_number(overall.get("metrics/precision(M)")),
        "recall_m": _clean_number(overall.get("metrics/recall(M)")),
        "map50_m": _clean_number(overall.get("metrics/mAP50(M)")),
        "map50_95_m": _clean_number(overall.get("metrics/mAP50-95(M)")),
    }


def _write_summary_files(
    *,
    output_dir: Path,
    results: list[MultiValModelResult],
    sort_by: str,
) -> tuple[Path | None, Path | None, Path | None]:
    csv_path = _write_summary_csv(output_dir=output_dir, results=results, sort_by=sort_by)
    json_path = _write_summary_json(output_dir=output_dir, results=results, sort_by=sort_by)
    md_path = _write_summary_md(output_dir=output_dir, results=results, sort_by=sort_by)
    return csv_path, json_path, md_path


def _write_summary_csv(
    *,
    output_dir: Path,
    results: list[MultiValModelResult],
    sort_by: str,
) -> Path | None:
    summary_path = output_dir / "summary.csv"
    try:
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_SUMMARY_COLUMNS))
            writer.writeheader()
            for result in results:
                writer.writerow(result.to_summary_row(sort_by=sort_by))
    except OSError as exc:
        logger.warning("Failed to write summary.csv: %s", exc)
        return None
    logger.info("Summary CSV: %s", summary_path)
    return summary_path


def _write_summary_json(
    *,
    output_dir: Path,
    results: list[MultiValModelResult],
    sort_by: str,
) -> Path | None:
    summary_path = output_dir / "summary.json"
    best_result = next((result for result in results if result.success), None)
    payload = {
        "schema_version": 1,
        "kind": "val_multi",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sort_by": sort_by,
        "best_model": best_result.model if best_result else None,
        "best_score": _clean_number(best_result.sort_value) if best_result else None,
        "total_models": len(results),
        "success_count": sum(1 for result in results if result.success),
        "failure_count": sum(1 for result in results if not result.success),
        "results": [result.to_dict(sort_by=sort_by) for result in results],
    }
    try:
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to write summary.json: %s", exc)
        return None
    logger.info("Summary JSON: %s", summary_path)
    return summary_path


def _write_summary_md(
    *,
    output_dir: Path,
    results: list[MultiValModelResult],
    sort_by: str,
) -> Path | None:
    summary_path = output_dir / "summary.md"
    best_result = next((result for result in results if result.success), None)
    lines = [
        "# Multi-Model Evaluation Summary",
        "",
        f"- Created at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Sort by: `{sort_by}`",
        f"- Total models: {len(results)}",
        f"- Successes: {sum(1 for result in results if result.success)}",
        f"- Failures: {sum(1 for result in results if not result.success)}",
    ]
    if best_result is not None:
        lines.append(
            f"- Best model: `{best_result.model}` ({sort_by}={_format_float(best_result.sort_value)})"
        )
    else:
        lines.append("- Best model: none")
    lines.extend(
        [
            "",
            "| Rank | Model | Success | mAP50-95(B) | mAP50(B) | Precision(B) | Recall(B) | Output | Error |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in results:
        metrics_summary = _metrics_summary(result.metrics)
        lines.append(
            "| {rank} | `{model}` | {success} | {map50_95} | {map50} | {precision} | {recall} | {output_dir} | {error} |".format(
                rank=result.rank if result.rank is not None else "-",
                model=result.model,
                success="yes" if result.success else "no",
                map50_95=_format_float(metrics_summary["map50_95_b"]),
                map50=_format_float(metrics_summary["map50_b"]),
                precision=_format_float(metrics_summary["precision_b"]),
                recall=_format_float(metrics_summary["recall_b"]),
                output_dir=f"`{result.output_dir}`" if result.output_dir else "-",
                error=result.error or "-",
            )
        )
    try:
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to write summary.md: %s", exc)
        return None
    logger.info("Summary Markdown: %s", summary_path)
    return summary_path


def _find_project_log_path() -> Path | None:
    root = logging.getLogger("od_platform")
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


def _format_float(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"
