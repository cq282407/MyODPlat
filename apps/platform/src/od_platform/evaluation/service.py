#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : service.py
# @Project   : ODPlatform
# @Function  : ValService orchestration.
from __future__ import annotations

import json
import logging
from argparse import Namespace
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from od_platform.common.config_log import log_effective_config, log_override_chains
from od_platform.common.dataset_path import resolve_dataset_path
from od_platform.common.log_rename import rename_log_to_save_dir
from od_platform.common.paths import RUNS_DIR
from od_platform.common.refs import resolve_trained_model
from od_platform.common.result import TrainMetrics, log_train_metrics
from od_platform.common.system_utils import log_device_info
from od_platform.runtime_config import build_val_config

logger = logging.getLogger(__name__)

ValMetrics = TrainMetrics


@dataclass(frozen=True)
class ValResult:
    """One model evaluation result snapshot."""

    success: bool
    save_dir: Path | None
    metrics: ValMetrics | None
    error: str | None = None
    audit_path: Path | None = None
    log_path: Path | None = None
    val_time: float | None = None


class ValService:
    """YOLO evaluation orchestrator. It wires D2/D3/D5/ultralytics together."""

    def evaluate(
        self,
        config_path: str | Path | None,
        model: str | Path,
        data: str | Path,
        *,
        cli_overrides: dict[str, Any] | Namespace | None = None,
        rename_log: bool = True,
    ) -> ValResult:
        """Run one evaluation workflow and never raise across the service boundary."""

        start = datetime.now()
        try:
            return self._evaluate(
                config_path or "val.yaml",
                model,
                data,
                cli_overrides=cli_overrides,
                rename_log=rename_log,
                start=start,
            )
        except Exception as exc:
            logger.error("Evaluation failed: %s", exc, exc_info=True)
            return ValResult(
                success=False,
                save_dir=None,
                metrics=None,
                error=str(exc),
                log_path=_find_project_log_path(),
                val_time=(datetime.now() - start).total_seconds(),
            )

    def _evaluate(
        self,
        config_path: str | Path,
        model: str | Path,
        data: str | Path,
        *,
        cli_overrides: dict[str, Any] | Namespace | None,
        rename_log: bool,
        start: datetime,
    ) -> ValResult:
        merged_cli = _merge_cli_overrides(cli_overrides, model=model, data=data)
        config, merger = build_val_config(
            yaml_path=config_path,
            cli_args=_as_namespace(merged_cli),
        )
        assert config is not None

        logger.info("=" * 60)
        logger.info("Start YOLO evaluation (task=%s)", config.task)
        logger.info("=" * 60)

        raw_model = config.model or str(model)
        raw_data = config.data or str(data)
        weight = resolve_trained_model(raw_model)
        if not weight.exists():
            message = f"Trained weight not found: {weight}"
            logger.error(message)
            return ValResult(
                success=False,
                save_dir=None,
                metrics=None,
                error=message,
                log_path=_find_project_log_path(),
                val_time=(datetime.now() - start).total_seconds(),
            )

        data_path = resolve_dataset_path(raw_data)
        logger.info("Task: %s", config.task)
        logger.info("Dataset config: %s", data_path)
        logger.info("Trained weight: %s", weight)

        log_device_info(target_logger=logger)
        log_effective_config(config, merger, logger=logger)
        log_override_chains(config, merger, logger=logger)

        yolo_kwargs = config.to_ultralytics_kwargs()
        yolo_kwargs.pop("model", None)
        yolo_kwargs["data"] = str(data_path)
        yolo_kwargs.setdefault("project", str(RUNS_DIR / f"{config.task}_val"))
        if config.experiment_name and not yolo_kwargs.get("name"):
            yolo_kwargs["name"] = config.experiment_name

        logger.info("=" * 60)
        logger.info("Run evaluation")
        logger.info("Output root(project): %s", yolo_kwargs["project"])
        logger.info("=" * 60)

        yolo_results = self._run_eval(weight, yolo_kwargs)
        save_dir = _extract_save_dir(yolo_results)
        metrics = replace(ValMetrics.from_yolo_results(yolo_results), task=str(config.task))
        log_train_metrics(metrics, logger=logger)

        if rename_log and save_dir is not None:
            rename_log_to_save_dir(save_dir, weight.stem)

        val_time = (datetime.now() - start).total_seconds()
        log_path = _find_project_log_path()
        audit_path = _write_audit(
            save_dir=save_dir,
            config=config,
            merger=merger,
            metrics=metrics,
            val_time=val_time,
            model_path=weight,
            data_path=data_path,
            log_path=log_path,
        )

        logger.info("Evaluation time: %.2f sec", val_time)
        if save_dir is not None:
            logger.info("Output dir: %s", save_dir)

        return ValResult(
            success=True,
            save_dir=save_dir,
            metrics=metrics,
            audit_path=audit_path,
            log_path=log_path,
            val_time=val_time,
        )

    @staticmethod
    def _run_eval(weight: Path, yolo_kwargs: dict[str, Any]) -> Any:
        from ultralytics import YOLO

        model = YOLO(str(weight))
        return model.val(**yolo_kwargs)


def evaluate_yolo(
    config_path: str | Path | None,
    model: str | Path,
    data: str | Path,
    *,
    cli_overrides: dict[str, Any] | Namespace | None = None,
    rename_log: bool = True,
) -> ValResult:
    """Convenience wrapper around ValService().evaluate()."""

    return ValService().evaluate(
        config_path,
        model,
        data,
        cli_overrides=cli_overrides,
        rename_log=rename_log,
    )


def _as_namespace(args: dict[str, Any] | Namespace | None) -> Namespace | None:
    if args is None or isinstance(args, Namespace):
        return args
    return Namespace(**args)


def _merge_cli_overrides(
    cli_overrides: dict[str, Any] | Namespace | None,
    *,
    model: str | Path,
    data: str | Path,
) -> dict[str, Any]:
    raw = vars(cli_overrides) if isinstance(cli_overrides, Namespace) else dict(cli_overrides or {})
    merged = {key: value for key, value in raw.items() if value is not None}
    merged["model"] = str(model)
    merged["data"] = str(data)
    return merged


def _extract_save_dir(results: Any) -> Path | None:
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        save_dir = getattr(getattr(results, "validator", None), "save_dir", None)
    return Path(save_dir) if save_dir else None


def _find_project_log_path() -> Path | None:
    root = logging.getLogger("od_platform")
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


def _write_audit(
    *,
    save_dir: Path | None,
    config: Any,
    merger: Any,
    metrics: ValMetrics,
    val_time: float,
    model_path: Path,
    data_path: Path,
    log_path: Path | None,
) -> Path | None:
    if save_dir is None:
        logger.warning("Evaluation output dir is unknown; skip audit snapshot")
        return None

    audit_path = save_dir / "odp_audit.json"
    payload = {
        "schema_version": 1,
        "kind": "val",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config.to_audit_snapshot(),
        "merger": merger.to_audit_log(),
        "metrics": metrics.to_dict(),
        "result_summary": {
            "model_path": str(model_path),
            "data_path": str(data_path),
            "val_time_sec": val_time,
            "log_path": str(log_path) if log_path else None,
        },
    }
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Write evaluation audit snapshot failed: %s", exc)
        return None
    logger.info("Evaluation audit snapshot: %s", audit_path)
    return audit_path
