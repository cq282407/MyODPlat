#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : service.py
# @Project   : ODPlatform
# @Function  : TrainService orchestration.
from __future__ import annotations

import json
import logging
from argparse import Namespace
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from od_platform.common.config_log import log_effective_config, log_override_chains
from od_platform.common.dataset_path import resolve_dataset_path
from od_platform.common.log_rename import rename_log_to_save_dir
from od_platform.common.model_path import resolve_model_path
from od_platform.common.paths import RUNS_DIR
from od_platform.common.result import TrainMetrics, log_train_metrics
from od_platform.common.system_utils import log_device_info
from od_platform.data_validation import validate_dataset
from od_platform.data_validation.render import render_to_logger
from od_platform.data_validation.registry import ValidationOptions
from od_platform.runtime_config import build_train_config
from od_platform.training.archive import archive_checkpoints
from od_platform.training.visualization import render_training_results_chart

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainResult:
    """One training run result snapshot."""

    success: bool
    output_dir: Path
    best_weight: Path | None = None
    last_weight: Path | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    train_time: float | None = None
    error: str | None = None
    audit_path: Path | None = None
    log_path: Path | None = None
    visualization_path: Path | None = None


class TrainService:
    """YOLO training orchestrator. It wires D2/D4/D5/ultralytics together."""

    def train(
        self,
        yaml_path: str | Path | None = None,
        cli_args: dict[str, Any] | Namespace | None = None,
        *,
        pre_validate: bool = True,
        archive: bool = True,
        rename_log: bool = True,
    ) -> TrainResult:
        """Run one training workflow and never raise across the service boundary."""

        start = datetime.now()
        output_dir: Path | None = None
        try:
            config, merger = build_train_config(
                yaml_path=yaml_path or "train.yaml",
                cli_args=_as_namespace(cli_args),
            )
            assert config is not None

            logger.info("=" * 60)
            logger.info("开始 YOLO 训练 (task=%s)", config.task)
            logger.info("=" * 60)

            raw_model = config.model or "yolo11n.pt"
            raw_data = config.data
            logger.info("任务类型: %s", config.task)
            logger.info("数据集声明: %s", raw_data)
            data_path = resolve_dataset_path(raw_data)
            logger.info("数据集路径: %s", data_path)
            logger.info("模型声明: %s", raw_model)
            model_path = resolve_model_path(raw_model)
            logger.info("模型路径: %s", model_path)

            log_device_info(target_logger=logger)
            log_effective_config(config, merger, logger=logger)
            log_override_chains(config, merger, logger=logger)

            if pre_validate:
                report = validate_dataset(
                    data_path,
                    options=ValidationOptions(operation="training_pre_validation"),
                )
                render_to_logger(report, logger)
                if report.exit_code >= 2:
                    error_count = sum(1 for result in report.results if result.severity == "ERROR")
                    return TrainResult(
                        success=False,
                        output_dir=Path("unknown"),
                        train_time=(datetime.now() - start).total_seconds(),
                        error=f"数据集校验失败 ({error_count} 个 ERROR), 训练未开始",
                        log_path=_find_project_log_path(),
                    )

            yolo_kwargs = config.to_ultralytics_kwargs()
            yolo_kwargs["data"] = str(data_path)
            yolo_kwargs["model"] = str(model_path)
            yolo_kwargs.setdefault("project", str(RUNS_DIR / f"{config.task}_train"))
            if config.experiment_name and not yolo_kwargs.get("name"):
                yolo_kwargs["name"] = config.experiment_name

            logger.info("=" * 60)
            logger.info("启动训练")
            logger.info("输出目录(project): %s", yolo_kwargs["project"])
            logger.info("=" * 60)

            yolo_results = self._run_training(model_path, yolo_kwargs)
            output_dir = Path(yolo_results.save_dir)

            logger.info("=" * 60)
            logger.info("训练完成")
            logger.info("=" * 60)
            metrics = TrainMetrics.from_yolo_results(
                yolo_results,
                model_trainer=getattr(getattr(yolo_results, "model", None), "trainer", None),
            )
            log_train_metrics(metrics, logger=logger)

            visualization_path = render_training_results_chart(output_dir, logger_=logger)

            model_stem = Path(raw_model).stem
            if rename_log:
                rename_log_to_save_dir(output_dir, model_stem)

            archived: dict[str, Path] = {}
            if archive:
                archived = archive_checkpoints(output_dir, raw_model)

            train_time = (datetime.now() - start).total_seconds()
            log_path = _find_project_log_path()
            audit_path = _write_audit(
                output_dir=output_dir,
                config=config,
                merger=merger,
                metrics=metrics,
                train_time=train_time,
                archived=archived,
                log_path=log_path,
                visualization_path=visualization_path,
            )

            best_weight = archived.get("best") or (output_dir / "weights" / "best.pt")
            last_weight = archived.get("last") or (output_dir / "weights" / "last.pt")
            logger.info("训练总耗时: %.2f 秒", train_time)
            logger.info("输出目录: %s", output_dir)
            logger.info("最佳权重: %s", best_weight)
            logger.info("最后权重: %s", last_weight)
            if visualization_path:
                logger.info("训练曲线: %s", visualization_path)

            return TrainResult(
                success=True,
                output_dir=output_dir,
                best_weight=best_weight if best_weight.exists() else None,
                last_weight=last_weight if last_weight.exists() else None,
                metrics=metrics.overall,
                train_time=train_time,
                audit_path=audit_path,
                log_path=log_path,
                visualization_path=visualization_path,
            )
        except Exception as exc:
            logger.error("训练失败: %s", exc, exc_info=True)
            return TrainResult(
                success=False,
                output_dir=output_dir or Path("unknown"),
                metrics={},
                train_time=(datetime.now() - start).total_seconds(),
                error=str(exc),
                log_path=_find_project_log_path(),
            )

    @staticmethod
    def _run_training(model_path: Path, yolo_kwargs: dict[str, Any]) -> Any:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        kwargs = dict(yolo_kwargs)
        kwargs.pop("model", None)
        return model.train(**kwargs)


def train_yolo(
    yaml_path: str | Path | None = None,
    cli_args: dict[str, Any] | Namespace | None = None,
    *,
    pre_validate: bool = True,
    archive: bool = True,
    rename_log: bool = True,
) -> TrainResult:
    """Convenience wrapper around TrainService().train()."""

    return TrainService().train(
        yaml_path=yaml_path,
        cli_args=cli_args,
        pre_validate=pre_validate,
        archive=archive,
        rename_log=rename_log,
    )


def _as_namespace(args: dict[str, Any] | Namespace | None) -> Namespace | None:
    if args is None or isinstance(args, Namespace):
        return args
    return Namespace(**args)


def _find_project_log_path() -> Path | None:
    root = logging.getLogger("od_platform")
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


def _write_audit(
    *,
    output_dir: Path,
    config: Any,
    merger: Any,
    metrics: TrainMetrics,
    train_time: float,
    archived: dict[str, Path],
    log_path: Path | None,
    visualization_path: Path | None,
) -> Path | None:
    audit_path = output_dir / "odp_audit.json"
    payload = {
        "schema_version": 1,
        "kind": "train",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config.to_audit_snapshot(),
        "merger": merger.to_audit_log(),
        "metrics": metrics.to_dict(),
        "result_summary": {
            "best_archive": str(archived.get("best", "")) or None,
            "last_archive": str(archived.get("last", "")) or None,
            "train_time_sec": train_time,
            "log_path": str(log_path) if log_path else None,
            "visualization_path": str(visualization_path) if visualization_path else None,
        },
    }
    try:
        audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("写审计快照失败: %s", exc)
        return None
    logger.info("审计快照: %s", audit_path)
    return audit_path
