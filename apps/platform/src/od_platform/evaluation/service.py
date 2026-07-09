#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : service.py
# @Project   : ODPlatform
# @Function  : Evaluation orchestration service.
from __future__ import annotations

import json
import logging
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from od_platform.common.config_log import log_effective_config, log_override_chains
from od_platform.common.dataset_path import prepare_ultralytics_dataset_yaml, resolve_dataset_path
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
    """One evaluation run result snapshot."""

    success: bool
    output_dir: Path | None
    metrics: ValMetrics | None = None
    error: str | None = None
    audit_path: Path | None = None
    log_path: Path | None = None


class ValService:
    """YOLO evaluation orchestrator. It wires D3/D5/D6/ultralytics together."""

    def evaluate(
        self,
        yaml_path: str | Path | None = None,
        model: str | Path | None = None,
        data: str | Path | None = None,
        cli_args: dict[str, Any] | Namespace | None = None,
        *,
        rename_log: bool = True,
    ) -> ValResult:
        """Run one evaluation workflow and never raise across the service boundary."""

        try:
            merged_cli_args = _merge_cli_args(cli_args, model=model, data=data)
            config, merger = build_val_config(
                yaml_path=yaml_path or "val.yaml",
                cli_args=merged_cli_args,
            )
            assert config is not None

            raw_model = config.model
            if not raw_model:
                return ValResult(
                    success=False,
                    output_dir=None,
                    error="评估模型 model 不能为空; 请在 val.yaml 或 --model 中指定已训练权重",
                    log_path=_find_project_log_path(),
                )

            raw_data = config.data
            logger.info("=" * 60)
            logger.info("开始 YOLO 模型评估 (task=%s)", config.task)
            logger.info("=" * 60)
            logger.info("任务类型: %s", config.task)
            logger.info("数据集声明: %s", raw_data)
            data_path = resolve_dataset_path(raw_data)
            logger.info("数据集路径: %s", data_path)
            logger.info("模型声明: %s", raw_model)
            model_path = resolve_trained_model(raw_model)
            logger.info("模型路径: %s", model_path)

            if not model_path.exists():
                return ValResult(
                    success=False,
                    output_dir=None,
                    error=f"找不到已训练权重: {model_path}。请确认名字正确，并且已被 D6 归档到 models/trained/",
                    log_path=_find_project_log_path(),
                )

            log_device_info(target_logger=logger)
            log_effective_config(config, merger, logger=logger)
            log_override_chains(config, merger, logger=logger)

            ultra_data_path = prepare_ultralytics_dataset_yaml(data_path)
            logger.info("Ultralytics 数据集配置: %s", ultra_data_path)

            yolo_kwargs = config.to_ultralytics_kwargs()
            yolo_kwargs.pop("model", None)
            yolo_kwargs["data"] = str(ultra_data_path)
            yolo_kwargs.setdefault("project", str(RUNS_DIR / f"{config.task}_val"))
            if config.experiment_name and not yolo_kwargs.get("name"):
                yolo_kwargs["name"] = config.experiment_name

            logger.info("=" * 60)
            logger.info("启动评估")
            logger.info("输出目录(project): %s", yolo_kwargs["project"])
            logger.info("=" * 60)

            yolo_results = self._run_evaluation(model_path, yolo_kwargs)
            output_dir = Path(getattr(yolo_results, "save_dir", "unknown"))

            logger.info("=" * 60)
            logger.info("评估完成")
            logger.info("=" * 60)
            metrics = ValMetrics.from_yolo_results(
                yolo_results,
                model_trainer=getattr(getattr(yolo_results, "model", None), "trainer", None),
            )
            log_train_metrics(metrics, logger=logger, title="评估结果")

            log_path = _find_project_log_path()
            if rename_log and output_dir != Path("unknown"):
                renamed = rename_log_to_save_dir(output_dir, Path(raw_model).stem)
                if renamed is not None:
                    log_path = renamed

            audit_path = _write_audit(
                output_dir=output_dir,
                config=config,
                merger=merger,
                metrics=metrics,
                model_path=model_path,
                data_path=data_path,
                log_path=log_path,
            )

            logger.info("输出目录: %s", output_dir)
            logger.info("评估模型: %s", model_path)
            logger.info("评估数据: %s", data_path)

            return ValResult(
                success=True,
                output_dir=output_dir if output_dir != Path("unknown") else None,
                metrics=metrics,
                audit_path=audit_path,
                log_path=log_path,
            )
        except Exception as exc:
            logger.error("评估失败: %s", exc, exc_info=True)
            return ValResult(
                success=False,
                output_dir=None,
                error=str(exc),
                log_path=_find_project_log_path(),
            )

    @staticmethod
    def _run_evaluation(model_path: Path, yolo_kwargs: dict[str, Any]) -> Any:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        return model.val(**dict(yolo_kwargs))


def evaluate_yolo(
    yaml_path: str | Path | None = None,
    model: str | Path | None = None,
    data: str | Path | None = None,
    cli_args: dict[str, Any] | Namespace | None = None,
    *,
    rename_log: bool = True,
) -> ValResult:
    """Convenience wrapper around ValService().evaluate()."""

    return ValService().evaluate(
        yaml_path=yaml_path,
        model=model,
        data=data,
        cli_args=cli_args,
        rename_log=rename_log,
    )


def _merge_cli_args(
    cli_args: dict[str, Any] | Namespace | None,
    *,
    model: str | Path | None,
    data: str | Path | None,
) -> Namespace | None:
    if cli_args is None and model is None and data is None:
        return None
    if cli_args is None:
        payload: dict[str, Any] = {}
    elif isinstance(cli_args, Namespace):
        payload = vars(cli_args).copy()
    else:
        payload = dict(cli_args)
    if model is not None:
        payload["model"] = str(model)
    if data is not None:
        payload["data"] = str(data)
    return Namespace(**payload)


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
    metrics: ValMetrics,
    model_path: Path,
    data_path: Path,
    log_path: Path | None,
) -> Path | None:
    if output_dir == Path("unknown"):
        logger.warning("评估 save_dir 未知, 跳过写审计快照")
        return None

    audit_path = output_dir / "odp_audit.json"
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
            "log_path": str(log_path) if log_path else None,
        },
    }
    try:
        audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("写审计快照失败: %s", exc)
        return None
    logger.info("审计快照: %s", audit_path)
    return audit_path
