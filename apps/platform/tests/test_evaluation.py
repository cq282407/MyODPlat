#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_evaluation.py
# @Project   : ODPlatform
# @Function  : D7 evaluation service and CLI tests.
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from od_platform.cli.evaluate_model import build_parser, main as val_main
from od_platform.common.result import TrainMetrics
from od_platform.evaluation import ValMetrics
from od_platform.evaluation.service import ValResult, ValService

S = "od_platform.evaluation.service"


def test_val_metrics_is_train_metrics_alias() -> None:
    assert ValMetrics is TrainMetrics


def test_val_result_has_no_best_weight() -> None:
    result = ValResult(success=False, save_dir=None, metrics=None)

    assert not hasattr(result, "best_weight")


def test_val_service_success_writes_audit_and_returns_metrics(tmp_path: Path) -> None:
    cfg, merger = _fake_config()
    weight = _fake_weight(tmp_path)
    yolo_results = _fake_yolo_results(tmp_path)

    with patch(f"{S}.build_val_config", return_value=(cfg, merger)), \
         patch(f"{S}.resolve_trained_model", return_value=weight), \
         patch(f"{S}.resolve_dataset_path", return_value=tmp_path / "dataset.yaml"), \
         patch(f"{S}.log_device_info"), \
         patch(f"{S}.log_effective_config"), \
         patch(f"{S}.log_override_chains"), \
         patch(f"{S}.rename_log_to_save_dir", return_value=tmp_path / "val.log") as rename, \
         patch.object(ValService, "_run_eval", return_value=yolo_results) as run:
        result = ValService().evaluate("val.yaml", "train3-best.pt", "sample.yaml")

    assert result.success is True
    assert result.save_dir == yolo_results.save_dir
    assert result.metrics is not None
    assert result.metrics.task == "detect"
    assert result.metrics.overall["metrics/mAP50(B)"] == 0.8
    assert result.audit_path is not None
    assert result.audit_path.exists()
    payload = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "val"
    assert payload["result_summary"]["model_path"] == str(weight)
    assert "model" not in run.call_args.args[1]
    rename.assert_called_once()


def test_val_service_missing_weight_fails_before_yolo(tmp_path: Path) -> None:
    cfg, merger = _fake_config()
    missing = tmp_path / "models" / "trained" / "missing.pt"

    with patch(f"{S}.build_val_config", return_value=(cfg, merger)), \
         patch(f"{S}.resolve_trained_model", return_value=missing), \
         patch.object(ValService, "_run_eval") as run:
        result = ValService().evaluate("val.yaml", "missing.pt", "sample.yaml")

    assert result.success is False
    assert "Trained weight not found" in (result.error or "")
    run.assert_not_called()


def test_val_service_exception_returns_failed_result(tmp_path: Path) -> None:
    cfg, merger = _fake_config()
    weight = _fake_weight(tmp_path)

    with patch(f"{S}.build_val_config", return_value=(cfg, merger)), \
         patch(f"{S}.resolve_trained_model", return_value=weight), \
         patch(f"{S}.resolve_dataset_path", return_value=tmp_path / "dataset.yaml"), \
         patch(f"{S}.log_device_info"), \
         patch(f"{S}.log_effective_config"), \
         patch(f"{S}.log_override_chains"), \
         patch.object(ValService, "_run_eval", side_effect=RuntimeError("boom")):
        result = ValService().evaluate("val.yaml", "train3-best.pt", "sample.yaml")

    assert result.success is False
    assert "boom" in (result.error or "")


def test_evaluation_module_keeps_expected_boundaries() -> None:
    service_text = Path("apps/platform/src/od_platform/evaluation/service.py").read_text(encoding="utf-8")

    assert "from od_platform.training" not in service_text
    assert "addHandler" not in service_text
    assert "setLevel" not in service_text


def test_val_cli_parser_accepts_expected_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "val.yaml",
            "--model",
            "train3-best.pt",
            "--data",
            "sample.yaml",
            "--batch",
            "0.5",
            "--split",
            "test",
            "--no-plots",
        ]
    )

    assert args.config == "val.yaml"
    assert args.model == "train3-best.pt"
    assert args.data == "sample.yaml"
    assert args.batch == 0.5
    assert args.split == "test"
    assert args.plots is False


def test_val_cli_returns_success_with_mocked_service() -> None:
    fake_result = MagicMock(success=True, save_dir=Path("runs/val"), val_time=1.0)

    with patch("od_platform.cli.evaluate_model.get_logger"), \
         patch("od_platform.cli.evaluate_model.ValService") as service_cls:
        service_cls.return_value.evaluate.return_value = fake_result
        exit_code = val_main(["--model", "train3-best.pt", "--data", "sample.yaml", "--imgsz", "320"])

    assert exit_code == 0
    service_cls.return_value.evaluate.assert_called_once()
    call = service_cls.return_value.evaluate.call_args
    assert call.kwargs["model"] == "train3-best.pt"
    assert call.kwargs["data"] == "sample.yaml"
    assert call.kwargs["cli_overrides"]["imgsz"] == 320


def _fake_config():
    cfg = MagicMock()
    cfg.task = "detect"
    cfg.model = "train3-best.pt"
    cfg.data = "sample.yaml"
    cfg.experiment_name = None
    cfg.to_ultralytics_kwargs.return_value = {}
    cfg.to_audit_snapshot.return_value = {"values": {"task": "detect"}}
    merger = MagicMock()
    merger.to_audit_log.return_value = {"fields": {}}
    return cfg, merger


def _fake_weight(root: Path) -> Path:
    weight = root / "models" / "trained" / "train3-best.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weight")
    return weight


def _fake_yolo_results(root: Path):
    save_dir = root / "runs" / "detect_val" / "val"
    save_dir.mkdir(parents=True)
    result = MagicMock()
    result.task = "unknown"
    result.save_dir = save_dir
    result.fitness = 0.75
    result.speed = {}
    result.results_dict = {
        "metrics/precision(B)": 0.7,
        "metrics/recall(B)": 0.6,
        "metrics/mAP50(B)": 0.8,
        "metrics/mAP50-95(B)": 0.5,
        "fitness": 0.75,
    }
    result.names = {}
    result.maps = []
    return result
