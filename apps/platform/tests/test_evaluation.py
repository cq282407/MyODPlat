#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_evaluation.py
# @Project   : ODPlatform
# @Function  : D7 evaluation service and CLI tests.
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from od_platform.cli.evaluate_model import build_parser, main as eval_main
from od_platform.common.result import TrainMetrics
from od_platform.evaluation import ValMetrics
from od_platform.evaluation.service import ValService

S = "od_platform.evaluation.service"


def test_val_metrics_aliases_train_metrics() -> None:
    assert ValMetrics is TrainMetrics


def test_val_service_missing_weight_fails_fast(tmp_path: Path) -> None:
    cfg, merger = _fake_config()

    with patch(f"{S}.build_val_config", return_value=(cfg, merger)), \
         patch(f"{S}.resolve_dataset_path", return_value=tmp_path / "dataset.yaml"), \
         patch(f"{S}.resolve_trained_model", return_value=tmp_path / "missing.pt"), \
         patch(f"{S}.log_device_info"), \
         patch(f"{S}.log_effective_config"), \
         patch(f"{S}.log_override_chains"), \
         patch.object(ValService, "_run_evaluation") as run:
        result = ValService().evaluate("val.yaml")

    assert result.success is False
    assert "找不到已训练权重" in (result.error or "")
    run.assert_not_called()


def test_val_service_success_writes_audit_and_skips_archive(tmp_path: Path) -> None:
    cfg, merger = _fake_config()
    weight = tmp_path / "models" / "trained" / "train3-best.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weights")
    yolo_results = _fake_yolo_results(tmp_path)
    ultra_data_path = tmp_path / "dataset.runtime.yaml"

    with patch(f"{S}.build_val_config", return_value=(cfg, merger)), \
         patch(f"{S}.resolve_dataset_path", return_value=tmp_path / "dataset.yaml"), \
         patch(f"{S}.prepare_ultralytics_dataset_yaml", return_value=ultra_data_path) as prepare, \
         patch(f"{S}.resolve_trained_model", return_value=weight), \
         patch(f"{S}.log_device_info"), \
         patch(f"{S}.log_effective_config"), \
         patch(f"{S}.log_override_chains"), \
         patch(f"{S}.rename_log_to_save_dir", return_value=tmp_path / "val.log") as rename, \
         patch.object(ValService, "_run_evaluation", return_value=yolo_results) as run:
        result = ValService().evaluate("val.yaml")

    assert result.success is True
    assert result.audit_path is not None
    assert result.audit_path.exists()
    assert '"kind": "val"' in result.audit_path.read_text(encoding="utf-8")
    assert result.metrics is not None
    assert run.call_args.args[1]["data"] == str(ultra_data_path)
    assert not hasattr(result, "best_weight")
    prepare.assert_called_once()
    rename.assert_called_once()


def test_val_service_exception_returns_failed_result(tmp_path: Path) -> None:
    cfg, merger = _fake_config()
    weight = tmp_path / "models" / "trained" / "train3-best.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weights")

    with patch(f"{S}.build_val_config", return_value=(cfg, merger)), \
         patch(f"{S}.resolve_dataset_path", return_value=tmp_path / "dataset.yaml"), \
         patch(f"{S}.prepare_ultralytics_dataset_yaml", return_value=tmp_path / "dataset.runtime.yaml"), \
         patch(f"{S}.resolve_trained_model", return_value=weight), \
         patch(f"{S}.log_device_info"), \
         patch(f"{S}.log_effective_config"), \
         patch(f"{S}.log_override_chains"), \
         patch.object(ValService, "_run_evaluation", side_effect=RuntimeError("boom")):
        result = ValService().evaluate("val.yaml")

    assert result.success is False
    assert "boom" in (result.error or "")


def test_eval_cli_parser_accepts_expected_arguments() -> None:
    args = build_parser().parse_args(["--model", "train3-best.pt", "--data", "rsod.yaml", "--split", "test"])

    assert args.model == "train3-best.pt"
    assert args.data == "rsod.yaml"
    assert args.split == "test"


def test_eval_cli_returns_success_with_mocked_service() -> None:
    fake_result = MagicMock(success=True, output_dir=Path("runs/val"), error=None)

    with patch("od_platform.cli.evaluate_model.get_logger"), \
         patch("od_platform.cli.evaluate_model.ValService") as service_cls:
        service_cls.return_value.evaluate.return_value = fake_result
        exit_code = eval_main(["--model", "train3-best.pt", "--data", "rsod.yaml"])

    assert exit_code == 0
    service_cls.return_value.evaluate.assert_called_once()


def _fake_config():
    cfg = MagicMock()
    cfg.task = "detect"
    cfg.model = "train3-best.pt"
    cfg.data = "rsod.yaml"
    cfg.experiment_name = None
    cfg.to_ultralytics_kwargs.return_value = {"split": "val", "batch": 8}
    cfg.to_audit_snapshot.return_value = {"values": {"split": "val"}}
    merger = MagicMock()
    merger.to_audit_log.return_value = {"fields": {}}
    return cfg, merger


def _fake_yolo_results(tmp_path: Path):
    save_dir = tmp_path / "runs" / "detect_val" / "val3"
    save_dir.mkdir(parents=True)
    result = MagicMock()
    result.task = "detect"
    result.save_dir = save_dir
    result.fitness = 0.6
    result.speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 1.0}
    result.results_dict = {
        "metrics/precision(B)": 0.7,
        "metrics/recall(B)": 0.6,
        "metrics/mAP50(B)": 0.8,
        "metrics/mAP50-95(B)": 0.5,
        "fitness": 0.6,
    }
    result.names = {0: "airplane"}
    result.maps = [0.4]
    return result
