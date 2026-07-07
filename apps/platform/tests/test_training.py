#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_training.py
# @Project   : ODPlatform
# @Function  : D6 training service and archive tests.
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from od_platform.cli.train_model import build_parser, main as train_main
from od_platform.training.archive import archive_checkpoints
from od_platform.training.service import TrainService
from od_platform.training.visualization import render_training_results_chart

S = "od_platform.training.service"


def test_archive_copies_best_and_last(tmp_path: Path) -> None:
    train_dir = _fake_train_dir(tmp_path)
    archive_dir = tmp_path / "archive"

    result = archive_checkpoints(train_dir, "yolo11n.pt", checkpoint_dir=archive_dir)

    assert set(result) == {"best", "last"}
    assert result["best"].exists()
    assert "train3" in result["best"].name
    assert "yolo11n" in result["best"].name


def test_archive_missing_train_dir_returns_empty(tmp_path: Path) -> None:
    assert archive_checkpoints(tmp_path / "missing", "yolo11n.pt", checkpoint_dir=tmp_path / "archive") == {}


def test_visualization_writes_png_from_results_csv(tmp_path: Path) -> None:
    csv_text = (
        "epoch,train/box_loss,val/box_loss,train/cls_loss,val/cls_loss,train/dfl_loss,val/dfl_loss,"
        "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),"
        "lr/pg0,lr/pg1,lr/pg2,time\n"
        "0,1,1.1,0.8,0.9,0.7,0.75,0.5,0.4,0.3,0.2,0.01,0.01,0.01,3\n"
        "1,0.9,1.0,0.7,0.8,0.6,0.7,0.6,0.5,0.4,0.3,0.009,0.009,0.009,4\n"
    )
    (tmp_path / "results.csv").write_text(csv_text, encoding="utf-8")

    output = render_training_results_chart(tmp_path)

    assert output is not None
    assert output.exists()


def test_train_service_validation_failure_blocks_training(tmp_path: Path) -> None:
    cfg, merger = _fake_config(tmp_path)
    report = MagicMock(exit_code=2, results=[MagicMock(severity="ERROR")])

    with patch(f"{S}.build_train_config", return_value=(cfg, merger)), \
         patch(f"{S}.resolve_dataset_path", return_value=tmp_path / "dataset.yaml"), \
         patch(f"{S}.resolve_model_path", return_value=Path("yolo11n.pt")), \
         patch(f"{S}.log_device_info"), \
         patch(f"{S}.log_effective_config"), \
         patch(f"{S}.log_override_chains"), \
         patch(f"{S}.validate_dataset", return_value=report), \
         patch(f"{S}.render_to_logger"), \
         patch.object(TrainService, "_run_training") as run:
        result = TrainService().train("train.yaml", {"data": "sample.yaml"})

    assert result.success is False
    assert "数据集校验失败" in (result.error or "")
    run.assert_not_called()


def test_train_service_success_writes_audit_and_calls_artifacts(tmp_path: Path) -> None:
    cfg, merger = _fake_config(tmp_path)
    yolo_results = _fake_yolo_results(tmp_path)
    report = MagicMock(exit_code=0, results=[])

    with patch(f"{S}.build_train_config", return_value=(cfg, merger)), \
         patch(f"{S}.resolve_dataset_path", return_value=tmp_path / "dataset.yaml"), \
         patch(f"{S}.resolve_model_path", return_value=Path("yolo11n.pt")), \
         patch(f"{S}.log_device_info"), \
         patch(f"{S}.log_effective_config"), \
         patch(f"{S}.log_override_chains"), \
         patch(f"{S}.validate_dataset", return_value=report), \
         patch(f"{S}.render_to_logger"), \
         patch(f"{S}.rename_log_to_save_dir", return_value=tmp_path / "train.log") as rename, \
         patch(f"{S}.archive_checkpoints", return_value={"best": yolo_results.save_dir / "archive-best.pt"}) as archive, \
         patch(f"{S}.render_training_results_chart", return_value=yolo_results.save_dir / "training_results.png") as viz, \
         patch.object(TrainService, "_run_training", return_value=yolo_results):
        (yolo_results.save_dir / "archive-best.pt").write_bytes(b"best")
        (yolo_results.save_dir / "training_results.png").write_bytes(b"png")
        result = TrainService().train("train.yaml", {"data": "sample.yaml"})

    assert result.success is True
    assert result.audit_path is not None
    assert result.audit_path.exists()
    assert result.visualization_path is not None
    rename.assert_called_once()
    archive.assert_called_once()
    viz.assert_called_once()


def test_train_service_exception_returns_failed_result(tmp_path: Path) -> None:
    cfg, merger = _fake_config(tmp_path)

    with patch(f"{S}.build_train_config", return_value=(cfg, merger)), \
         patch(f"{S}.resolve_dataset_path", return_value=tmp_path / "dataset.yaml"), \
         patch(f"{S}.resolve_model_path", return_value=Path("yolo11n.pt")), \
         patch(f"{S}.log_device_info"), \
         patch(f"{S}.log_effective_config"), \
         patch(f"{S}.log_override_chains"), \
         patch(f"{S}.validate_dataset", return_value=MagicMock(exit_code=0, results=[])), \
         patch(f"{S}.render_to_logger"), \
         patch.object(TrainService, "_run_training", side_effect=RuntimeError("CUDA OOM")):
        result = TrainService().train("train.yaml", {"data": "sample.yaml"})

    assert result.success is False
    assert "CUDA OOM" in (result.error or "")


def test_train_cli_parser_accepts_expected_arguments() -> None:
    args = build_parser().parse_args(["--data", "sample.yaml", "--epochs", "1", "--batch", "0.5", "--no-archive"])

    assert args.data == "sample.yaml"
    assert args.epochs == 1
    assert args.batch == 0.5
    assert args.archive is False


def test_train_cli_returns_success_with_mocked_service() -> None:
    fake_result = MagicMock(success=True, output_dir=Path("runs/train"), train_time=1.0)

    with patch("od_platform.cli.train_model.get_logger"), \
         patch("od_platform.cli.train_model.TrainService") as service_cls:
        service_cls.return_value.train.return_value = fake_result
        exit_code = train_main(["--data", "sample.yaml", "--epochs", "1"])

    assert exit_code == 0
    service_cls.return_value.train.assert_called_once()


def _fake_train_dir(root: Path) -> Path:
    train_dir = root / "runs" / "detect_train" / "train3"
    (train_dir / "weights").mkdir(parents=True)
    (train_dir / "weights" / "best.pt").write_bytes(b"best")
    (train_dir / "weights" / "last.pt").write_bytes(b"last")
    return train_dir


def _fake_config(tmp_path: Path):
    cfg = MagicMock()
    cfg.task = "detect"
    cfg.model = "yolo11n.pt"
    cfg.data = "sample.yaml"
    cfg.experiment_name = None
    cfg.to_ultralytics_kwargs.return_value = {"epochs": 1}
    cfg.to_audit_snapshot.return_value = {"values": {"epochs": 1}}
    merger = MagicMock()
    merger.to_audit_log.return_value = {"fields": {}}
    return cfg, merger


def _fake_yolo_results(tmp_path: Path):
    save_dir = _fake_train_dir(tmp_path)
    result = MagicMock()
    result.task = "detect"
    result.save_dir = save_dir
    result.fitness = 0.5
    result.speed = {}
    result.results_dict = {"metrics/mAP50(B)": 0.5, "fitness": 0.5}
    result.names = {}
    result.maps = []
    return result
