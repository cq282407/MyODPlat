#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_training_common.py
# @Project   : ODPlatform
# @Function  : D6 common utility tests.
from __future__ import annotations

import logging
import math
from pathlib import Path
from unittest.mock import MagicMock

from od_platform.common.dataset_path import prepare_ultralytics_dataset_yaml, resolve_dataset_path
from od_platform.common.log_rename import rename_log_to_save_dir
from od_platform.common.logging_utils import ROOT_LOGGER_NAME
from od_platform.common.model_path import resolve_model_path
from od_platform.common.refs import resolve_pretrained_model, resolve_yaml
from od_platform.common.result import TrainMetrics


def test_refs_resolve_bare_yaml_and_model_names() -> None:
    assert resolve_yaml("nwpu").name == "nwpu.yaml"
    assert resolve_pretrained_model("yolo11n").name == "yolo11n.pt"


def test_dataset_path_returns_resolved_yaml(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    (dataset_dir / "sample.yaml").write_text("path: .\n", encoding="utf-8")
    monkeypatch.setattr("od_platform.common.refs.DATASET_CONFIGS_DIR", dataset_dir)

    assert resolve_dataset_path("sample") == (dataset_dir / "sample.yaml").resolve()


def test_prepare_ultralytics_dataset_yaml_uses_yaml_parent_when_path_missing(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data" / "processed" / "sample"
    dataset_dir.mkdir(parents=True)
    yaml_path = dataset_dir / "dataset.yaml"
    yaml_path.write_text(
        "train: images/train\nval: images/val\nnc: 1\nnames:\n  0: airplane\n",
        encoding="utf-8",
    )

    runtime_yaml = prepare_ultralytics_dataset_yaml(yaml_path)
    content = runtime_yaml.read_text(encoding="utf-8")

    assert f"path: {dataset_dir.as_posix()}" in content
    assert "train: images/train" in content


def test_prepare_ultralytics_dataset_yaml_resolves_relative_config_path(tmp_path: Path) -> None:
    config_dir = tmp_path / "apps" / "platform" / "configs" / "datasets"
    dataset_dir = tmp_path / "data" / "processed" / "sample"
    config_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)
    yaml_path = config_dir / "sample.yaml"
    yaml_path.write_text(
        "path: ../../../../data/processed/sample\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: airplane\n",
        encoding="utf-8",
    )

    runtime_yaml = prepare_ultralytics_dataset_yaml(yaml_path)

    assert f"path: {dataset_dir.as_posix()}" in runtime_yaml.read_text(encoding="utf-8")


def test_model_path_search_dirs_hits_first_existing(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (second / "yolo11n.pt").write_bytes(b"weights")

    assert resolve_model_path("yolo11n.pt", search_dirs=[first, second]) == second / "yolo11n.pt"


def test_model_path_missing_returns_original_name(tmp_path: Path) -> None:
    result = resolve_model_path("missing.pt", search_dirs=[tmp_path])

    assert result == Path("missing.pt")


def test_model_path_missing_bare_name_gets_pt_suffix(tmp_path: Path) -> None:
    result = resolve_model_path("yolo11n", search_dirs=[tmp_path])

    assert result == Path("yolo11n.pt")


def test_log_rename_named_root(tmp_path: Path) -> None:
    root = logging.getLogger(ROOT_LOGGER_NAME)
    _clear_handlers(root)
    old_log = tmp_path / "train_20260707-121314.log"
    old_log.write_text("hello\n", encoding="utf-8")
    handler = logging.FileHandler(old_log, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    save_dir = tmp_path / "runs" / "detect_train" / "train3"
    save_dir.mkdir(parents=True)

    try:
        new_path = rename_log_to_save_dir(save_dir, "yolo11n")
        assert new_path is not None
        assert new_path.name == "train3_20260707-121314_yolo11n.log"
        assert new_path.exists()
    finally:
        _clear_handlers(root)


def test_log_rename_no_handler_returns_none() -> None:
    root = logging.getLogger(ROOT_LOGGER_NAME)
    _clear_handlers(root)

    assert rename_log_to_save_dir(Path("train3"), "yolo11n") is None


def test_train_metrics_from_yolo_results_and_nan_cleanup() -> None:
    results = MagicMock()
    results.task = "detect"
    results.save_dir = Path("/tmp/train")
    results.fitness = None
    results.speed = {"preprocess": 1.0, "inference": 2.0, "loss": None, "postprocess": 3.0}
    results.results_dict = {
        "metrics/precision(B)": 0.7,
        "metrics/recall(B)": 0.6,
        "metrics/mAP50(B)": 0.8,
        "fitness": 0.75,
    }
    results.names = {0: "airplane", 1: "ship"}
    results.maps = [0.5, 0.4]

    metrics = TrainMetrics.from_yolo_results(results)
    payload = metrics.to_dict()

    assert metrics.task == "detect"
    assert metrics.overall["metrics/mAP50(B)"] == 0.8
    assert metrics.speed_ms["total"] == 6.0
    assert payload["speed_ms"]["loss"] is None
    assert payload["class_map_50_95"]["airplane"] == 0.5


def test_train_metrics_unknown_task_still_serializes() -> None:
    metrics = TrainMetrics(
        task="unknown",
        save_dir=Path("/tmp"),
        timestamp="2026-07-07T12:00:00",
        speed_ms={"preprocess": math.nan},
        overall={"fitness": 0.1, "custom": 0.2},
    )

    assert metrics.to_dict()["speed_ms"]["preprocess"] is None


def _clear_handlers(root: logging.Logger) -> None:
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
