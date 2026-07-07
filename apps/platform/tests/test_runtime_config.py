#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_runtime_config.py
# @Project   : ODPlatform
# @Function  : runtime_config smoke and regression tests.

from argparse import Namespace
from pathlib import Path

import pytest
from pydantic import ValidationError

from od_platform.common.paths import RUNTIME_CONFIGS_DIR, runtime_config_path
from od_platform.runtime_config import (
    CLILoader,
    ConfigGenerator,
    ConfigMerger,
    ConfigSource,
    YOLOInferConfig,
    YOLOTrainConfig,
    YOLOValConfig,
    build_train_config,
)


def test_runtime_config_path_points_to_runtime_dir() -> None:
    assert runtime_config_path("train") == RUNTIME_CONFIGS_DIR / "train.yaml"


def test_extra_field_is_forbidden() -> None:
    with pytest.raises(ValidationError):
        YOLOTrainConfig(epoch=100)


def test_cli_loader_keeps_explicit_false_zero_and_empty_string() -> None:
    args = Namespace(save=False, workers=0, name="", batch=None, config="train.yaml")

    assert CLILoader().load(args) == {
        "save": False,
        "workers": 0,
        "name": "",
    }


def test_val_and_infer_conf_defaults_match_yolo_semantics() -> None:
    assert YOLOValConfig().conf == 0.001
    assert YOLOInferConfig().conf == 0.25


def test_framework_only_fields_are_not_sent_to_ultralytics() -> None:
    train_kwargs = YOLOTrainConfig(verbose=False, experiment_name="baseline").to_ultralytics_kwargs()
    val_kwargs = YOLOValConfig(task="detect", experiment_name="val_baseline").to_ultralytics_kwargs()
    infer_kwargs = YOLOInferConfig(task="detect", experiment_name="demo").to_ultralytics_kwargs()

    assert "verbose" not in train_kwargs
    assert "experiment_name" not in train_kwargs
    assert "task" in train_kwargs
    assert "task" not in val_kwargs
    assert "task" not in infer_kwargs


def test_config_merger_uses_default_yaml_cli_precedence() -> None:
    merger = ConfigMerger()

    cfg = merger.merge(
        YOLOTrainConfig,
        sources=[
            (ConfigSource.YAML, {"batch": 8, "epochs": 50}),
            (ConfigSource.CLI, {"batch": 4}),
        ],
    )

    assert cfg.batch == 4
    assert cfg.epochs == 50
    report = merger.get_override_report()
    assert "batch: 4(CLI) ← 8(YAML) ← 16(DEFAULT)" in report


def test_build_train_config_merges_yaml_and_cli(tmp_path: Path) -> None:
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text("batch: 8\nepochs: 3\nsave: false\n", encoding="utf-8")
    args = Namespace(batch=4, config=None)

    cfg, merger = build_train_config(yaml_path=yaml_path, cli_args=args)

    assert cfg is not None
    assert cfg.batch == 4
    assert cfg.epochs == 3
    assert cfg.save is False
    assert "batch" in merger.get_override_report()


def test_generator_output_keeps_teacher_format() -> None:
    text = ConfigGenerator()._generate_yaml(YOLOTrainConfig, "YOLO 训练配置")
    lines = text.splitlines()

    assert lines[0] == "#=============================================================================="
    assert lines[1] == "# YOLO 训练配置"
    assert lines[2].startswith("# 自动生成时间: ")
    assert "#------------------------------------------------------------------------------" in text
    assert "# 核心参数" in text
    assert "model: null" in text
    assert "save: true" in text
    assert "deterministic: true" in text


def test_generator_writes_and_skips_existing_file(tmp_path: Path) -> None:
    output_path = tmp_path / "train.yaml"
    generator = ConfigGenerator()

    assert generator.generate(YOLOTrainConfig, output_path, title="YOLO 训练配置") is True
    original = output_path.read_text(encoding="utf-8")
    assert generator.generate(YOLOTrainConfig, output_path, title="YOLO 训练配置") is False
    assert output_path.read_text(encoding="utf-8") == original
