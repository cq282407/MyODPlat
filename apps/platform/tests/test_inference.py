#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_inference.py
# @Project   : ODPlatform
# @Function  : D8 inference service and CLI tests.
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from od_platform.cli.infer_model import build_parser, main as infer_main
from od_platform.inference import service as infer_service_module
from od_platform.inference.pipeline_config import load_pipeline_config
from od_platform.inference.service import InferService

S = "od_platform.inference.service"


def test_load_pipeline_config_missing_file_returns_defaults(tmp_path: Path) -> None:
    cfg = load_pipeline_config(tmp_path / "missing.yaml")

    assert cfg.viz_enabled is True
    assert cfg.label_mapping == {}
    assert cfg.color_mapping == {}


def test_box_conf_preserves_fractional_values() -> None:
    class _DummyTensor:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def cpu(self):
            return self

        def numpy(self):
            import numpy as np

            return np.asarray(self._values, dtype=float)

    boxes = SimpleNamespace(conf=_DummyTensor([0.91, 0.37]))

    conf = infer_service_module._box_conf(boxes, 2)

    assert conf.tolist() == pytest.approx([0.91, 0.37])


def test_infer_service_missing_source_fails_fast() -> None:
    cfg, merger = _fake_config()
    cfg.source = None
    pipe = _fake_pipe()

    with patch(f"{S}.build_infer_config", return_value=(cfg, merger)), \
         patch(f"{S}.load_pipeline_config", return_value=pipe), \
         patch(f"{S}.log_device_info"), \
         patch(f"{S}.log_effective_config"), \
         patch(f"{S}.log_override_chains"):
        result = InferService().predict("infer.yaml")

    assert result.success is False
    assert "未指定推理输入源" in (result.error or "")


def test_infer_service_missing_explicit_model_fails_fast(tmp_path: Path) -> None:
    cfg, merger = _fake_config()
    cfg.model = str(tmp_path / "missing.engine")
    pipe = _fake_pipe()

    with patch(f"{S}.build_infer_config", return_value=(cfg, merger)), \
         patch(f"{S}.load_pipeline_config", return_value=pipe), \
         patch(f"{S}.log_device_info"), \
         patch(f"{S}.log_effective_config"), \
         patch(f"{S}.log_override_chains"):
        result = InferService().predict("infer.yaml")

    assert result.success is False
    assert "找不到推理模型" in (result.error or "")


def test_infer_service_success_writes_audit_and_uses_threaded_pipeline(tmp_path: Path) -> None:
    cfg, merger = _fake_config()
    weight = tmp_path / "models" / "trained" / "demo.engine"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"engine")
    cfg.model = str(weight)
    cfg.experiment_name = "demo_infer"
    cfg.show = True
    cfg.to_ultralytics_kwargs.return_value = {"conf": 0.25, "imgsz": 640}
    pipe = _fake_pipe()

    fake_model = MagicMock()
    fake_model.names = {0: "airplane", 1: "ship"}

    fake_pipeline = MagicMock()

    def _run(stats) -> bool:
        stats.frames = 10
        stats.detections = 6
        stats.infer_time_sec = 0.5
        stats.capture_fps = 88.0
        stats.infer_fps = 80.0
        stats.render_fps = 72.0
        stats.loop_fps = 70.0
        stats.current_fps = 69.5
        stats.speed_ms = {"preprocess": 0.8, "inference": 2.0, "postprocess": 0.5}
        stats.per_class = {"ship": 4, "airplane": 2}
        return False

    fake_pipeline.run.side_effect = _run

    with patch(f"{S}.build_infer_config", return_value=(cfg, merger)), \
         patch(f"{S}.load_pipeline_config", return_value=pipe), \
         patch(f"{S}.YOLO", return_value=fake_model), \
         patch(f"{S}.ThreadedPipeline", return_value=fake_pipeline) as pipeline_cls, \
         patch(f"{S}.log_device_info"), \
         patch(f"{S}.log_effective_config"), \
         patch(f"{S}.log_override_chains"), \
         patch(f"{S}.rename_log_to_save_dir", return_value=tmp_path / "infer.log"):
        result = InferService().predict("infer.yaml", pipeline_yaml="infer_pipeline.yaml")

    assert result.success is True
    assert result.output_dir.exists()
    assert result.audit_path is not None and result.audit_path.exists()
    assert '"kind": "infer"' in result.audit_path.read_text(encoding="utf-8")
    assert result.stats["frames"] == 10
    assert result.stats["detections"] == 6
    assert result.stats["fps"]["loop"] == 70.0
    kwargs = pipeline_cls.call_args.kwargs
    assert kwargs["show"] is True
    assert kwargs["show_info"] is True
    assert kwargs["save"] is True
    assert kwargs["batch_size"] == 16


def test_infer_cli_parser_accepts_pipeline_flags() -> None:
    args = build_parser().parse_args(
        ["--model", "demo.engine", "--source", "demo.mp4", "--pipeline-yaml", "infer_pipeline.yaml", "--no-viz", "--warmup", "3"]
    )

    assert args.model == "demo.engine"
    assert args.source == "demo.mp4"
    assert args.pipeline_yaml == "infer_pipeline.yaml"
    assert args.no_viz is True
    assert args.warmup == 3
    assert args.show is None
    assert args.save is None


def test_infer_cli_returns_success_with_mocked_service() -> None:
    fake_result = MagicMock(success=True, output_dir=Path("runs/infer"), stats={"frames": 12}, error=None)

    with patch("od_platform.cli.infer_model.get_logger"), \
         patch("od_platform.cli.infer_model.infer_yolo", return_value=fake_result) as infer_fn:
        exit_code = infer_main(["--model", "demo.engine", "--source", "demo.mp4"])

    assert exit_code == 0
    infer_fn.assert_called_once()


def test_infer_cli_returns_130_on_keyboard_interrupt() -> None:
    with patch("od_platform.cli.infer_model.get_logger"), \
         patch("od_platform.cli.infer_model.infer_yolo", side_effect=KeyboardInterrupt):
        exit_code = infer_main(["--model", "demo.engine", "--source", "demo.mp4"])

    assert exit_code == 130


def _fake_config():
    cfg = MagicMock()
    cfg.task = "detect"
    cfg.model = "demo.engine"
    cfg.source = "demo.mp4"
    cfg.show = False
    cfg.save = True
    cfg.exist_ok = False
    cfg.name = None
    cfg.experiment_name = None
    cfg.to_ultralytics_kwargs.return_value = {"conf": 0.25, "imgsz": 640}
    cfg.to_audit_snapshot.return_value = {"values": {"source": "demo.mp4"}}
    merger = MagicMock()
    merger.to_audit_log.return_value = {"fields": {}}
    return cfg, merger


def _fake_pipe():
    pipe = MagicMock()
    pipe.viz_enabled = True
    pipe.label_mapping = {}
    pipe.color_mapping = {}
    pipe.default_color = (0, 255, 0)
    pipe.font_path = None
    pipe.use_label_mapping = True
    pipe.style_overrides = {}
    pipe.build_camera_config.return_value = None
    pipe.to_audit.return_value = {"viz_enabled": True}
    return pipe
