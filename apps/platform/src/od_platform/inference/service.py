#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : service.py
# @Project   : ODPlatform
# @Function  : InferService — D8 threaded inference orchestration.
from __future__ import annotations

import json
import logging
import time
from argparse import Namespace
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from od_platform.common.config_log import log_effective_config, log_override_chains
from od_platform.common.log_rename import rename_log_to_save_dir
from od_platform.common.model_path import resolve_model_path
from od_platform.common.paths import PRETRAINED_MODELS_DIR, ROOT_DIR, RUNS_DIR, TRAINED_MODELS_DIR
from od_platform.common.system_utils import log_device_info
from od_platform.inference.cancel import CancelToken
from od_platform.inference.hooks import InferHooks
from od_platform.inference.pipeline import ThreadedPipeline
from od_platform.inference.pipeline_config import PipelineConfig, load_pipeline_config
from od_platform.inference.sinks import LocalFileSink, NullSink, OutputSink
from od_platform.runtime_config import build_infer_config
from od_platform.visualization import BeautifyVisualizer, DrawStyle

logger = logging.getLogger(__name__)

_PREDICT_KEYS: tuple[str, ...] = (
    "conf",
    "iou",
    "imgsz",
    "max_det",
    "classes",
    "agnostic_nms",
    "augment",
    "device",
    "retina_masks",
)


@dataclass
class InferStats:
    frames: int = 0
    detections: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    infer_time_sec: float = 0.0
    capture_fps: float = 0.0
    infer_fps: float = 0.0
    render_fps: float = 0.0
    loop_fps: float = 0.0
    current_fps: float = 0.0
    speed_ms: dict[str, float] = field(default_factory=dict)

    @property
    def avg_fps(self) -> float:
        return self.frames / self.infer_time_sec if self.infer_time_sec > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return (self.infer_time_sec / self.frames * 1000.0) if self.frames else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "detections": self.detections,
            "per_class": dict(self.per_class),
            "infer_time_sec": round(self.infer_time_sec, 4),
            "avg_fps": round(self.avg_fps, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "fps": {
                "capture": self.capture_fps,
                "infer": self.infer_fps,
                "render": self.render_fps,
                "loop": self.loop_fps,
                "current": self.current_fps,
            },
            "speed_ms": dict(self.speed_ms),
        }


def log_infer_stats(stats: InferStats, *, logger: logging.Logger = logger) -> None:
    logger.info("处理帧数:   %s", stats.frames)
    logger.info("检测总数:   %s", stats.detections)
    logger.info("平均延迟:   %.2f ms/帧", stats.avg_latency_ms)
    logger.info(
        "帧率(FPS):  捕获 %.1f | 推理 %.1f | 渲染 %.1f | loop %.1f | 当前 %.1f",
        stats.capture_fps,
        stats.infer_fps,
        stats.render_fps,
        stats.loop_fps,
        stats.current_fps,
    )
    if stats.speed_ms:
        logger.info(
            "模型 speed(ms): 预处理 %.2f | 推理 %.2f | 后处理 %.2f",
            stats.speed_ms.get("preprocess", 0.0),
            stats.speed_ms.get("inference", 0.0),
            stats.speed_ms.get("postprocess", 0.0),
        )
    if stats.per_class:
        logger.info("各类别检测数:")
        for name, count in sorted(stats.per_class.items(), key=lambda item: (-item[1], item[0])):
            logger.info("    %-20s %s", name, count)


@dataclass(frozen=True)
class InferResult:
    success: bool
    output_dir: Path
    stats: dict[str, Any] = field(default_factory=dict)
    infer_time: float | None = None
    saved: bool = False
    error: str | None = None
    audit_path: Path | None = None
    log_path: Path | None = None


class InferService:
    """YOLO inference orchestrator."""

    def predict(
        self,
        yaml_path: str | Path | None = None,
        pipeline_yaml: str | Path | None = None,
        cli_args: dict[str, Any] | Namespace | None = None,
        *,
        model: str | Path | None = None,
        source: str | Path | None = None,
        beautify: bool = True,
        rename_log: bool = True,
        threaded: bool = False,
        warmup_frames: int = 0,
        window_name: str = "odp-infer",
        show_info: bool = True,
        output_sink: OutputSink | None = None,
        hooks: InferHooks | None = None,
        cancel_token: CancelToken | None = None,
    ) -> InferResult:
        if hooks is None:
            hooks = InferHooks()
        merged_cli_args = _merge_cli_args(cli_args, model=model, source=source)

        start = datetime.now()
        output_dir: Path | None = None

        try:
            config, merger = build_infer_config(
                yaml_path=yaml_path or "infer.yaml",
                cli_args=merged_cli_args,
            )
            pipe: PipelineConfig = load_pipeline_config(pipeline_yaml)

            logger.info("=" * 60)
            logger.info("开始 YOLO 推理 (task=%s)", config.task)
            logger.info("=" * 60)

            raw_model = config.model or "yolo11n.pt"
            raw_source = config.source
            logger.info("任务类型:    %s", config.task)
            logger.info("输入源(声明): %r", raw_source)
            logger.info("模型(声明):   %s", raw_model)

            log_device_info(target_logger=logger)
            log_effective_config(config, merger, logger=logger)
            log_override_chains(config, merger, logger=logger)

            if raw_source is None:
                raise RuntimeError(
                    "未指定推理输入源。请在 infer.yaml 写 source，或用 `odp-infer --source <图/视频/目录/摄像头号>` 传入。"
                )

            model_path = resolve_model_path(raw_model, search_dirs=[TRAINED_MODELS_DIR, PRETRAINED_MODELS_DIR])
            logger.info("模型(解析):   %s", model_path)
            if _looks_like_explicit_path(raw_model) and not model_path.exists():
                raise FileNotFoundError(f"找不到推理模型: {model_path}")

            model_obj = YOLO(str(model_path))
            class_names = list((model_obj.names or {}).values())

            do_beautify = bool(beautify and pipe.viz_enabled)
            visualizer: BeautifyVisualizer | None = None
            if do_beautify:
                visualizer = BeautifyVisualizer(
                    labels=class_names,
                    label_mapping=pipe.label_mapping or None,
                    color_mapping=pipe.color_mapping or None,
                    default_color=pipe.default_color,
                    font_path=pipe.font_path,
                )
            else:
                logger.info("美化已关闭, 使用 YOLO 原生 plot() 绘制.")

            run_name = config.experiment_name or getattr(config, "name", None) or "predict"
            output_dir = _resolve_output_dir(
                RUNS_DIR / f"{config.task}_infer",
                run_name,
                exist_ok=bool(getattr(config, "exist_ok", False)),
            )
            logger.info("输出目录:     %s", output_dir)

            predict_kwargs = {
                key: getattr(config, key)
                for key in _PREDICT_KEYS
                if getattr(config, key, None) is not None
            }
            predict_kwargs["verbose"] = False

            want_save = bool(getattr(config, "save", True))
            want_show = bool(getattr(config, "show", False))

            if output_sink is None:
                output_sink = LocalFileSink() if want_save else NullSink()
            else:
                logger.info("使用调用方提供的 sink: %s", output_sink.__class__.__name__)

            logger.info("=" * 60)
            logger.info("启动推理")
            logger.info("=" * 60)

            stats = InferStats()
            processor = _FrameProcessor(
                model=model_obj,
                predict_kwargs=predict_kwargs,
                do_beautify=do_beautify,
                visualizer=visualizer,
                use_label_mapping=pipe.use_label_mapping,
                style_overrides=pipe.style_overrides,
                names=model_obj.names or {},
            )
            raw_batch = getattr(config, "batch", 16)
            batch_size = raw_batch if isinstance(raw_batch, int) and raw_batch >= 1 else 16

            pipeline = ThreadedPipeline(
                processor=processor,
                source=str(raw_source),
                camera_config=pipe.build_camera_config(),
                output_dir=output_dir,
                output_sink=output_sink,
                batch_size=batch_size,
                save=want_save,
                show=want_show,
                show_info=show_info,
                window_name=window_name,
                warmup_frames=warmup_frames,
                hooks=hooks,
                cancel_token=cancel_token,
            )
            interrupted = pipeline.run(stats)
            if interrupted:
                logger.warning("推理被用户提前结束.")

            logger.info("=" * 60)
            logger.info("推理完成")
            logger.info("=" * 60)
            log_infer_stats(stats, logger=logger)

            if rename_log:
                rename_log_to_save_dir(output_dir, Path(str(raw_model)).stem)

            infer_time = (datetime.now() - start).total_seconds()
            log_path = _find_project_log_path()
            audit_path = _write_audit(
                output_dir=output_dir,
                config=config,
                merger=merger,
                pipe=pipe,
                stats=stats,
                saved=want_save,
                beautified=do_beautify,
                infer_time=infer_time,
                log_path=log_path,
            )

            result = InferResult(
                success=True,
                output_dir=output_dir,
                stats=stats.to_dict(),
                infer_time=infer_time,
                saved=want_save,
                audit_path=audit_path,
                log_path=log_path,
            )
            hooks.fire_complete(result)
            return result
        except Exception as exc:
            logger.error("推理失败: %s", exc, exc_info=True)
            hooks.fire_error(exc)
            return InferResult(
                success=False,
                output_dir=output_dir or Path("unknown"),
                stats={},
                infer_time=(datetime.now() - start).total_seconds(),
                saved=False,
                error=str(exc),
                log_path=_find_project_log_path(),
            )


def infer_yolo(
    yaml_path: str | Path | None = None,
    pipeline_yaml: str | Path | None = None,
    cli_args: dict[str, Any] | Namespace | None = None,
    *,
    model: str | Path | None = None,
    source: str | Path | None = None,
    beautify: bool = True,
    rename_log: bool = True,
    threaded: bool = False,
    warmup_frames: int = 0,
    window_name: str = "odp-infer",
    show_info: bool = True,
    output_sink: OutputSink | None = None,
    hooks: InferHooks | None = None,
    cancel_token: CancelToken | None = None,
) -> InferResult:
    service = InferService()
    return service.predict(
        yaml_path=yaml_path,
        pipeline_yaml=pipeline_yaml,
        cli_args=cli_args,
        model=model,
        source=source,
        beautify=beautify,
        rename_log=rename_log,
        threaded=threaded,
        warmup_frames=warmup_frames,
        window_name=window_name,
        show_info=show_info,
        output_sink=output_sink,
        hooks=hooks,
        cancel_token=cancel_token,
    )


def _merge_cli_args(
    cli_args: dict[str, Any] | Namespace | None,
    *,
    model: str | Path | None,
    source: str | Path | None,
) -> dict[str, Any] | None:
    if cli_args is None and model is None and source is None:
        return None
    if cli_args is None:
        payload: dict[str, Any] = {}
    elif isinstance(cli_args, Namespace):
        payload = vars(cli_args).copy()
    else:
        payload = dict(cli_args)
    if model is not None:
        payload["model"] = str(model)
    if source is not None:
        payload["source"] = str(source)
    return payload


def _find_project_log_path() -> Path | None:
    root = logging.getLogger("od_platform")
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


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


def _looks_like_explicit_path(raw_model: str | Path) -> bool:
    model_path = Path(str(raw_model))
    return model_path.is_absolute() or len(model_path.parts) > 1 or model_path.suffix.lower() == ".engine"


def _write_audit(
    *,
    output_dir: Path,
    config: Any,
    merger: Any,
    pipe: PipelineConfig,
    stats: InferStats,
    saved: bool,
    beautified: bool,
    infer_time: float,
    log_path: Path | None,
) -> Path | None:
    audit_path = output_dir / "odp_audit.json"
    payload = {
        "schema_version": 1,
        "kind": "infer",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config.to_audit_snapshot(),
        "merger": merger.to_audit_log(),
        "pipeline": pipe.to_audit(),
        "stats": stats.to_dict(),
        "result_summary": {
            "output_dir": str(output_dir),
            "saved": saved,
            "beautified": beautified,
            "infer_time_sec": infer_time,
            "log_path": str(log_path) if log_path else None,
        },
    }
    try:
        audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("写审计快照失败 (不影响推理结果): %s", exc)
        return None
    logger.info("审计快照:   %s", audit_path)
    return audit_path


@dataclass
class _FrameProcessor:
    model: Any
    predict_kwargs: dict[str, Any]
    do_beautify: bool
    visualizer: BeautifyVisualizer | None
    use_label_mapping: bool
    style_overrides: dict[str, Any]
    names: dict[int, str]
    _style: DrawStyle | None = None

    def infer_batch(self, images: list):
        started = time.perf_counter()
        results = self.model(images, **self.predict_kwargs)
        batch_dt = time.perf_counter() - started
        labels_list: list[list[str]] = []
        n_list: list[int] = []
        for result in results:
            class_ids = _box_class_ids(getattr(result, "boxes", None))
            labels = [self.names.get(class_id, str(class_id)) for class_id in class_ids]
            labels_list.append(labels)
            n_list.append(len(labels))
        return results, labels_list, n_list, batch_dt

    def draw(self, image, result, labels, n_dets):
        if self.do_beautify and self.visualizer is not None:
            if self._style is None:
                height, width = image.shape[:2]
                self._style = DrawStyle.from_image_size(height, width, **self.style_overrides)
            boxes = getattr(result, "boxes", None)
            detections = BeautifyVisualizer.from_yolo_results(
                boxes=_box_xyxy(boxes, n_dets),
                confidences=_box_conf(boxes, n_dets),
                labels=labels,
            )
            return self.visualizer.draw(
                image,
                detections,
                style=self._style,
                use_label_mapping=self.use_label_mapping,
            )
        return result.plot()


def _box_class_ids(boxes) -> list[int]:
    if boxes is None:
        return []
    raw = getattr(boxes, "cls", None)
    if raw is None:
        return []
    values = _to_int_list(raw)
    out: list[int] = []
    for value in values:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _box_xyxy(boxes, count: int) -> np.ndarray:
    if boxes is None or count == 0:
        return np.zeros((0, 4), dtype=float)
    raw = getattr(boxes, "xyxy", None)
    if raw is None:
        return np.zeros((0, 4), dtype=float)
    arr = np.asarray(_to_nested_list(raw), dtype=float)
    return arr.reshape((-1, 4)) if arr.size else np.zeros((0, 4), dtype=float)


def _box_conf(boxes, count: int) -> np.ndarray:
    if boxes is None or count == 0:
        return np.zeros((0,), dtype=float)
    raw = getattr(boxes, "conf", None)
    if raw is None:
        return np.zeros((0,), dtype=float)
    return np.asarray(_to_float_list(raw), dtype=float).reshape((-1,))


def _to_nested_list(value) -> list:
    obj = value
    for method in ("cpu",):
        if hasattr(obj, method):
            obj = getattr(obj, method)()
    if hasattr(obj, "numpy"):
        obj = obj.numpy()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return list(obj)


def _to_float_list(value) -> list:
    obj = value
    for method in ("cpu",):
        if hasattr(obj, method):
            try:
                obj = getattr(obj, method)()
            except TypeError:
                obj = getattr(obj, method)
    if hasattr(obj, "numpy"):
        obj = obj.numpy()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


def _to_int_list(value) -> list:
    obj = value
    for method in ("int", "cpu"):
        if hasattr(obj, method):
            try:
                obj = getattr(obj, method)()
            except TypeError:
                obj = getattr(obj, method)
    if hasattr(obj, "numpy"):
        obj = obj.numpy()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]

