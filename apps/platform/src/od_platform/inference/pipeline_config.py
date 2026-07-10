#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : pipeline_config.py
# @Project   : ODPlatform
# @Function  : Read infer_pipeline.yaml for frame source and visualization.
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from od_platform.common.paths import RUNTIME_CONFIGS_DIR

logger = logging.getLogger(__name__)


def _to_bgr_tuple(value: Any) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return int(value[0]), int(value[1]), int(value[2])
    raise ValueError(f"颜色必须是 [B, G, R]，收到: {value!r}")


@dataclass
class PipelineConfig:
    camera_raw: dict[str, Any] = field(default_factory=dict)
    viz_enabled: bool = True
    viz_adapter: str = "yolo"
    viz_renderer: str = "auto"
    viz_theme: str = "classic"
    prefer_contour_when_mask_exists: bool = True
    use_label_mapping: bool = True
    label_mapping: dict[str, str] = field(default_factory=dict)
    color_mapping: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    default_color: tuple[int, int, int] = (0, 255, 0)
    font_path: str | None = None
    style_overrides: dict[str, Any] = field(default_factory=dict)

    def build_camera_config(self):
        from od_platform.frame_source import CameraConfig

        if not self.camera_raw:
            return None
        try:
            return CameraConfig(**self.camera_raw)
        except Exception as exc:
            logger.warning("camera 配置无效，回退默认: %s", exc)
            return None

    def to_audit(self) -> dict[str, Any]:
        return {
            "viz_enabled": self.viz_enabled,
            "viz_adapter": self.viz_adapter,
            "viz_renderer": self.viz_renderer,
            "viz_theme": self.viz_theme,
            "prefer_contour_when_mask_exists": self.prefer_contour_when_mask_exists,
            "use_label_mapping": self.use_label_mapping,
            "label_mapping_n": len(self.label_mapping),
            "color_mapping_n": len(self.color_mapping),
            "default_color": list(self.default_color),
            "font_path": self.font_path,
            "camera": dict(self.camera_raw),
            "style_overrides_n": len(self.style_overrides),
        }


def load_pipeline_config(yaml_path: str | Path | None) -> PipelineConfig:
    if yaml_path is None:
        resolved = RUNTIME_CONFIGS_DIR / "infer_pipeline.yaml"
    else:
        raw_path = Path(yaml_path)
        if raw_path.is_absolute():
            resolved = raw_path
        else:
            resolved = raw_path if raw_path.exists() else (RUNTIME_CONFIGS_DIR / raw_path.name)

    if not resolved.exists():
        logger.warning("pipeline yaml 不存在，使用默认配置: %s", resolved)
        return PipelineConfig()

    try:
        with resolved.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except Exception as exc:
        logger.warning("pipeline yaml 解析失败 (%s)，使用默认配置: %s", resolved, exc)
        return PipelineConfig()

    config = PipelineConfig()
    frame_source = raw.get("frame_source", {}) or {}
    config.camera_raw = dict(frame_source.get("camera", {}) or {})

    visualization = raw.get("visualization", {}) or {}
    config.viz_enabled = bool(visualization.get("enabled", True))
    config.viz_adapter = str(visualization.get("adapter", "yolo") or "yolo").lower()
    config.viz_renderer = str(visualization.get("renderer", "auto") or "auto").lower()
    config.viz_theme = str(visualization.get("theme", "classic") or "classic").lower()
    config.prefer_contour_when_mask_exists = bool(visualization.get("prefer_contour_when_mask_exists", True))
    config.use_label_mapping = bool(visualization.get("use_label_mapping", True))
    config.label_mapping = dict(visualization.get("label_mapping", {}) or {})
    config.color_mapping = {
        key: _to_bgr_tuple(value)
        for key, value in (visualization.get("color_mapping", {}) or {}).items()
    }
    if "default_color" in visualization:
        config.default_color = _to_bgr_tuple(visualization["default_color"])
    config.font_path = visualization.get("font_path")
    style_node = visualization.get("style", {}) or {}
    config.style_overrides = dict(style_node) if isinstance(style_node, dict) else {}

    logger.info(
        "pipeline 配置加载: %s, 美化=%s, adapter=%s, renderer=%s, theme=%s",
        resolved,
        config.viz_enabled,
        config.viz_adapter,
        config.viz_renderer,
        config.viz_theme,
    )
    return config
