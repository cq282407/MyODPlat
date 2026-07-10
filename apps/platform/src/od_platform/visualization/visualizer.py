#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project   : ODPlatform
# @Function  : Visualization facade that composes adapter and renderer.
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from od_platform.visualization.adapters import YOLOResultAdapter
from od_platform.visualization.renderers import AutoRenderer, BoxRenderer, ContourRenderer, RenderContext
from od_platform.visualization.schema import AnnotationSet, BoxAnno, LabelAnno
from od_platform.visualization.style import DrawStyle

logger = logging.getLogger(__name__)


class BeautifyVisualizer:
    """Visualization facade that keeps adapter/render responsibilities separate."""

    def __init__(
        self,
        *,
        labels: Iterable[str] | None = None,
        label_mapping: dict[str, str] | None = None,
        color_mapping: dict[str, tuple[int, int, int]] | None = None,
        default_color: tuple[int, int, int] = (0, 255, 0),
        font_path: str | None = None,
        adapter: str = "yolo",
        renderer: str = "auto",
        theme: str = "classic",
        prefer_contour_when_mask_exists: bool = True,
    ) -> None:
        self.labels = list(labels or [])
        self.label_mapping = dict(label_mapping or {})
        self.color_mapping = dict(color_mapping or {})
        self.default_color = tuple(default_color)
        self.font_path = font_path
        self.adapter_name = str(adapter or "yolo").lower()
        self.renderer_name = str(renderer or "auto").lower()
        self.theme = str(theme or "classic").lower()
        self.prefer_contour_when_mask_exists = bool(prefer_contour_when_mask_exists)
        self.adapter = self._build_adapter(self.adapter_name)
        self.renderer = self._build_renderer(self.renderer_name)

    @staticmethod
    def from_yolo_results(*, boxes, confidences, labels) -> AnnotationSet:
        detections: list[BoxAnno] = []
        label_annos: list[LabelAnno] = []
        for box, confidence, label in zip(boxes, confidences, labels):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            label_text = str(label)
            conf_value = float(confidence)
            detections.append(
                BoxAnno(
                    box=(x1, y1, x2, y2),
                    confidence=conf_value,
                    label=label_text,
                    source="box",
                )
            )
            label_annos.append(LabelAnno(label=label_text, confidence=conf_value, source="box"))
        return AnnotationSet(boxes=tuple(detections), labels=tuple(label_annos))

    def adapt_result(self, result, *, names=None) -> AnnotationSet:
        return self.adapter.adapt(result, names=names or {})

    def choose_renderer_name(self, annotations: AnnotationSet) -> str:
        selector = getattr(self.renderer, "select_renderer", None)
        if callable(selector):
            return selector(annotations).name
        return getattr(self.renderer, "name", self.renderer_name)

    def draw(
        self,
        image: np.ndarray,
        annotations: AnnotationSet,
        *,
        style: DrawStyle | None = None,
        use_label_mapping: bool = True,
    ) -> np.ndarray:
        canvas = image.copy()
        if style is None:
            height, width = canvas.shape[:2]
            style = DrawStyle.from_image_size(height, width, theme=self.theme)
        context = RenderContext(
            label_mapping=self.label_mapping,
            color_mapping=self.color_mapping,
            default_color=self.default_color,
            use_label_mapping=use_label_mapping,
        )
        return self.renderer.render(
            canvas,
            annotations,
            style=style,
            context=context,
            font_path=self.font_path,
        )

    def draw_result(
        self,
        image: np.ndarray,
        result,
        *,
        names=None,
        style: DrawStyle | None = None,
        use_label_mapping: bool = True,
    ) -> np.ndarray:
        annotations = self.adapt_result(result, names=names or {})
        return self.draw(image, annotations, style=style, use_label_mapping=use_label_mapping)

    def _build_adapter(self, name: str):
        if name == "yolo":
            return YOLOResultAdapter()
        logger.warning("未知 visualization.adapter=%s, 回退 yolo", name)
        return YOLOResultAdapter()

    def _build_renderer(self, name: str):
        box_renderer = BoxRenderer()
        contour_renderer = ContourRenderer(fallback_renderer=box_renderer)
        if name == "box":
            return box_renderer
        if name == "contour":
            return contour_renderer
        if name == "auto":
            return AutoRenderer(
                prefer_contour_when_mask_exists=self.prefer_contour_when_mask_exists,
                box_renderer=box_renderer,
                contour_renderer=contour_renderer,
            )
        logger.warning("未知 visualization.renderer=%s, 回退 auto", name)
        return AutoRenderer(
            prefer_contour_when_mask_exists=self.prefer_contour_when_mask_exists,
            box_renderer=box_renderer,
            contour_renderer=contour_renderer,
        )
