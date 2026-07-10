#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project   : ODPlatform
# @Function  : Renderer layer for beautified inference output.
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from od_platform.visualization.schema import AnnotationSet, BoxAnno, KeypointsAnno, MaskAnno, PolygonAnno
from od_platform.visualization.style import DrawStyle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderContext:
    label_mapping: Mapping[str, str]
    color_mapping: Mapping[str, tuple[int, int, int]]
    default_color: tuple[int, int, int]
    use_label_mapping: bool = True

    def display_label(self, label: str) -> str:
        if self.use_label_mapping:
            return str(self.label_mapping.get(label, label))
        return str(label)

    def color_for(self, label: str) -> tuple[int, int, int]:
        color = self.color_mapping.get(label, self.default_color)
        return int(color[0]), int(color[1]), int(color[2])


class _LabelDrawer:
    def __init__(
        self,
        canvas: np.ndarray,
        *,
        style: DrawStyle,
        font_path: str | None,
        label_texts: list[str],
    ) -> None:
        self.canvas = canvas
        self.style = style
        self.font_path = font_path
        self._font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
        self._pil_image: Image.Image | None = None
        self._pil_draw: ImageDraw.ImageDraw | None = None
        self._font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None

        if any(_has_non_ascii(text) for text in label_texts):
            self._pil_image = Image.fromarray(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))
            self._pil_draw = ImageDraw.Draw(self._pil_image, "RGBA")
            self._font = self._get_font(max(14, int(self.style.text_scale * 28)))

    def draw(self, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
        if self._pil_draw is not None and self._font is not None:
            self._draw_pil(text, origin, color)
            return
        self._draw_cv(text, origin, color)

    def finalize(self) -> np.ndarray:
        if self._pil_image is not None:
            np.copyto(self.canvas, cv2.cvtColor(np.asarray(self._pil_image), cv2.COLOR_RGB2BGR))
        return self.canvas

    def _draw_cv(self, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.style.text_scale,
            self.style.text_thickness,
        )
        x, y = origin
        top = max(0, y - text_h - self.style.label_padding * 2 - baseline)
        bottom = max(text_h + baseline, y)
        left = max(0, x)
        right = left + text_w + self.style.label_padding * 2
        overlay = self.canvas.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), color, -1)
        cv2.addWeighted(overlay, self.style.text_bg_alpha, self.canvas, 1 - self.style.text_bg_alpha, 0, self.canvas)
        cv2.putText(
            self.canvas,
            text,
            (left + self.style.label_padding, bottom - self.style.label_padding - baseline // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.style.text_scale,
            self.style.text_color,
            self.style.text_thickness,
            cv2.LINE_AA,
        )

    def _draw_pil(self, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
        assert self._pil_draw is not None
        assert self._font is not None
        bbox = self._pil_draw.textbbox((0, 0), text, font=self._font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x, y = origin
        top = max(0, y - text_h - self.style.label_padding * 2)
        left = max(0, x)
        right = left + text_w + self.style.label_padding * 2
        bottom = top + text_h + self.style.label_padding * 2
        rgba = (int(color[2]), int(color[1]), int(color[0]), int(255 * self.style.text_bg_alpha))
        self._pil_draw.rounded_rectangle((left, top, right, bottom), radius=4, fill=rgba)
        text_rgb = (self.style.text_color[2], self.style.text_color[1], self.style.text_color[0])
        self._pil_draw.text((left + self.style.label_padding, top + self.style.label_padding - 1), text, fill=text_rgb, font=self._font)

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if size in self._font_cache:
            return self._font_cache[size]
        font = None
        if self.font_path:
            path = Path(self.font_path)
            if path.exists():
                try:
                    font = ImageFont.truetype(str(path), size=size)
                except OSError as exc:
                    logger.warning("加载字体失败，回退默认字体: %s", exc)
        if font is None:
            try:
                font = ImageFont.truetype("arial.ttf", size=size)
            except OSError:
                font = ImageFont.load_default()
        self._font_cache[size] = font
        return font


class BaseRenderer:
    name = "base"

    def render(
        self,
        image: np.ndarray,
        annotations: AnnotationSet,
        *,
        style: DrawStyle,
        context: RenderContext,
        font_path: str | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def _label_text(self, label: str, confidence: float | None, context: RenderContext) -> str:
        text = context.display_label(label)
        if confidence is None:
            return text
        return f"{text} {confidence:.2f}"

    def _draw_keypoints(
        self,
        canvas: np.ndarray,
        annotations: tuple[KeypointsAnno, ...],
        context: RenderContext,
        style: DrawStyle,
    ) -> None:
        for anno in annotations:
            color = context.color_for(anno.label)
            for x, y, point_conf in anno.keypoints:
                if point_conf is not None and point_conf <= 0:
                    continue
                cv2.circle(canvas, (x, y), style.point_radius, color, -1, cv2.LINE_AA)


class BoxRenderer(BaseRenderer):
    name = "box"

    def render(
        self,
        image: np.ndarray,
        annotations: AnnotationSet,
        *,
        style: DrawStyle,
        context: RenderContext,
        font_path: str | None = None,
    ) -> np.ndarray:
        canvas = image.copy()
        label_texts = [self._label_text(item.label, item.confidence, context) for item in annotations.boxes]
        drawer = _LabelDrawer(canvas, style=style, font_path=font_path, label_texts=label_texts)
        for anno in annotations.boxes:
            color = context.color_for(anno.label)
            x1, y1, x2, y2 = anno.box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, style.box_thickness, cv2.LINE_AA)
            drawer.draw(self._label_text(anno.label, anno.confidence, context), (x1, y1), color)
        self._draw_keypoints(canvas, annotations.keypoints, context, style)
        return drawer.finalize()


class ContourRenderer(BaseRenderer):
    name = "contour"

    def __init__(self, fallback_renderer: BaseRenderer | None = None) -> None:
        self.fallback_renderer = fallback_renderer if fallback_renderer is not None else BoxRenderer()

    def render(
        self,
        image: np.ndarray,
        annotations: AnnotationSet,
        *,
        style: DrawStyle,
        context: RenderContext,
        font_path: str | None = None,
    ) -> np.ndarray:
        if not annotations.has_contours():
            return self.fallback_renderer.render(
                image,
                annotations,
                style=style,
                context=context,
                font_path=font_path,
            )

        canvas = image.copy()
        label_texts = [self._label_text(item.label, item.confidence, context) for item in annotations.masks]
        label_texts.extend(self._label_text(item.label, item.confidence, context) for item in annotations.polygons)
        drawer = _LabelDrawer(canvas, style=style, font_path=font_path, label_texts=label_texts)

        for anno in annotations.masks:
            color = context.color_for(anno.label)
            anchor = None
            for contour in anno.contours:
                anchor = self._draw_polygon(canvas, contour, color, style, close=True) or anchor
            if anchor is not None:
                drawer.draw(self._label_text(anno.label, anno.confidence, context), anchor, color)

        for anno in annotations.polygons:
            color = context.color_for(anno.label)
            anchor = self._draw_polygon(canvas, anno.points, color, style, close=True)
            if anchor is not None:
                drawer.draw(self._label_text(anno.label, anno.confidence, context), anchor, color)

        self._draw_keypoints(canvas, annotations.keypoints, context, style)
        return drawer.finalize()

    def _draw_polygon(
        self,
        canvas: np.ndarray,
        points: tuple[tuple[int, int], ...],
        color: tuple[int, int, int],
        style: DrawStyle,
        *,
        close: bool,
    ) -> tuple[int, int] | None:
        if len(points) < 2:
            return None
        pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        overlay = canvas.copy()
        if len(points) >= 3:
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, style.contour_fill_alpha, canvas, 1 - style.contour_fill_alpha, 0, canvas)
        cv2.polylines(canvas, [pts], close, color, style.contour_thickness, cv2.LINE_AA)
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        return min(xs), min(ys)


class AutoRenderer(BaseRenderer):
    name = "auto"

    def __init__(
        self,
        *,
        prefer_contour_when_mask_exists: bool = True,
        box_renderer: BaseRenderer | None = None,
        contour_renderer: BaseRenderer | None = None,
    ) -> None:
        self.prefer_contour_when_mask_exists = prefer_contour_when_mask_exists
        self.box_renderer = box_renderer if box_renderer is not None else BoxRenderer()
        self.contour_renderer = contour_renderer if contour_renderer is not None else ContourRenderer(fallback_renderer=self.box_renderer)

    def select_renderer(self, annotations: AnnotationSet) -> BaseRenderer:
        if self.prefer_contour_when_mask_exists and annotations.has_contours():
            return self.contour_renderer
        return self.box_renderer

    def render(
        self,
        image: np.ndarray,
        annotations: AnnotationSet,
        *,
        style: DrawStyle,
        context: RenderContext,
        font_path: str | None = None,
    ) -> np.ndarray:
        renderer = self.select_renderer(annotations)
        return renderer.render(image, annotations, style=style, context=context, font_path=font_path)


def _has_non_ascii(text: str) -> bool:
    return any(ord(char) > 127 for char in text)
