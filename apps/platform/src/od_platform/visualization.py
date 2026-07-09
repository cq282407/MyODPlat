#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : visualization.py
# @Project   : ODPlatform
# @Function  : Lightweight detection visualization for D8 inference.
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    confidence: float
    label: str


@dataclass(frozen=True)
class DrawStyle:
    box_thickness: int
    text_scale: float
    text_thickness: int
    text_color: tuple[int, int, int]
    text_bg_alpha: float
    label_padding: int = 4

    @classmethod
    def from_image_size(cls, height: int, width: int, **overrides) -> "DrawStyle":
        base = max(1, min(height, width) // 320)
        default = cls(
            box_thickness=max(2, base),
            text_scale=max(0.5, min(height, width) / 1400.0),
            text_thickness=max(1, base),
            text_color=(255, 255, 255),
            text_bg_alpha=0.65,
            label_padding=4,
        )
        payload = {**default.__dict__, **overrides}
        if "text_color" in payload and isinstance(payload["text_color"], list):
            payload["text_color"] = tuple(int(v) for v in payload["text_color"])
        return cls(**payload)


class BeautifyVisualizer:
    """Simple visualizer with label mapping and color mapping support."""

    def __init__(
        self,
        *,
        labels: Iterable[str] | None = None,
        label_mapping: dict[str, str] | None = None,
        color_mapping: dict[str, tuple[int, int, int]] | None = None,
        default_color: tuple[int, int, int] = (0, 255, 0),
        font_path: str | None = None,
    ) -> None:
        self.labels = list(labels or [])
        self.label_mapping = dict(label_mapping or {})
        self.color_mapping = dict(color_mapping or {})
        self.default_color = tuple(default_color)
        self.font_path = font_path
        self._font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    @staticmethod
    def from_yolo_results(*, boxes, confidences, labels) -> list[Detection]:
        detections: list[Detection] = []
        for box, confidence, label in zip(boxes, confidences, labels):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    confidence=float(confidence),
                    label=str(label),
                )
            )
        return detections

    def draw(
        self,
        image: np.ndarray,
        detections: Iterable[Detection],
        *,
        style: DrawStyle | None = None,
        use_label_mapping: bool = True,
    ) -> np.ndarray:
        canvas = image.copy()
        if style is None:
            h, w = canvas.shape[:2]
            style = DrawStyle.from_image_size(h, w)

        render_with_pil = any(_has_non_ascii(self._display_label(det.label, use_label_mapping)) for det in detections)
        pil_image: Image.Image | None = None
        pil_draw: ImageDraw.ImageDraw | None = None
        font = None
        if render_with_pil:
            pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            pil_draw = ImageDraw.Draw(pil_image, "RGBA")
            font = self._get_font(max(14, int(style.text_scale * 28)))

        for det in detections:
            color = self.color_mapping.get(det.label, self.default_color)
            x1, y1, x2, y2 = det.box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, style.box_thickness, cv2.LINE_AA)
            label = f"{self._display_label(det.label, use_label_mapping)} {det.confidence:.2f}"
            if pil_draw is not None and pil_image is not None and font is not None:
                self._draw_pil_label(pil_draw, label, (x1, y1), color, style, font)
            else:
                self._draw_cv_label(canvas, label, (x1, y1), color, style)

        if pil_image is not None:
            np.copyto(canvas, cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR))
        return canvas

    def _display_label(self, label: str, use_label_mapping: bool) -> str:
        if use_label_mapping:
            return self.label_mapping.get(label, label)
        return label

    def _draw_cv_label(
        self,
        canvas: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        style: DrawStyle,
    ) -> None:
        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            style.text_scale,
            style.text_thickness,
        )
        x, y = origin
        top = max(0, y - text_h - style.label_padding * 2 - baseline)
        bottom = max(text_h + baseline, y)
        left = max(0, x)
        right = left + text_w + style.label_padding * 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), color, -1)
        cv2.addWeighted(overlay, style.text_bg_alpha, canvas, 1 - style.text_bg_alpha, 0, canvas)
        cv2.putText(
            canvas,
            text,
            (left + style.label_padding, bottom - style.label_padding - baseline // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            style.text_scale,
            style.text_color,
            style.text_thickness,
            cv2.LINE_AA,
        )

    def _draw_pil_label(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        style: DrawStyle,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ) -> None:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x, y = origin
        top = max(0, y - text_h - style.label_padding * 2)
        left = max(0, x)
        right = left + text_w + style.label_padding * 2
        bottom = top + text_h + style.label_padding * 2
        rgba = (int(color[2]), int(color[1]), int(color[0]), int(255 * style.text_bg_alpha))
        draw.rounded_rectangle((left, top, right, bottom), radius=4, fill=rgba)
        text_rgb = (style.text_color[2], style.text_color[1], style.text_color[0])
        draw.text((left + style.label_padding, top + style.label_padding - 1), text, fill=text_rgb, font=font)

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
                    logger.warning("加载字体失败, 回退默认字体: %s", exc)
        if font is None:
            try:
                font = ImageFont.truetype("arial.ttf", size=size)
            except OSError:
                font = ImageFont.load_default()
        self._font_cache[size] = font
        return font


def _has_non_ascii(text: str) -> bool:
    return any(ord(char) > 127 for char in text)
