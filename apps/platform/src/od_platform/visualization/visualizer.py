#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Beautified detection result drawing."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from od_platform.visualization.core.data_types import Detection, DrawStyle, KeypointDet, SegmentationDet
from od_platform.visualization.core.draw_utils import LayoutCalculator, RoundedRect
from od_platform.visualization.core.renderers import PillowTextRenderer
from od_platform.visualization.core.text_cache import TextSizeCache


class BeautifyVisualizer:
    """Draw rounded boxes, label backgrounds and optional mapped labels."""

    def __init__(
        self,
        labels: List[str],
        label_mapping: Optional[Dict[str, str]] = None,
        color_mapping: Optional[Dict[str, Tuple[int, int, int]]] = None,
        default_color: Tuple[int, int, int] = (0, 255, 0),
        font_path: Optional[str] = None,
        font_sizes: Optional[Tuple[int, ...]] = None,
    ) -> None:
        self.label_mapping = label_mapping or {}
        self.color_mapping = color_mapping or {}
        self.default_color = default_color
        self._size_cache = TextSizeCache(
            labels=labels,
            label_mapping=self.label_mapping,
            font_path=font_path,
            font_sizes=font_sizes,
        )
        self._renderer = PillowTextRenderer(size_cache=self._size_cache)

    def draw(
        self,
        image: np.ndarray,
        detections: List[Detection],
        style: Optional[DrawStyle] = None,
        use_label_mapping: bool = False,
    ) -> np.ndarray:
        if not detections:
            return image.copy()

        height, width = image.shape[:2]
        style = style or DrawStyle.from_image_size(height, width)
        result = image.copy()
        texts: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]] = []

        for det in detections:
            # ── 按类型分发 ──
            if isinstance(det, SegmentationDet):
                self._draw_mask(result, det)
            elif isinstance(det, KeypointDet):
                self._draw_keypoints(result, det, style)

            # ── 公共的框 + 标签绘制 (所有类型都画) ──
            x1, y1, x2, y2 = det.box
            color = self.color_mapping.get(det.label, det.color or self.default_color)
            display_label = self.label_mapping.get(det.label, det.label) if use_label_mapping else det.label
            label_text = f"{display_label} {det.confidence * 100:.1f}%"
            text_size = self._size_cache.get_size(display_label, style.font_size)
            layout = LayoutCalculator.compute(det.box, text_size, (height, width), style)

            RoundedRect.bordered(
                result,
                (x1, y1),
                (x2, y2),
                color,
                style.line_width,
                style.radius,
                LayoutCalculator.get_corners(layout, for_detection=True),
            )
            lx1, ly1, lx2, ly2 = layout.box
            RoundedRect.filled(
                result,
                (lx1, ly1),
                (lx2, ly2),
                color,
                style.radius,
                LayoutCalculator.get_corners(layout, for_detection=False),
            )
            texts.append((label_text, layout.text_pos, style.text_color))

        return self._renderer.render_batch(result, texts, style)

    # ------------------------------------------------------------------
    # 类型专用绘制——只在这里加新类型, draw() 不变
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_mask(image: np.ndarray, det: SegmentationDet) -> None:
        """绘制分割 mask: 半透明彩色遮罩叠加."""
        import cv2
        if det.mask is None or not det.mask.any():
            return
        overlay = image.copy()
        overlay[det.mask] = det.color
        alpha = 0.35
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    @staticmethod
    def _draw_keypoints(image: np.ndarray, det: KeypointDet, style: DrawStyle) -> None:
        """绘制关键点骨架: 圆点 + 连线."""
        import cv2
        if det.keypoints is None or len(det.keypoints) == 0:
            return
        kpts = det.keypoints
        color = det.color
        lw = max(1, style.line_width - 1)

        # 画骨架连线 (COCO 17 点连接关系)
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),       # 头 → 肩 → 肘 → 腕
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 髋 → 膝 → 踝
            (5, 11), (6, 12), (11, 12),            # 髋 ↔ 肩
        ]
        for i, j in skeleton:
            if i >= len(kpts) or j >= len(kpts):
                continue
            if kpts[i][2] > 0 and kpts[j][2] > 0:  # 两点都可见
                pt1 = (int(kpts[i][0]), int(kpts[i][1]))
                pt2 = (int(kpts[j][0]), int(kpts[j][1]))
                cv2.line(image, pt1, pt2, color, lw, cv2.LINE_AA)

        # 画关键点圆点
        for x, y, v in kpts:
            if v > 0:
                cv2.circle(image, (int(x), int(y)), max(2, lw), color, -1, cv2.LINE_AA)

    @staticmethod
    def from_yolo_results(
        boxes: np.ndarray,
        confidences: np.ndarray,
        labels: Iterable[str],
        color_mapping: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> List[Detection]:
        color_mapping = color_mapping or {}
        return [
            Detection(
                box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                confidence=float(confidence),
                label=label,
                color=color_mapping.get(label, (0, 255, 0)),
            )
            for box, confidence, label in zip(boxes, confidences, labels)
        ]

