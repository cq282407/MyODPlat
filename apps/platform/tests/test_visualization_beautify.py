#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations

import numpy as np
import pytest

from od_platform.visualization import BeautifyVisualizer, Detection, DrawStyle


def test_beautify_visualizer_draws_detection_without_mutating_input() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    original = image.copy()
    visualizer = BeautifyVisualizer(
        labels=["airplane"],
        label_mapping={"airplane": "plane"},
        color_mapping={"airplane": (0, 128, 255)},
    )

    result = visualizer.draw(
        image,
        [Detection(box=(10, 20, 80, 70), confidence=0.87, label="airplane")],
        style=DrawStyle(font_size=14, line_width=2, padding_x=4, padding_y=4, radius=4),
        use_label_mapping=True,
    )

    assert result.shape == image.shape
    assert np.array_equal(image, original)
    assert result.sum() > 0


def test_from_yolo_results_builds_detections() -> None:
    detections = BeautifyVisualizer.from_yolo_results(
        boxes=np.array([[1.2, 2.5, 10.1, 20.8]]),
        confidences=np.array([0.5]),
        labels=["ship"],
    )

    assert detections == [Detection(box=(1, 2, 10, 20), confidence=0.5, label="ship")]


def test_draw_style_rejects_invalid_color() -> None:
    with pytest.raises(ValueError):
        DrawStyle(text_color=(0, 0, 999))


def test_default_font_resolution_prefers_chinese_capable_font() -> None:
    from od_platform.visualization.core.text_cache import _resolve_font_path

    resolved = _resolve_font_path(None)

    assert resolved
