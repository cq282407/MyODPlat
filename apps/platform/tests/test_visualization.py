#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project   : ODPlatform
# @Function  : Visualization adapter and renderer regression tests.
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from od_platform.inference.pipeline_config import load_pipeline_config
from od_platform.visualization import AnnotationSet, AutoRenderer, BeautifyVisualizer, BoxAnno, DrawStyle, MaskAnno
from od_platform.visualization.renderers import ContourRenderer, RenderContext


class _DummyTensor:
    def __init__(self, values) -> None:
        self._values = values

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._values, dtype=float)


def test_visualizer_detection_path_renders_boxes() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=_DummyTensor([[8, 8, 40, 40]]),
            conf=_DummyTensor([0.87]),
            cls=_DummyTensor([0]),
        ),
        masks=None,
        keypoints=None,
        obb=None,
    )
    visualizer = BeautifyVisualizer(label_mapping={"aircraft": "飞机"}, renderer="auto")

    annotations = visualizer.adapt_result(result, names={0: "aircraft"})
    rendered = visualizer.draw(image, annotations, style=DrawStyle.from_image_size(64, 64))

    assert annotations.primary_labels() == ["aircraft"]
    assert annotations.boxes[0].confidence == 0.87
    assert np.any(rendered != image)


def test_auto_renderer_prefers_contour_when_mask_exists() -> None:
    annotations = AnnotationSet(
        boxes=(BoxAnno(box=(6, 6, 30, 30), label="aircraft", confidence=0.91),),
        masks=(
            MaskAnno(
                contours=(((6, 6), (30, 6), (30, 30), (6, 30)),),
                label="aircraft",
                confidence=0.91,
            ),
        ),
    )
    renderer = AutoRenderer(prefer_contour_when_mask_exists=True)

    assert renderer.select_renderer(annotations).name == "contour"


def test_auto_renderer_uses_box_when_no_masks_exist() -> None:
    annotations = AnnotationSet(boxes=(BoxAnno(box=(6, 6, 30, 30), label="aircraft", confidence=0.91),))
    visualizer = BeautifyVisualizer(renderer="auto")

    assert visualizer.choose_renderer_name(annotations) == "box"


def test_contour_renderer_falls_back_to_box_rendering() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    annotations = AnnotationSet(boxes=(BoxAnno(box=(10, 10, 36, 36), label="aircraft", confidence=0.75),))
    renderer = ContourRenderer()
    context = RenderContext(label_mapping={}, color_mapping={}, default_color=(0, 255, 0), use_label_mapping=True)

    rendered = renderer.render(image, annotations, style=DrawStyle.from_image_size(64, 64), context=context)

    assert np.any(rendered != image)


def test_load_pipeline_config_keeps_backward_compatible_visualization_fields(tmp_path: Path) -> None:
    yaml_path = tmp_path / "infer_pipeline.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "visualization:",
                "  enabled: true",
                "  use_label_mapping: false",
                "  label_mapping:",
                "    aircraft: 飞机",
                "  color_mapping:",
                "    aircraft: [1, 2, 3]",
                "  default_color: [4, 5, 6]",
                "  style:",
                "    text_bg_alpha: 0.55",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_pipeline_config(yaml_path)

    assert cfg.viz_enabled is True
    assert cfg.viz_adapter == "yolo"
    assert cfg.viz_renderer == "auto"
    assert cfg.viz_theme == "classic"
    assert cfg.prefer_contour_when_mask_exists is True
    assert cfg.use_label_mapping is False
    assert cfg.label_mapping["aircraft"] == "飞机"
    assert cfg.color_mapping["aircraft"] == (1, 2, 3)
    assert cfg.default_color == (4, 5, 6)
    assert cfg.style_overrides["text_bg_alpha"] == 0.55
