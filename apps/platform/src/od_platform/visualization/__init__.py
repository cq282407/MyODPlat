#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project   : ODPlatform
# @Function  : Public visualization API for D8 inference.
from od_platform.visualization.adapters import YOLOResultAdapter
from od_platform.visualization.renderers import AutoRenderer, BoxRenderer, ContourRenderer
from od_platform.visualization.schema import AnnotationSet, BoxAnno, KeypointsAnno, LabelAnno, MaskAnno, PolygonAnno
from od_platform.visualization.style import DrawStyle
from od_platform.visualization.visualizer import BeautifyVisualizer

Detection = BoxAnno

__all__ = [
    "AnnotationSet",
    "AutoRenderer",
    "BeautifyVisualizer",
    "BoxAnno",
    "BoxRenderer",
    "ContourRenderer",
    "Detection",
    "DrawStyle",
    "KeypointsAnno",
    "LabelAnno",
    "MaskAnno",
    "PolygonAnno",
    "YOLOResultAdapter",
]
