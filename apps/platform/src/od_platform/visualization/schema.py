#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project   : ODPlatform
# @Function  : Unified annotation schema for visualization rendering.
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

Point: TypeAlias = tuple[int, int]
Contour: TypeAlias = tuple[Point, ...]
Keypoint: TypeAlias = tuple[int, int, float | None]


@dataclass(frozen=True)
class BoxAnno:
    box: tuple[int, int, int, int]
    label: str
    confidence: float | None = None
    class_id: int | None = None
    source: str = "box"


@dataclass(frozen=True)
class PolygonAnno:
    points: Contour
    label: str
    confidence: float | None = None
    class_id: int | None = None
    source: str = "polygon"


@dataclass(frozen=True)
class MaskAnno:
    contours: tuple[Contour, ...]
    label: str
    confidence: float | None = None
    class_id: int | None = None
    source: str = "mask"


@dataclass(frozen=True)
class KeypointsAnno:
    keypoints: tuple[Keypoint, ...]
    label: str
    confidence: float | None = None
    class_id: int | None = None
    source: str = "keypoints"


@dataclass(frozen=True)
class LabelAnno:
    label: str
    confidence: float | None = None
    class_id: int | None = None
    source: str = "label"


@dataclass(frozen=True)
class AnnotationSet:
    boxes: tuple[BoxAnno, ...] = ()
    polygons: tuple[PolygonAnno, ...] = ()
    masks: tuple[MaskAnno, ...] = ()
    keypoints: tuple[KeypointsAnno, ...] = ()
    labels: tuple[LabelAnno, ...] = ()

    def has_contours(self) -> bool:
        return bool(self.masks or self.polygons)

    def primary_instances(self) -> tuple[object, ...]:
        if self.boxes:
            return self.boxes
        if self.masks:
            return self.masks
        if self.polygons:
            return self.polygons
        if self.keypoints:
            return self.keypoints
        return self.labels

    def primary_labels(self) -> list[str]:
        out: list[str] = []
        for item in self.primary_instances():
            label = getattr(item, "label", None)
            if label:
                out.append(str(label))
        return out
