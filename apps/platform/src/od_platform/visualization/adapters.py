#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project   : ODPlatform
# @Function  : YOLO result adapter to unified visualization annotations.
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from od_platform.visualization.schema import AnnotationSet, BoxAnno, KeypointsAnno, LabelAnno, MaskAnno, PolygonAnno


class YOLOResultAdapter:
    """Convert Ultralytics result objects into a unified annotation schema."""

    name = "yolo"

    def adapt(self, result: Any, *, names: Mapping[int, str] | None = None) -> AnnotationSet:
        name_map = dict(names or {})
        boxes = self._adapt_boxes(getattr(result, "boxes", None), name_map)
        masks = self._adapt_masks(getattr(result, "masks", None), boxes)
        polygons = self._adapt_obb(getattr(result, "obb", None), name_map)
        keypoints = self._adapt_keypoints(getattr(result, "keypoints", None), boxes)
        labels = self._build_labels(boxes=boxes, masks=masks, polygons=polygons, keypoints=keypoints)
        return AnnotationSet(
            boxes=boxes,
            polygons=polygons,
            masks=masks,
            keypoints=keypoints,
            labels=labels,
        )

    def _adapt_boxes(self, boxes: Any, names: Mapping[int, str]) -> tuple[BoxAnno, ...]:
        xyxy = _to_xyxy_array(getattr(boxes, "xyxy", None))
        if xyxy.size == 0:
            return ()
        class_ids = _to_int_list(getattr(boxes, "cls", None))
        confidences = _to_float_list(getattr(boxes, "conf", None))
        out: list[BoxAnno] = []
        for idx, box in enumerate(xyxy):
            class_id = class_ids[idx] if idx < len(class_ids) else None
            confidence = float(confidences[idx]) if idx < len(confidences) else None
            label = _label_for(class_id, idx, names)
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            out.append(
                BoxAnno(
                    box=(x1, y1, x2, y2),
                    label=label,
                    confidence=confidence,
                    class_id=class_id,
                    source="box",
                )
            )
        return tuple(out)

    def _adapt_masks(self, masks: Any, boxes: tuple[BoxAnno, ...]) -> tuple[MaskAnno, ...]:
        raw_items = getattr(masks, "xy", None)
        if raw_items is None:
            return ()
        out: list[MaskAnno] = []
        for idx, item in enumerate(raw_items):
            contour = _normalize_points(_to_plain_list(item))
            if len(contour) < 3:
                continue
            label, confidence, class_id = _meta_for_index(boxes, idx)
            out.append(
                MaskAnno(
                    contours=(contour,),
                    label=label,
                    confidence=confidence,
                    class_id=class_id,
                    source="mask",
                )
            )
        return tuple(out)

    def _adapt_obb(self, obb: Any, names: Mapping[int, str]) -> tuple[PolygonAnno, ...]:
        raw_polygons = _to_plain_list(getattr(obb, "xyxyxyxy", None))
        if not raw_polygons:
            return ()
        class_ids = _to_int_list(getattr(obb, "cls", None))
        confidences = _to_float_list(getattr(obb, "conf", None))
        out: list[PolygonAnno] = []
        for idx, item in enumerate(raw_polygons):
            points = _normalize_points(item)
            if len(points) < 4:
                continue
            class_id = class_ids[idx] if idx < len(class_ids) else None
            confidence = float(confidences[idx]) if idx < len(confidences) else None
            out.append(
                PolygonAnno(
                    points=points,
                    label=_label_for(class_id, idx, names),
                    confidence=confidence,
                    class_id=class_id,
                    source="obb",
                )
            )
        return tuple(out)

    def _adapt_keypoints(self, keypoints: Any, boxes: tuple[BoxAnno, ...]) -> tuple[KeypointsAnno, ...]:
        raw_xy = _to_plain_list(getattr(keypoints, "xy", None))
        if not raw_xy:
            return ()
        raw_conf = _to_plain_list(getattr(keypoints, "conf", None))
        out: list[KeypointsAnno] = []
        for idx, item in enumerate(raw_xy):
            point_conf_row = raw_conf[idx] if idx < len(raw_conf) and isinstance(raw_conf[idx], list) else []
            points = _normalize_keypoints(item, point_conf_row)
            if not points:
                continue
            label, confidence, class_id = _meta_for_index(boxes, idx)
            out.append(
                KeypointsAnno(
                    keypoints=points,
                    label=label,
                    confidence=confidence,
                    class_id=class_id,
                    source="keypoints",
                )
            )
        return tuple(out)

    def _build_labels(
        self,
        *,
        boxes: tuple[BoxAnno, ...],
        masks: tuple[MaskAnno, ...],
        polygons: tuple[PolygonAnno, ...],
        keypoints: tuple[KeypointsAnno, ...],
    ) -> tuple[LabelAnno, ...]:
        source = boxes or masks or polygons or keypoints
        return tuple(
            LabelAnno(
                label=item.label,
                confidence=getattr(item, "confidence", None),
                class_id=getattr(item, "class_id", None),
                source=getattr(item, "source", "label"),
            )
            for item in source
        )


def _meta_for_index(boxes: tuple[BoxAnno, ...], idx: int) -> tuple[str, float | None, int | None]:
    if idx < len(boxes):
        item = boxes[idx]
        return item.label, item.confidence, item.class_id
    return f"instance_{idx}", None, None


def _label_for(class_id: int | None, idx: int, names: Mapping[int, str]) -> str:
    if class_id is not None:
        return str(names.get(class_id, class_id))
    return f"instance_{idx}"


def _to_xyxy_array(value: Any) -> np.ndarray:
    raw = _to_plain_list(value)
    if not raw:
        return np.zeros((0, 4), dtype=float)
    arr = np.asarray(raw, dtype=float)
    return arr.reshape((-1, 4)) if arr.size else np.zeros((0, 4), dtype=float)


def _normalize_points(value: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        return ()
    points: list[tuple[int, int]] = []
    if value and isinstance(value[0], (list, tuple)):
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            points.append((int(round(float(item[0]))), int(round(float(item[1])))))
        return tuple(points)
    if len(value) % 2 != 0:
        return ()
    for index in range(0, len(value), 2):
        points.append((int(round(float(value[index]))), int(round(float(value[index + 1])))))
    return tuple(points)


def _normalize_keypoints(points: Any, confidences: list[Any]) -> tuple[tuple[int, int, float | None], ...]:
    if not isinstance(points, list):
        return ()
    out: list[tuple[int, int, float | None]] = []
    for idx, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        point_conf = None
        if idx < len(confidences):
            try:
                point_conf = float(confidences[idx])
            except (TypeError, ValueError):
                point_conf = None
        out.append((int(round(float(point[0]))), int(round(float(point[1]))), point_conf))
    return tuple(out)


def _to_plain_list(value: Any) -> list[Any]:
    obj = value
    if obj is None:
        return []
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
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    return [obj]


def _to_float_list(value: Any) -> list[float]:
    out: list[float] = []
    for item in _to_plain_list(value):
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _to_int_list(value: Any) -> list[int]:
    out: list[int] = []
    for item in _to_plain_list(value):
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out
