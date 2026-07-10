#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project   : ODPlatform
# @Function  : Shared draw style for visualization renderers.
from __future__ import annotations

from dataclasses import dataclass

_THEME_OVERRIDES: dict[str, dict[str, object]] = {
    "classic": {},
    "contour": {
        "box_thickness": 2,
        "contour_thickness": 3,
        "contour_fill_alpha": 0.18,
    },
    "bold": {
        "box_thickness": 3,
        "contour_thickness": 4,
        "text_bg_alpha": 0.7,
    },
}


@dataclass(frozen=True)
class DrawStyle:
    box_thickness: int
    contour_thickness: int
    text_scale: float
    text_thickness: int
    text_color: tuple[int, int, int]
    text_bg_alpha: float
    label_padding: int = 4
    point_radius: int = 3
    contour_fill_alpha: float = 0.16

    @classmethod
    def from_image_size(cls, height: int, width: int, *, theme: str = "classic", **overrides) -> "DrawStyle":
        base = max(1, min(height, width) // 320)
        default = cls(
            box_thickness=max(2, base),
            contour_thickness=max(2, base + 1),
            text_scale=max(0.5, min(height, width) / 1400.0),
            text_thickness=max(1, base),
            text_color=(255, 255, 255),
            text_bg_alpha=0.65,
            label_padding=4,
            point_radius=max(2, base + 1),
            contour_fill_alpha=0.16,
        )
        theme_payload = _THEME_OVERRIDES.get((theme or "classic").lower(), {})
        payload = {**default.__dict__, **theme_payload, **overrides}
        if "text_color" in payload and isinstance(payload["text_color"], list):
            payload["text_color"] = tuple(int(v) for v in payload["text_color"])
        return cls(**payload)
