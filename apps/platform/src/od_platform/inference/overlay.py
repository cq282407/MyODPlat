#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : overlay.py
# @Project   : ODPlatform
# @Function  : HUD overlay and FPS metrics.
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


class FPSCounter:
    def __init__(self, window_size: int = 30) -> None:
        self._samples: deque[float] = deque(maxlen=window_size)
        self._last_ms: float = 0.0

    def update(self, ms: float) -> None:
        if ms > 0:
            self._samples.append(ms)
            self._last_ms = ms

    @property
    def fps(self) -> float:
        if not self._samples:
            return 0.0
        avg = sum(self._samples) / len(self._samples)
        return 1000.0 / avg if avg > 0 else 0.0

    @property
    def inst(self) -> float:
        return 1000.0 / self._last_ms if self._last_ms > 0 else 0.0


@dataclass
class Metrics:
    capture: FPSCounter = field(default_factory=FPSCounter)
    infer: FPSCounter = field(default_factory=FPSCounter)
    render: FPSCounter = field(default_factory=FPSCounter)
    loop: FPSCounter = field(default_factory=FPSCounter)
    _spp: float = 0.0
    _sinf: float = 0.0
    _spost: float = 0.0
    _sn: int = 0

    def add_speed(self, speed: dict | None) -> None:
        if not speed:
            return
        self._spp += float(speed.get("preprocess", 0.0))
        self._sinf += float(speed.get("inference", 0.0))
        self._spost += float(speed.get("postprocess", 0.0))
        self._sn += 1
        if speed.get("inference"):
            self.infer.update(float(speed["inference"]))

    def speed_avg_ms(self) -> dict[str, float]:
        if not self._sn:
            return {}
        return {
            "preprocess": round(self._spp / self._sn, 2),
            "inference": round(self._sinf / self._sn, 2),
            "postprocess": round(self._spost / self._sn, 2),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "capture_fps": round(self.capture.fps, 1),
            "infer_fps": round(self.infer.fps, 1),
            "render_fps": round(self.render.fps, 1),
            "loop_fps": round(self.loop.fps, 1),
            "current_fps": round(self.loop.inst, 1),
            "speed_ms": self.speed_avg_ms(),
        }


_C_LABEL = (200, 200, 200)
_C_CAPTURE = (0, 200, 255)
_C_INFER = (0, 230, 0)
_C_RENDER = (255, 120, 220)
_C_LOOP = (255, 200, 0)
_C_CURRENT = (255, 255, 255)
_C_OBJ = (120, 220, 255)


def draw_hud(frame, metrics: Metrics, *, n_dets: int = 0, recording: bool = False, show_info: bool = True) -> None:
    import cv2

    if not show_info:
        return

    rows = [
        ("Capture", f"{metrics.capture.fps:5.1f} FPS", _C_CAPTURE),
        ("Infer", f"{metrics.infer.fps:5.1f} FPS", _C_INFER),
        ("Render", f"{metrics.render.fps:5.1f} FPS", _C_RENDER),
        ("Loop", f"{metrics.loop.fps:5.1f} FPS", _C_LOOP),
        ("Current", f"{metrics.loop.inst:5.1f} FPS", _C_CURRENT),
        ("Objects", f"{n_dets}", _C_OBJ),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness, row_height = 0.5, 1, 22
    padding = 10
    label_width = 84
    panel_width = 230
    panel_height = padding * 2 + row_height * len(rows)
    x0, y0 = 12, 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_width, y0 + panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + 4, y0 + panel_height), (0, 200, 255), -1)

    y = y0 + padding + 14
    for label, value, color in rows:
        cv2.putText(frame, label, (x0 + padding + 6, y), font, font_scale, _C_LABEL, thickness, cv2.LINE_AA)
        cv2.putText(frame, value, (x0 + padding + label_width, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += row_height

    if recording:
        height, width = frame.shape[:2]
        cv2.circle(frame, (width - 70, 24), 7, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (width - 56, 30), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


def draw_pause(frame) -> None:
    import cv2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text, font_scale, thickness = "PAUSED", 1.8, 3
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    center_x, center_y = (width - text_width) // 2, (height + text_height) // 2
    cv2.putText(frame, text, (center_x, center_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    hint = "Press SPACE to resume  |  Q / Esc to quit"
    (hint_width, _), _ = cv2.getTextSize(hint, font, 0.6, 1)
    cv2.putText(frame, hint, ((width - hint_width) // 2, center_y + 44), font, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
