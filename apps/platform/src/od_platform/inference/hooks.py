#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : hooks.py
# @Project   : ODPlatform
# @Function  : Inference lifecycle hooks.
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameEvent:
    frame_idx: int
    image: np.ndarray
    annotated: np.ndarray
    n_detections: int
    detections: list | None = None


@dataclass
class ProgressEvent:
    frame_idx: int
    total_frames: int | None
    elapsed_sec: float
    fps_loop: float
    fps_infer: float
    detections_total: int


@dataclass
class InferHooks:
    on_frame: Callable[[FrameEvent], None] | None = None
    on_progress: Callable[[ProgressEvent], None] | None = None
    on_complete: Callable[[Any], None] | None = None
    on_error: Callable[[Exception], None] | None = None
    progress_interval_frames: int = 30

    def fire_frame(self, evt: FrameEvent) -> None:
        if self.on_frame is None:
            return
        try:
            self.on_frame(evt)
        except Exception as exc:
            logger.warning("on_frame 回调异常 (已吞): %s", exc)

    def fire_progress(self, evt: ProgressEvent) -> None:
        if self.on_progress is None:
            return
        try:
            self.on_progress(evt)
        except Exception as exc:
            logger.warning("on_progress 回调异常 (已吞): %s", exc)

    def fire_complete(self, result: Any) -> None:
        if self.on_complete is None:
            return
        try:
            self.on_complete(result)
        except Exception as exc:
            logger.warning("on_complete 回调异常 (已吞): %s", exc)

    def fire_error(self, exc: Exception) -> None:
        if self.on_error is None:
            return
        try:
            self.on_error(exc)
        except Exception as inner:
            logger.warning("on_error 回调异常 (已吞): %s", inner)
