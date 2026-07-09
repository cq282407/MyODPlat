#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Video and network-stream frame source."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2

from od_platform.frame_source.core.base import FrameSource
from od_platform.frame_source.core.types import Frame, FrameInfo, SourceType, VIDEO_EXTENSIONS
from od_platform.frame_source.registry import register_source

logger = logging.getLogger(__name__)

NETWORK_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")

def _is_video_source(source_text: str) -> bool:
    lowered = source_text.lower()
    if lowered.startswith(NETWORK_PREFIXES):
        return True
    from pathlib import Path
    p = Path(source_text)
    return p.suffix.lower() in VIDEO_EXTENSIONS


@register_source(_is_video_source, priority=20)
class VideoSource(FrameSource):
    """FrameSource for video files and OpenCV-readable network streams."""

    def __init__(self, video_path: str, *, stride: int = 1) -> None:
        super().__init__(video_path, stride=stride)
        self._cap: Optional[cv2.VideoCapture] = None
        self._next_index = 0
        self._total_frames: Optional[int] = None
        self._fps: Optional[float] = None

    def open(self) -> bool:
        self.close()
        self._next_index = 0
        self._cap = cv2.VideoCapture(self.source_path)
        if not self._cap.isOpened():
            logger.error("failed to open video source: %s", self.source_path)
            return False

        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._fps = fps if fps > 0 else None
        self._total_frames = total if total > 0 else None
        return True

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None

        frame_index = self._next_index
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        self._next_index += 1

        if self._stride > 1:
            for _ in range(self._stride - 1):
                if not self._cap.grab():
                    break
                self._next_index += 1

        height, width = image.shape[:2]
        timestamp = frame_index / self._fps if self._fps else 0.0
        return Frame(
            image=image,
            info=FrameInfo(
                width=width,
                height=height,
                source_type=SourceType.VIDEO,
                source_path=self.source_path,
                frame_index=frame_index,
                total_frames=self._total_frames,
                timestamp=timestamp,
                fps=self._fps,
                filename=Path(self.source_path).name,
            ),
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    def get_source_type(self) -> SourceType:
        return SourceType.VIDEO

    @property
    def seekable(self) -> bool:
        return True

    def seek(self, frame: Optional[int] = None, time_sec: Optional[float] = None) -> bool:
        if self._cap is None:
            return False
        if frame is None:
            if time_sec is None or not self._fps:
                return False
            frame = int(time_sec * self._fps)

        target = max(0, int(frame))
        if self._total_frames is not None:
            target = min(target, max(0, self._total_frames - 1))
        if not self._cap.set(cv2.CAP_PROP_POS_FRAMES, target):
            return False
        self._next_index = target
        return True
