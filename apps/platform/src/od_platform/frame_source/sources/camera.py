#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Camera frame source with explicit backend and parameter negotiation."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import cv2

from od_platform.frame_source.core.base import FrameSource
from od_platform.frame_source.core.config import CameraBackend, CameraConfig
from od_platform.frame_source.core.types import Frame, FrameInfo, SourceType
from od_platform.frame_source.registry import register_source

logger = logging.getLogger(__name__)


@register_source(lambda s: s.isdigit(), priority=10)
class CameraSource(FrameSource):
    """FrameSource for a local camera device."""

    def __init__(
        self,
        camera_id: int | str | CameraConfig = 0,
        *,
        camera_config: CameraConfig | None = None,
        stride: int = 1,
    ) -> None:
        if isinstance(camera_id, CameraConfig):
            camera_config = camera_id
            camera_id = camera_config.camera_id
        self.camera_config = camera_config or CameraConfig(camera_id=int(camera_id))
        self.camera_id = self.camera_config.camera_id
        super().__init__(str(self.camera_id), stride=stride)
        self._cap: Optional[cv2.VideoCapture] = None
        self._start_time = 0.0
        self._frame_index = 0
        self.actual_width: Optional[int] = None
        self.actual_height: Optional[int] = None
        self.actual_fps: Optional[float] = None

    def open(self) -> bool:
        self.close()
        self._frame_index = 0
        self._start_time = time.perf_counter()

        if self.camera_config.backend == CameraBackend.MSMF:
            os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

        backend = _opencv_backend(self.camera_config.backend)
        self._cap = (
            cv2.VideoCapture(self.camera_id, backend)
            if backend is not None
            else cv2.VideoCapture(self.camera_id)
        )
        if not self._cap.isOpened():
            logger.error("failed to open camera: %s", self.camera_id)
            return False

        cfg = self.camera_config
        if cfg.width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        if cfg.height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        if cfg.codec is not None:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.codec.value))
        if cfg.fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, cfg.fps)

        # Some backends finish negotiation only after the first read.
        self._cap.read()
        self.actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.actual_fps = fps if fps > 0 else None
        self._frame_index = 0
        return True

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None

        ok, image = self._cap.read()
        if not ok or image is None:
            return None

        frame_index = self._frame_index
        self._frame_index += 1
        height, width = image.shape[:2]
        return Frame(
            image=image,
            info=FrameInfo(
                width=width,
                height=height,
                source_type=SourceType.CAMERA,
                source_path=self.source_path,
                frame_index=frame_index,
                timestamp=time.perf_counter() - self._start_time,
                fps=self.actual_fps,
            ),
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    def get_source_type(self) -> SourceType:
        return SourceType.CAMERA

    def set_stride(self, stride: int) -> None:
        if stride != 1:
            logger.warning("camera sources ignore stride; use ThreadedSource for realtime downsampling")
        self._stride = 1


def _opencv_backend(backend: CameraBackend) -> Optional[int]:
    if backend == CameraBackend.AUTO:
        return None
    mapping = {
        CameraBackend.MSMF: getattr(cv2, "CAP_MSMF", None),
        CameraBackend.DSHOW: getattr(cv2, "CAP_DSHOW", None),
        CameraBackend.V4L2: getattr(cv2, "CAP_V4L2", None),
        CameraBackend.AVFOUNDATION: getattr(cv2, "CAP_AVFOUNDATION", None),
    }
    return mapping.get(backend)
