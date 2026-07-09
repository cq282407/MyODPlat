#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : frame_source.py
# @Project   : ODPlatform
# @Function  : Frame source abstraction for D8 inference.
from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from od_platform.common.paths import ROOT_DIR

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
_REMOTE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "tcp://")
_BACKEND_MAP = {
    "auto": 0,
    "msmf": getattr(cv2, "CAP_MSMF", 0),
    "dshow": getattr(cv2, "CAP_DSHOW", 0),
    "v4l2": getattr(cv2, "CAP_V4L2", 0),
    "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", 0),
    "ffmpeg": getattr(cv2, "CAP_FFMPEG", 0),
}


class SourceType(str, Enum):
    IMAGE = "image"
    IMAGES = "images"
    VIDEO = "video"
    CAMERA = "camera"


@dataclass
class CameraConfig:
    camera_id: int = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    backend: str | None = None
    codec: str | None = None
    buffer_size: int | None = None
    auto_focus: bool | None = None


@dataclass
class FrameInfo:
    frame_index: int
    filename: str | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    source_type: SourceType | None = None


@dataclass
class Frame:
    image: np.ndarray
    info: FrameInfo


class FrameSource(AbstractContextManager["FrameSource"]):
    """Base frame source."""

    def __enter__(self) -> "FrameSource":
        return self

    def __iter__(self) -> Iterator[Frame]:
        raise NotImplementedError

    def get_source_type(self) -> SourceType:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class _ImageFrameSource(FrameSource):
    def __init__(self, image_path: Path) -> None:
        self._image_path = image_path

    def __iter__(self) -> Iterator[Frame]:
        image = cv2.imread(str(self._image_path))
        if image is None:
            raise FileNotFoundError(f"读取图片失败: {self._image_path}")
        h, w = image.shape[:2]
        yield Frame(
            image=image,
            info=FrameInfo(
                frame_index=0,
                filename=self._image_path.name,
                width=w,
                height=h,
                source_type=SourceType.IMAGE,
            ),
        )

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGE


class _ImageDirectoryFrameSource(FrameSource):
    def __init__(self, image_dir: Path) -> None:
        self._image_dir = image_dir
        self._files = sorted(
            path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __iter__(self) -> Iterator[Frame]:
        for index, image_path in enumerate(self._files):
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("跳过无法读取的图片: %s", image_path)
                continue
            h, w = image.shape[:2]
            yield Frame(
                image=image,
                info=FrameInfo(
                    frame_index=index,
                    filename=image_path.name,
                    width=w,
                    height=h,
                    source_type=SourceType.IMAGES,
                ),
            )

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGES


class _VideoCaptureFrameSource(FrameSource):
    def __init__(self, source: str | int, *, camera_config: CameraConfig | None, source_type: SourceType) -> None:
        self._source = source
        self._camera_config = camera_config
        self._source_type = source_type
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> "_VideoCaptureFrameSource":
        self._capture = self._open_capture()
        return self

    def __iter__(self) -> Iterator[Frame]:
        if self._capture is None:
            self._capture = self._open_capture()
        capture = self._capture
        assert capture is not None

        frame_index = 0
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
        while True:
            ok, image = capture.read()
            if not ok or image is None:
                break
            h, w = image.shape[:2]
            filename = None if self._source_type == SourceType.CAMERA else Path(str(self._source)).name
            yield Frame(
                image=image,
                info=FrameInfo(
                    frame_index=frame_index,
                    filename=filename,
                    fps=fps,
                    width=w,
                    height=h,
                    source_type=self._source_type,
                ),
            )
            frame_index += 1

    def get_source_type(self) -> SourceType:
        return self._source_type

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _open_capture(self) -> cv2.VideoCapture:
        capture_source = self._source
        backend = 0
        if self._source_type == SourceType.CAMERA and self._camera_config is not None:
            backend_name = (self._camera_config.backend or "auto").lower()
            backend = _BACKEND_MAP.get(backend_name, 0)
            capture_source = int(self._camera_config.camera_id)

        cap = cv2.VideoCapture(capture_source, backend) if backend else cv2.VideoCapture(capture_source)
        if not cap.isOpened():
            raise RuntimeError(f"打开视频源失败: {self._source}")

        if self._source_type == SourceType.CAMERA and self._camera_config is not None:
            if self._camera_config.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_config.width)
            if self._camera_config.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_config.height)
            if self._camera_config.fps:
                cap.set(cv2.CAP_PROP_FPS, float(self._camera_config.fps))
            if self._camera_config.buffer_size is not None and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self._camera_config.buffer_size))
            if self._camera_config.auto_focus is not None and hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self._camera_config.auto_focus else 0)
            if self._camera_config.codec:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._camera_config.codec[:4]))
        return cap


def create_frame_source(source: str | Path | int, camera_config: CameraConfig | None = None) -> FrameSource:
    """Create a frame source from camera / image / directory / video / remote URL."""

    if isinstance(source, int):
        return _VideoCaptureFrameSource(source, camera_config=camera_config, source_type=SourceType.CAMERA)

    raw = str(source).strip()
    if raw.isdigit():
        cfg = camera_config or CameraConfig(camera_id=int(raw))
        cfg.camera_id = int(raw)
        return _VideoCaptureFrameSource(int(raw), camera_config=cfg, source_type=SourceType.CAMERA)

    if raw.lower().startswith(_REMOTE_PREFIXES):
        return _VideoCaptureFrameSource(raw, camera_config=camera_config, source_type=SourceType.CAMERA)

    path = Path(raw)
    if not path.is_absolute():
        cwd_candidate = (Path.cwd() / path).resolve()
        root_candidate = (ROOT_DIR / path).resolve()
        path = cwd_candidate if cwd_candidate.exists() else root_candidate if root_candidate.exists() else path

    if path.is_dir():
        return _ImageDirectoryFrameSource(path)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return _ImageFrameSource(path)
    if path.suffix.lower() in VIDEO_EXTENSIONS or len(path.parts) > 1 or path.exists():
        return _VideoCaptureFrameSource(str(path), camera_config=camera_config, source_type=SourceType.VIDEO)
    return _VideoCaptureFrameSource(raw, camera_config=camera_config, source_type=SourceType.VIDEO)
