#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Factory helpers for frame sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from od_platform.frame_source.core.config import CameraConfig
from od_platform.frame_source.core.types import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from od_platform.frame_source.sources.camera import CameraSource
from od_platform.frame_source.sources.image import ImageFolderSource, ImageSource
from od_platform.frame_source.sources.video import VideoSource
from od_platform.frame_source.wrappers.aio import AsyncSource
from od_platform.frame_source.wrappers.threaded import ThreadedSource

NETWORK_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")


def create_frame_source(
    source: str | int | Path,
    *,
    camera_config: CameraConfig | None = None,
    stride: int = 1,
) -> CameraSource | VideoSource | ImageSource | ImageFolderSource:
    """Create a concrete source from a camera id, path or network URL."""

    source_text = str(source)
    if source_text.isdigit():
        config = camera_config or CameraConfig(camera_id=int(source_text))
        if camera_config is not None:
            config = camera_config.model_copy(update={"camera_id": int(source_text)})
        return CameraSource(config, stride=stride)

    lowered = source_text.lower()
    if lowered.startswith(NETWORK_PREFIXES):
        return VideoSource(source_text, stride=stride)

    path = Path(source_text)
    if not path.exists():
        raise ValueError(f"source path does not exist: {source_text}")
    if path.is_dir():
        return ImageFolderSource(str(path), stride=stride)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return ImageSource(str(path), stride=stride)
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return VideoSource(str(path), stride=stride)
    raise ValueError(f"unsupported source format: {source_text}")


def create_threaded_source(
    source: str | int | Path,
    *,
    camera_config: CameraConfig | None = None,
    stride: int = 1,
    buffer_size: int = 1,
    drop_oldest: bool = True,
) -> ThreadedSource:
    return ThreadedSource(
        create_frame_source(source, camera_config=camera_config, stride=stride),
        buffer_size=buffer_size,
        drop_oldest=drop_oldest,
    )


def create_async_source(
    source: str | int | Path,
    *,
    camera_config: CameraConfig | None = None,
    stride: int = 1,
    threaded: bool = False,
    **threaded_kwargs: Any,
) -> AsyncSource:
    base = create_threaded_source(
        source,
        camera_config=camera_config,
        stride=stride,
        **threaded_kwargs,
    ) if threaded else create_frame_source(source, camera_config=camera_config, stride=stride)
    return AsyncSource(base)
