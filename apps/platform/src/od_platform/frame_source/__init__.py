#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Unified frame input sources for images, folders, videos and cameras."""

from __future__ import annotations

from od_platform.frame_source.core.base import FrameSource
from od_platform.frame_source.core.config import CameraBackend, CameraCodec, CameraConfig
from od_platform.frame_source.core.types import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    Frame,
    FrameInfo,
    SourceType,
)
from od_platform.frame_source.factory import (
    create_async_source,
    create_frame_source,
    create_threaded_source,
)
from od_platform.frame_source.registry import list_sources, register_source
from od_platform.frame_source.sources.camera import CameraSource
from od_platform.frame_source.sources.image import ImageFolderSource, ImageSource
from od_platform.frame_source.sources.video import VideoSource
from od_platform.frame_source.wrappers.aio import AsyncSource
from od_platform.frame_source.wrappers.threaded import ThreadedSource

__all__ = [
    "AsyncSource",
    "CameraBackend",
    "CameraCodec",
    "CameraConfig",
    "CameraSource",
    "Frame",
    "FrameInfo",
    "FrameSource",
    "IMAGE_EXTENSIONS",
    "ImageFolderSource",
    "ImageSource",
    "SourceType",
    "ThreadedSource",
    "VIDEO_EXTENSIONS",
    "VideoSource",
    "create_async_source",
    "create_frame_source",
    "create_threaded_source",
    "list_sources",
    "register_source",
]
