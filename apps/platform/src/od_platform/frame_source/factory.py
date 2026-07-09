#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Factory helpers for frame sources — registry-based (auto-registration).

每个帧源通过 @register_source(priority=N) 自动注册, factory 遍历注册表匹配.
新增源类型只需加装饰器, 无需改 factory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from od_platform.frame_source.core.config import CameraConfig
from od_platform.frame_source.registry import find_matching_source
from od_platform.frame_source.sources.camera import CameraSource  # noqa: F401  注册表副作用
from od_platform.frame_source.sources.image import ImageFolderSource, ImageSource  # noqa: F401
from od_platform.frame_source.sources.video import VideoSource  # noqa: F401
from od_platform.frame_source.wrappers.aio import AsyncSource
from od_platform.frame_source.wrappers.threaded import ThreadedSource


def create_frame_source(
    source: str | int | Path,
    *,
    camera_config: CameraConfig | None = None,
    stride: int = 1,
):
    """Create a concrete source from a camera id, path or network URL.

    遍历注册表, 第一个匹配的源类被实例化.
    camera_config 仅当匹配 CameraSource 时生效.
    """
    source_text = str(source)
    source_cls = find_matching_source(source_text)

    if source_cls is None:
        raise ValueError(f"unsupported source format: {source_text}")

    # CameraSource 特殊处理: 需要注入 camera_config
    if source_cls is CameraSource:
        config = camera_config or CameraConfig(camera_id=int(source_text))
        if camera_config is not None:
            config = camera_config.model_copy(update={"camera_id": int(source_text)})
        return CameraSource(config, stride=stride)

    # 非摄像头源: 路径不存在则报错
    path = Path(source_text)
    if not source_text.isdigit() and not path.exists():
        raise ValueError(f"source path does not exist: {source_text}")

    return source_cls(source_text, stride=stride)


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