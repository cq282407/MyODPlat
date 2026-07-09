#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np
import pytest

from od_platform.frame_source import (
    CameraConfig,
    CameraSource,
    ImageFolderSource,
    ImageSource,
    SourceType,
    VideoSource,
    create_async_source,
    create_frame_source,
    create_threaded_source,
)


def test_image_source_reads_single_frame_and_resets_on_open(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    _write_image(image_path, value=80)

    src = ImageSource(str(image_path))
    assert src.open() is True
    frame = src.read()
    assert frame is not None
    assert frame.info.source_type == SourceType.IMAGE
    assert frame.info.filename == "sample.jpg"
    assert src.read() is None
    src.close()

    assert src.open() is True
    assert src.read() is not None


def test_image_folder_source_supports_stride_and_seek(tmp_path: Path) -> None:
    for index in range(5):
        _write_image(tmp_path / f"{index}.jpg", value=20 + index)

    with ImageFolderSource(str(tmp_path), stride=2) as src:
        assert [frame.info.frame_index for frame in src] == [0, 2, 4]

    with ImageFolderSource(str(tmp_path)) as src:
        assert src.seek(frame=3) is True
        frame = src.read()
        assert frame is not None
        assert frame.info.frame_index == 3


def test_factory_detects_sources_and_fails_fast(tmp_path: Path) -> None:
    image_path = tmp_path / "a.png"
    video_path = tmp_path / "a.avi"
    _write_image(image_path)
    _write_video(video_path)

    assert isinstance(create_frame_source("0"), CameraSource)
    assert isinstance(create_frame_source(image_path), ImageSource)
    assert isinstance(create_frame_source(tmp_path), ImageFolderSource)
    assert isinstance(create_frame_source(video_path), VideoSource)
    assert isinstance(create_frame_source("rtsp://example.invalid/live"), VideoSource)

    with pytest.raises(ValueError):
        create_frame_source(tmp_path / "missing.jpg")


def test_video_source_reads_and_seeks(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.avi"
    _write_video(video_path, frame_count=6)

    with VideoSource(str(video_path), stride=2) as src:
        first = src.read()
        second = src.read()
        assert first is not None
        assert second is not None
        assert first.info.frame_index == 0
        assert second.info.frame_index == 2
        assert src.seek(frame=4) is True
        frame = src.read()
        assert frame is not None
        assert frame.info.frame_index == 4


def test_threaded_and_async_wrappers_read_frames(tmp_path: Path) -> None:
    for index in range(3):
        _write_image(tmp_path / f"{index}.jpg", value=40 + index)

    with create_threaded_source(tmp_path, buffer_size=2) as src:
        assert src.read() is not None

    async def _read_one() -> int:
        async with create_async_source(tmp_path) as src:
            frame = await src.__anext__()
            return frame.info.frame_index

    assert asyncio.run(_read_one()) == 0


def test_camera_config_rejects_unknown_fields() -> None:
    with pytest.raises(Exception):
        CameraConfig(width=1280, typo=True)  # type: ignore[call-arg]


def test_camera_config_matches_teacher_style() -> None:
    source = create_frame_source(
        "0",
        camera_config=CameraConfig(camera_id=0, backend="msmf", codec="MJPG", fps=90),
    )

    assert isinstance(source, CameraSource)
    assert source.camera_config.backend.value == "msmf"
    assert source.camera_config.codec.value == "MJPG"
    assert source.camera_config.fps == 90


def _write_image(path: Path, value: int = 255) -> None:
    image = np.full((16, 24, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_video(path: Path, frame_count: int = 4) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (24, 16))
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((16, 24, 3), 20 + index, dtype=np.uint8))
    writer.release()
