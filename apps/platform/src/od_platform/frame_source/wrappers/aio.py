#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Async wrapper for FrameSource."""

from __future__ import annotations

import asyncio
from typing import Optional

from od_platform.frame_source.core.base import FrameSource
from od_platform.frame_source.core.types import Frame


class AsyncSource:
    """Async iterator wrapper around a synchronous FrameSource."""

    def __init__(self, source: FrameSource) -> None:
        self.source = source

    async def __aenter__(self) -> "AsyncSource":
        ok = await asyncio.to_thread(self.source.open)
        if not ok:
            raise RuntimeError(f"failed to open frame source: {self.source.source_path}")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await asyncio.to_thread(self.source.close)

    def __aiter__(self) -> "AsyncSource":
        return self

    async def __anext__(self) -> Frame:
        frame: Optional[Frame] = await asyncio.to_thread(self.source.read)
        if frame is None:
            raise StopAsyncIteration
        return frame

