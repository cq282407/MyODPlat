#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Frame source wrappers."""

from __future__ import annotations

from od_platform.frame_source.wrappers.aio import AsyncSource
from od_platform.frame_source.wrappers.threaded import ThreadedSource

__all__ = ["AsyncSource", "ThreadedSource"]

