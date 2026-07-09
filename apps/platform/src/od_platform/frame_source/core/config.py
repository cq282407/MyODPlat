#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Camera configuration for frame sources."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class CameraBackend(str, Enum):
    """OpenCV camera backend choices."""

    AUTO = "auto"
    MSMF = "msmf"
    DSHOW = "dshow"
    V4L2 = "v4l2"
    AVFOUNDATION = "avfoundation"


class CameraCodec(str, Enum):
    """Common camera codec requests."""

    MJPG = "MJPG"
    YUY2 = "YUY2"
    H264 = "H264"


class CameraConfig(BaseModel):
    """Requested camera parameters.

    OpenCV camera setters are requests, not guarantees. CameraSource reads
    values back after negotiation and exposes them through ``actual_*`` fields.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    camera_id: int = Field(default=0, ge=0)
    width: Optional[int] = Field(default=None, gt=0)
    height: Optional[int] = Field(default=None, gt=0)
    fps: Optional[float] = Field(default=None, gt=0)
    backend: CameraBackend = CameraBackend.AUTO
    codec: Optional[CameraCodec] = CameraCodec.MJPG
