#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Project   : ODPlatform
# @Function  : Training subsystem public API.
from od_platform.common.result import TrainMetrics
from od_platform.training.service import TrainResult, TrainService, train_yolo

__all__ = ["TrainService", "TrainResult", "TrainMetrics", "train_yolo"]
