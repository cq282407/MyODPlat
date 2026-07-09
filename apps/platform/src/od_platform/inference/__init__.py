#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Project   : ODPlatform
# @Function  : Inference subsystem public API.
from od_platform.inference.cancel import CancelToken, InferenceCancelled
from od_platform.inference.hooks import FrameEvent, InferHooks, ProgressEvent
from od_platform.inference.pipeline_config import PipelineConfig, load_pipeline_config
from od_platform.inference.service import InferResult, InferService, InferStats, infer_yolo, log_infer_stats
from od_platform.inference.sinks import LocalFileSink, NullSink, OutputSink

__all__ = [
    "infer_yolo",
    "InferService",
    "InferResult",
    "InferStats",
    "log_infer_stats",
    "PipelineConfig",
    "load_pipeline_config",
    "OutputSink",
    "LocalFileSink",
    "NullSink",
    "InferHooks",
    "FrameEvent",
    "ProgressEvent",
    "CancelToken",
    "InferenceCancelled",
]
