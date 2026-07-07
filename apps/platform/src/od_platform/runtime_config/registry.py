#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : registry.py
# @Project   : ODPlatform
# @Function  : Config registry for runtime_config.

"""Single source of truth for config name -> config class."""

from od_platform.runtime_config.infer import YOLOInferConfig
from od_platform.runtime_config.train import YOLOTrainConfig
from od_platform.runtime_config.val import YOLOValConfig


CONFIG_REGISTRY: dict[str, tuple[type, str]] = {
    "train": (YOLOTrainConfig, "YOLO 训练配置"),
    "val": (YOLOValConfig, "YOLO 验证配置"),
    "infer": (YOLOInferConfig, "YOLO 推理配置"),
}
