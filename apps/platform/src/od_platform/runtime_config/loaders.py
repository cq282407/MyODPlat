#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : loaders.py
# @Author    : 雨霖同学 (ODPlatform team)
# @Project   : ODPlatform
# @Function  : runtime_config 配置加载器.
from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml

logger = logging.getLogger(__name__)


def _drop_none(d: Mapping[str, Any]) -> Dict[str, Any]:
    """Filter None while preserving explicit falsey values."""

    return {k: v for k, v in d.items() if v is not None}


class YAMLLoader:
    """Load a YAML config file into a dict."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else None

    def load(self, filename: Union[str, Path]) -> Dict[str, Any]:
        filepath = self._resolve_path(filename)

        if not filepath.exists():
            raise FileNotFoundError(
                f"YAML 配置文件不存在: {filepath}\n\n"
                f"请先生成默认配置模板:\n"
                f"  odp-gen-config {filepath.stem}\n\n"
                f"生成后编辑该文件再重新运行."
            )

        try:
            content = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("UTF-8 解码失败, 尝试系统默认编码: %s", filepath)
            content = filepath.read_text()

        if not content.strip():
            logger.debug("YAML 文件为空: %s", filepath)
            return {}

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"YAML 格式错误: {filepath}\n原始错误: {exc}\n"
                f"提示: 检查缩进、引号匹配、冒号后是否有空格"
            ) from exc

        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML 顶层必须是字典, 当前是 {type(data).__name__}: {filepath}")
        return data

    def _resolve_path(self, filename: Union[str, Path]) -> Path:
        p = Path(filename)
        if p.is_absolute():
            return p
        if p.exists():
            return p.resolve()
        if self.config_dir is not None:
            return self.config_dir / p
        return p


class CLILoader:
    """Extract config-like fields from argparse Namespace or dict-like payloads."""

    CONTROL_FIELDS = {"config", "func", "command"}

    def load(self, args: Namespace | Mapping[str, Any], *, exclude: Optional[set[str]] = None) -> Dict[str, Any]:
        exclude = (exclude or set()) | self.CONTROL_FIELDS
        if isinstance(args, Mapping):
            payload = dict(args)
        else:
            payload = vars(args)
        raw = {k: v for k, v in payload.items() if k not in exclude}
        return _drop_none(raw)
