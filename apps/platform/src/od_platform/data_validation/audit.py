#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : audit.py
# @Project   : ODPlatform
# @Function  : Build audit metadata for one data validation run.

from __future__ import annotations

import getpass
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any

from od_platform.data_validation.registry import ValidationOptions


def build_audit_context(
    *,
    yaml_path: Path,
    run_id: str,
    started_at_iso: str,
    options: ValidationOptions,
) -> dict[str, Any]:
    """Return a JSON-friendly audit record for "who did what, where and how"."""

    return {
        "run_id": run_id,
        "started_at": started_at_iso,
        "operation": options.operation,
        "operator": options.operator,
        "operator_role": options.operator_role,
        "device_tag": options.device_tag,
        "notes": options.notes,
        "dataset_yaml": str(Path(yaml_path).resolve()),
        "process": {
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "argv": list(sys.argv),
        },
        "user": {
            "login": _safe_getuser(),
            "home": str(Path.home()),
        },
        "device": _device_info(),
    }


def _safe_getuser() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore[import-not-found]

        memory = psutil.virtual_memory()
        info["memory"] = {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "used_percent": memory.percent,
        }
    except Exception:
        info["memory"] = {"available": False}

    try:
        import torch  # type: ignore[import-not-found]

        cuda_available = bool(torch.cuda.is_available())
        gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
        info["torch"] = {
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
            "gpus": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(gpu_count)
            ],
        }
    except Exception:
        info["torch"] = {"available": False}
    return info
