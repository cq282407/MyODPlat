#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : logging_utils.py
# @Project   : ODPlatform
# @Function  : Project logger setup with console and per-run file output.

import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from colorlog import ColoredFormatter
except ImportError:  # pragma: no cover - depends on optional local environment
    ColoredFormatter = None  # type: ignore[assignment]

ROOT_LOGGER_NAME = "od_platform"


def _safe_filename_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_-" else "_" for char in value)


def _build_log_file(
    base_path: Path,
    log_type: str,
    model_name: Optional[str],
    temp_log: bool,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:21]
    prefix = "temp" if temp_log else log_type.replace("_", "-")
    filename_parts = [prefix, timestamp]
    if model_name:
        filename_parts.append(_safe_filename_part(model_name))
    return base_path / log_type / ("_".join(filename_parts) + ".log")


def get_logger(
    base_path: Path,
    log_type: str = "general",
    model_name: Optional[str] = None,
    log_level: int = logging.INFO,
    temp_log: bool = False,
    encoding: str = "utf-8",
    logger_name: str = ROOT_LOGGER_NAME,
) -> logging.Logger:
    """Configure and return the project root logger.

    Args:
        base_path: Root directory for log files.
        log_type: Log category and subdirectory name.
        model_name: Optional model name appended to the log filename.
        log_level: Logging level for logger and handlers.
        temp_log: Whether to use `temp` as the filename prefix.
        encoding: File encoding for the file handler.
        logger_name: Logger name to configure.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    logger.propagate = False

    log_file = _build_log_file(base_path, log_type, model_name, temp_log)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_formatter = logging.Formatter(
        fmt=(
            "%(asctime)s - %(name)s - %(levelname)-8s - "
            "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding=encoding)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if ColoredFormatter is not None:
        console_formatter: logging.Formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s%(reset)s "
            "%(log_color)s[%(levelname)-8s]%(reset)s "
            "%(cyan)s%(filename)-25s%(reset)s:"
            "%(blue)s%(lineno)-4d%(reset)s "
            "%(log_color)s| %(message)s%(reset)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "white",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red,bg_white",
            },
        )
    else:
        console_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(filename)-25s:%(lineno)-4d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info("=" * 60)
    logger.info("Logging system initialized")
    logger.info("Runtime environment: %s %s", platform.system(), platform.release())
    logger.info("Log type: %s", log_type)
    logger.info("Log file: %s", log_file)
    logger.info("Log level: %s", logging.getLevelName(log_level))
    logger.info("Model name: %s", model_name or "N/A")
    logger.info("=" * 60)

    return logger
