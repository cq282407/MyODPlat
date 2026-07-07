#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : train_model.py
# @Project   : ODPlatform
# @Function  : odp-train CLI.
from __future__ import annotations

import argparse
import logging
import sys

from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR
from od_platform.training import TrainService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-train",
        description="Train a YOLO model through ODPlatform D6 TrainService.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--yaml", type=str, default=None, help="Runtime train yaml; defaults to configs/runtime/train.yaml")
    parser.add_argument("--model", type=str, help="Model path or pretrained filename")
    parser.add_argument("--data", type=str, help="Dataset yaml path/name")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--batch", type=_batch_value, help="Batch size, -1, or GPU memory fraction")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--device", type=str, help="Training device, e.g. 0/cpu/0,1")
    parser.add_argument("--workers", type=int, help="Dataloader workers")
    parser.add_argument("--fraction", type=float, help="Dataset fraction for quick smoke runs")
    parser.add_argument("--project", type=str, help="Ultralytics project/output root")
    parser.add_argument("--name", type=str, help="Ultralytics run name")
    parser.add_argument("--no-pre-validate", dest="pre_validate", action="store_false", default=True)
    parser.add_argument("--no-archive", dest="archive", action="store_false", default=True)
    parser.add_argument("--no-rename-log", dest="rename_log", action="store_false", default=True)
    parser.add_argument("--academic-plots", action="store_true", help="Apply matplotlib academic style for this process")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.academic_plots:
        from od_platform.common.plot_style import apply_academic_style

        apply_academic_style()

    get_logger(base_path=LOGGING_DIR, log_type="train", log_level=getattr(logging, args.log_level))
    log = logging.getLogger(__name__)

    non_config = {"yaml", "pre_validate", "archive", "rename_log", "academic_plots", "log_level"}
    cli_args = {key: value for key, value in vars(args).items() if value is not None and key not in non_config}

    try:
        result = TrainService().train(
            yaml_path=args.yaml,
            cli_args=cli_args,
            pre_validate=args.pre_validate,
            archive=args.archive,
            rename_log=args.rename_log,
        )
    except KeyboardInterrupt:
        log.warning("用户中断 (Ctrl+C)")
        return 130

    if result.success:
        log.info("训练成功: output=%s time=%.2fs", result.output_dir, result.train_time or 0.0)
        return 0
    log.error("训练失败: %s", result.error)
    return 1


def _batch_value(value: str) -> int | float:
    if "." in value:
        return float(value)
    return int(value)


if __name__ == "__main__":
    sys.exit(main())
