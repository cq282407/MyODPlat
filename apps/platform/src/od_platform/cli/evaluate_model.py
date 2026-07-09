#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : evaluate_model.py
# @Project   : ODPlatform
# @Function  : odp-val CLI.
from __future__ import annotations

import argparse
import logging
import sys

from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR
from od_platform.evaluation import ValService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-val",
        description="Evaluate a YOLO model through ODPlatform D7 ValService.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--yaml", type=str, default=None, help="Runtime val yaml; defaults to configs/runtime/val.yaml")
    parser.add_argument("--model", type=str, help="Trained model filename or path")
    parser.add_argument("--data", type=str, help="Dataset yaml path/name")
    parser.add_argument("--batch", type=int, help="Validation batch size")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--device", type=str, help="Validation device, e.g. 0/cpu/0,1")
    parser.add_argument("--workers", type=int, help="Dataloader workers")
    parser.add_argument("--project", type=str, help="Ultralytics project/output root")
    parser.add_argument("--name", type=str, help="Ultralytics run name")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--conf", type=float, help="Confidence threshold")
    parser.add_argument("--iou", type=float, help="NMS IoU threshold")
    parser.add_argument("--max-det", dest="max_det", type=int, help="Max detections per image")
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

    get_logger(base_path=LOGGING_DIR, log_type="val", log_level=getattr(logging, args.log_level))
    log = logging.getLogger(__name__)

    non_config = {"yaml", "rename_log", "academic_plots", "log_level"}
    cli_args = {key: value for key, value in vars(args).items() if value is not None and key not in non_config}

    try:
        result = ValService().evaluate(
            yaml_path=args.yaml,
            cli_args=cli_args,
            rename_log=args.rename_log,
        )
    except KeyboardInterrupt:
        log.warning("用户中断 (Ctrl+C)")
        return 130

    if result.success:
        log.info("评估成功: output=%s", result.output_dir)
        return 0
    log.error("评估失败: %s", result.error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
