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
    parser.add_argument("--config", "--yaml", dest="config", type=str, default="val.yaml", help="Runtime val yaml")
    parser.add_argument("--model", required=True, type=str, help="Trained weight path or filename under models/trained")
    parser.add_argument("--data", required=True, type=str, help="Dataset yaml path/name")
    parser.add_argument("--batch", type=_batch_value, help="Batch size, -1, or GPU memory fraction")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--device", type=str, help="Evaluation device, e.g. 0/cpu/0,1")
    parser.add_argument("--workers", type=int, help="Dataloader workers")
    parser.add_argument("--split", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--conf", type=float, help="Confidence threshold")
    parser.add_argument("--iou", type=float, help="NMS IoU threshold")
    parser.add_argument("--max-det", dest="max_det", type=int, help="Maximum detections per image")
    parser.add_argument("--project", type=str, help="Ultralytics project/output root")
    parser.add_argument("--name", type=str, help="Ultralytics run name")
    parser.add_argument("--half", dest="half", action="store_true", default=None)
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--plots", dest="plots", action="store_true", default=None)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--save-json", dest="save_json", action="store_true", default=None)
    parser.add_argument("--no-save-json", dest="save_json", action="store_false")
    parser.add_argument("--save-hybrid", dest="save_hybrid", action="store_true", default=None)
    parser.add_argument("--no-save-hybrid", dest="save_hybrid", action="store_false")
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

    non_config = {"config", "model", "data", "rename_log", "academic_plots", "log_level"}
    cli_overrides = {key: value for key, value in vars(args).items() if value is not None and key not in non_config}

    try:
        result = ValService().evaluate(
            config_path=args.config,
            model=args.model,
            data=args.data,
            cli_overrides=cli_overrides,
            rename_log=args.rename_log,
        )
    except KeyboardInterrupt:
        log.warning("User interrupted (Ctrl+C)")
        return 130

    if result.success:
        log.info("Evaluation success: output=%s time=%.2fs", result.save_dir, result.val_time or 0.0)
        return 0
    log.error("Evaluation failed: %s", result.error)
    return 1


def _batch_value(value: str) -> int | float:
    if "." in value:
        return float(value)
    return int(value)


if __name__ == "__main__":
    sys.exit(main())
