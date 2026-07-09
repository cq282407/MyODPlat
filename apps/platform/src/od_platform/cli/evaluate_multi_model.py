#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : evaluate_multi_model.py
# @Project   : ODPlatform
# @Function  : odp-val-multi CLI.
from __future__ import annotations

import argparse
import logging
import sys

from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR
from od_platform.evaluation.multi_service import DEFAULT_SORT_BY, MultiValService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-val-multi",
        description="Evaluate multiple YOLO models through ODPlatform MultiValService.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--yaml", type=str, default=None, help="Runtime val yaml; defaults to configs/runtime/val.yaml")
    parser.add_argument("--models", type=str, nargs="+", required=True, help="One or more trained model filenames or paths")
    parser.add_argument("--data", type=str, help="Dataset yaml path/name")
    parser.add_argument("--batch", type=int, help="Validation batch size")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--device", type=str, help="Validation device, e.g. 0/cpu/0,1")
    parser.add_argument("--workers", type=int, help="Dataloader workers")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--conf", type=float, help="Confidence threshold")
    parser.add_argument("--iou", type=float, help="NMS IoU threshold")
    parser.add_argument("--max-det", dest="max_det", type=int, help="Max detections per image")
    parser.add_argument("--run-name", type=str, default=None, help="Comparison output directory name under runs/<task>_val_compare/")
    parser.add_argument("--sort-by", type=str, default=DEFAULT_SORT_BY, help="Metric key used for ranking successful models")
    parser.add_argument("--academic-plots", action="store_true", help="Apply matplotlib academic style for this process")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.academic_plots:
        from od_platform.common.plot_style import apply_academic_style

        apply_academic_style()

    get_logger(base_path=LOGGING_DIR, log_type="val_multi", log_level=getattr(logging, args.log_level))
    log = logging.getLogger(__name__)

    non_config = {"yaml", "models", "run_name", "sort_by", "academic_plots", "log_level"}
    cli_args = {key: value for key, value in vars(args).items() if value is not None and key not in non_config}

    try:
        result = MultiValService().evaluate_many(
            models=args.models,
            yaml_path=args.yaml,
            cli_args=cli_args,
            run_name=args.run_name,
            sort_by=args.sort_by,
        )
    except KeyboardInterrupt:
        log.warning("用户中断 (Ctrl+C)")
        return 130

    if result.success:
        failure_count = sum(1 for item in result.results if not item.success)
        log.info("多模型评估完成: output=%s", result.output_dir)
        if result.best_model is not None:
            log.info("最佳模型: %s (score=%s)", result.best_model, f"{result.best_score:.4f}" if result.best_score is not None else "n/a")
        if result.summary_csv_path:
            log.info("summary.csv: %s", result.summary_csv_path)
        if result.summary_json_path:
            log.info("summary.json: %s", result.summary_json_path)
        if result.summary_md_path:
            log.info("summary.md: %s", result.summary_md_path)
        if failure_count:
            log.warning("其中 %s 个模型评估失败，已记录到 summary 中。", failure_count)
        return 0

    log.error("多模型评估失败: %s", result.error or "all models failed")
    if result.output_dir is not None:
        log.info("部分输出目录: %s", result.output_dir)
    return 1


if __name__ == "__main__":
    sys.exit(main())
