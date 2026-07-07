#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : transform_data.py
# @Project   : ODPlatform
# @Function  : Thin CLI for dataset conversion.

from __future__ import annotations

import argparse
import logging
import sys

from od_platform.common.constants import AnnotationFormat, SplitStrategy, Task
from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR
from od_platform.data_pipeline.convert.registry import list_capabilities
from od_platform.data_pipeline.orchestrator import DatasetPipeline
from od_platform.data_pipeline.render import render_transform_report

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_DATA_ERR = 1

DEFAULT_DATASET = "rsod"
DEFAULT_FORMAT = AnnotationFormat.PASCAL_VOC
DEFAULT_SPLIT_STRATEGY = SplitStrategy.RANDOM
DEFAULT_SEED = 42


def main(argv: list[str] | None = None) -> int:
    format_choices = _format_choices()
    parser = argparse.ArgumentParser(
        prog="odp-transform",
        description="Convert data/raw/<dataset> annotations into YOLO txt labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset folder name under data/raw")
    parser.add_argument(
        "--format",
        default=DEFAULT_FORMAT,
        choices=format_choices,
        help="Source annotation format",
    )
    parser.add_argument("--task", default=Task.DETECT, choices=Task.all())
    parser.add_argument("--split-strategy", default=DEFAULT_SPLIT_STRATEGY, choices=SplitStrategy.all())
    parser.add_argument("--classes", nargs="+", default=None, help="Class whitelist; auto-discover when omitted")
    parser.add_argument("--train-rate", type=float, default=0.8)
    parser.add_argument("--val-rate", type=float, default=0.1)
    parser.add_argument("--test-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-name", default=None, help="Output folder under data/processed")
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Include images without labels as background samples.",
    )
    parser.add_argument("--list-capabilities", action="store_true", help="Show registered converters and exit")
    raw_argv = sys.argv[1:] if argv is None else argv
    using_classroom_defaults = len(raw_argv) == 0
    args = parser.parse_args(raw_argv)

    get_logger(base_path=LOGGING_DIR, log_type="transform")

    if args.list_capabilities:
        for fmt, tasks in list_capabilities().items():
            logger.info("%s: %s", fmt, ", ".join(tasks))
        return EXIT_OK

    if using_classroom_defaults:
        logger.info(
            "No arguments supplied; using classroom defaults: "
            "--dataset %s --format %s --split-strategy %s --seed %s",
            args.dataset,
            args.format,
            args.split_strategy,
            args.seed,
        )

    pipeline = DatasetPipeline(
        dataset=args.dataset,
        annotation_format=args.format,
        task=args.task,
        train_rate=args.train_rate,
        classes=args.classes,
        random_state=args.seed,
        split_strategy=args.split_strategy,
        output_name=args.output_name,
        val_rate=args.val_rate,
        test_rate=args.test_rate,
        include_unlabeled=args.include_unlabeled,
    )

    try:
        result = pipeline.run()
    except (FileNotFoundError, ValueError) as e:
        logger.error("Processing failed: %s", e)
        return EXIT_DATA_ERR

    render_transform_report(
        result,
        logger,
        task=args.task,
        source_format=args.format,
        train_rate=args.train_rate,
        val_rate=args.val_rate,
        test_rate=args.test_rate,
        seed=args.seed,
    )
    return EXIT_OK


def _format_choices() -> tuple[str, ...]:
    formats = list(list_capabilities())
    if AnnotationFormat.PASCAL_VOC in formats and AnnotationFormat.VOC not in formats:
        formats.insert(formats.index(AnnotationFormat.PASCAL_VOC) + 1, AnnotationFormat.VOC)
    return tuple(formats)


if __name__ == "__main__":
    sys.exit(main())
