#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : validate_data.py
# @Project   : ODPlatform
# @Function  : Thin CLI for dataset validation.

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR
from od_platform.common.refs import resolve_yaml
from od_platform.data_validation.registry import ValidationOptions
from od_platform.data_validation.service import validate_dataset

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_WARNING = 1
EXIT_ERROR = 2
EXIT_TOOL_ERROR = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="odp-validate",
        description="Validate a YOLO dataset yaml produced by odp-transform.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dataset", help="Dataset config name under apps/platform/configs/datasets")
    target.add_argument("--yaml", type=Path, help="Dataset yaml path")
    parser.add_argument("--run-id", default=None, help="Optional fixed run id for reproducible output paths")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional report output directory")
    parser.add_argument("--operator", default=None, help="Operator name written to the report")
    parser.add_argument("--operator-role", default=None, help="Operator role/team written to audit artifacts")
    parser.add_argument("--device-tag", default=None, help="Human-readable device tag, e.g. lab-laptop-01")
    parser.add_argument("--operation", default="data_validation", help="Operation name written to audit artifacts")
    parser.add_argument("--notes", default=None, help="Free-form validation note written to reports")
    parser.add_argument(
        "--check-phash",
        action="store_true",
        help="Enable heavy perceptual-hash near-duplicate image check.",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=6,
        help="Maximum pHash Hamming distance treated as near-duplicate.",
    )
    args = parser.parse_args(argv)

    get_logger(base_path=LOGGING_DIR, log_type="validate")

    yaml_path = _resolve_yaml_path(args.dataset, args.yaml)
    options = ValidationOptions(
        run_id=args.run_id,
        output_dir=args.output_dir,
        operator=args.operator,
        operator_role=args.operator_role,
        device_tag=args.device_tag,
        operation=args.operation,
        notes=args.notes,
        check_phash=args.check_phash,
        phash_threshold=args.phash_threshold,
    )

    try:
        report = validate_dataset(yaml_path=yaml_path, options=options)
    except Exception as exc:
        logger.exception("Validation tool failed: %s", exc)
        return EXIT_TOOL_ERROR
    return report.exit_code


def _resolve_yaml_path(dataset: str | None, yaml_path: Path | None) -> Path:
    if yaml_path is not None:
        return yaml_path
    assert dataset is not None
    return resolve_yaml(dataset)


if __name__ == "__main__":
    sys.exit(main())
