#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_multi_evaluation.py
# @Project   : ODPlatform
# @Function  : D7 multi-model evaluation tests.
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from od_platform.cli.evaluate_multi_model import build_parser, main as multi_main
from od_platform.common.result import TrainMetrics
from od_platform.evaluation.multi_service import MultiValService
from od_platform.evaluation.service import ValResult

M = "od_platform.evaluation.multi_service"


class _StubValService:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.calls: list[dict[str, object]] = []
        self._scores = {
            "train3-best.pt": 0.62,
            "train4-best.pt": 0.51,
        }

    def evaluate(self, yaml_path=None, model=None, data=None, cli_args=None, *, rename_log=True):
        payload = dict(cli_args or {})
        self.calls.append(payload)
        model_ref = str(payload["model"])
        project = Path(str(payload["project"]))
        name = str(payload["name"])
        output_dir = project / name
        output_dir.mkdir(parents=True, exist_ok=True)

        if model_ref == "broken.pt":
            return ValResult(
                success=False,
                output_dir=None,
                error="boom",
                log_path=self.tmp_path / "val_multi.log",
            )

        audit_path = output_dir / "odp_audit.json"
        audit_path.write_text('{"kind": "val"}', encoding="utf-8")
        score = self._scores[model_ref]
        metrics = TrainMetrics(
            task="detect",
            save_dir=output_dir,
            timestamp="2026-07-09T15:00:00",
            speed_ms={"preprocess": 1.0, "inference": 2.0, "loss": 0.0, "postprocess": 1.0, "total": 4.0},
            overall={
                "fitness": score,
                "metrics/precision(B)": score + 0.10,
                "metrics/recall(B)": score + 0.05,
                "metrics/mAP50(B)": score + 0.20,
                "metrics/mAP50-95(B)": score,
            },
            class_map_50_95={"airplane": score},
        )
        return ValResult(
            success=True,
            output_dir=output_dir,
            metrics=metrics,
            audit_path=audit_path,
            log_path=self.tmp_path / "val_multi.log",
        )


def test_multi_val_service_keeps_going_and_writes_summaries(tmp_path: Path) -> None:
    cfg = MagicMock()
    cfg.task = "detect"
    stub = _StubValService(tmp_path)
    runs_dir = tmp_path / "runs"

    with patch(f"{M}.build_val_config", return_value=(cfg, MagicMock())), \
         patch(f"{M}.RUNS_DIR", runs_dir):
        result = MultiValService(val_service=stub).evaluate_many(
            models=["train3-best.pt", "broken.pt", "train4-best.pt"],
            yaml_path="val.yaml",
            cli_args={"data": "rsod.yaml", "batch": 8},
            run_name="compare_demo",
        )

    assert result.success is True
    assert result.output_dir == runs_dir / "detect_val_compare" / "compare_demo"
    assert result.summary_csv_path is not None and result.summary_csv_path.exists()
    assert result.summary_json_path is not None and result.summary_json_path.exists()
    assert result.summary_md_path is not None and result.summary_md_path.exists()
    assert result.best_model == "train3-best.pt"
    assert [item.model for item in result.results] == ["train3-best.pt", "train4-best.pt", "broken.pt"]
    assert [item.rank for item in result.results] == [1, 2, None]
    assert len(stub.calls) == 3
    assert all(call["project"] == str(result.output_dir) for call in stub.calls)
    assert [call["name"] for call in stub.calls] == ["001_train3-best", "002_broken", "003_train4-best"]

    payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert payload["best_model"] == "train3-best.pt"
    assert payload["success_count"] == 2
    assert payload["failure_count"] == 1


def test_multi_val_cli_parser_accepts_expected_arguments() -> None:
    args = build_parser().parse_args([
        "--models", "train3-best.pt", "train4-best.pt",
        "--data", "rsod.yaml",
        "--run-name", "compare_demo",
    ])

    assert args.models == ["train3-best.pt", "train4-best.pt"]
    assert args.data == "rsod.yaml"
    assert args.run_name == "compare_demo"


def test_multi_val_cli_returns_success_with_mocked_service() -> None:
    fake_result = MagicMock(
        success=True,
        output_dir=Path("runs/detect_val_compare/compare_demo"),
        results=[],
        best_model="train3-best.pt",
        best_score=0.62,
        summary_csv_path=Path("summary.csv"),
        summary_json_path=Path("summary.json"),
        summary_md_path=Path("summary.md"),
        error=None,
    )

    with patch("od_platform.cli.evaluate_multi_model.get_logger"), \
         patch("od_platform.cli.evaluate_multi_model.MultiValService") as service_cls:
        service_cls.return_value.evaluate_many.return_value = fake_result
        exit_code = multi_main(["--models", "train3-best.pt", "train4-best.pt", "--data", "rsod.yaml"])

    assert exit_code == 0
    service_cls.return_value.evaluate_many.assert_called_once()


def test_multi_val_cli_returns_130_on_keyboard_interrupt() -> None:
    with patch("od_platform.cli.evaluate_multi_model.get_logger"), \
         patch("od_platform.cli.evaluate_multi_model.MultiValService") as service_cls:
        service_cls.return_value.evaluate_many.side_effect = KeyboardInterrupt
        exit_code = multi_main(["--models", "train3-best.pt", "train4-best.pt", "--data", "rsod.yaml"])

    assert exit_code == 130
