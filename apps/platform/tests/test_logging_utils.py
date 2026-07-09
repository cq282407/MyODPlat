from pathlib import Path

from od_platform.common.logging_utils import _build_log_file


def test_temp_log_filename_includes_log_type_and_run_id(tmp_path: Path) -> None:
    log_file = _build_log_file(
        tmp_path,
        log_type="train_model",
        model_name="yolo/11n.pt",
        temp_log=True,
        run_id="smoke run",
    )

    assert log_file.parent == tmp_path / "train_model"
    assert log_file.name.startswith("temp-train-model_smoke_run_")
    assert log_file.name.endswith("_yolo_11n_pt.log")
