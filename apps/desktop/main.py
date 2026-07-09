#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""ODPlatform Desktop demo (PySide6)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1] / "platform" / "src"))

from infer_worker import InferWorker
from qt_sink import QtSignalSink

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("desktop")


class MainWindow(QMainWindow):
    request_start = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ODPlatform Desktop Demo")
        self.resize(1280, 800)
        self._build_ui()
        self._build_worker()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("Model:"))
        self.le_model = QLineEdit("yolo11n.pt")
        row_model.addWidget(self.le_model, 1)
        btn_browse_model = QPushButton("Browse...")
        btn_browse_model.clicked.connect(self._browse_model)
        row_model.addWidget(btn_browse_model)
        layout.addLayout(row_model)

        row_source = QHBoxLayout()
        row_source.addWidget(QLabel("Source:"))
        self.le_source = QLineEdit("0")
        self.le_source.setPlaceholderText("摄像头号 / 视频文件 / 图片目录")
        row_source.addWidget(self.le_source, 1)
        btn_browse_source = QPushButton("Browse...")
        btn_browse_source.clicked.connect(self._browse_source)
        row_source.addWidget(btn_browse_source)
        layout.addLayout(row_source)

        row_buttons = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        row_buttons.addWidget(self.btn_start)
        row_buttons.addWidget(self.btn_stop)
        layout.addLayout(row_buttons)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.video_label = QLabel("[ 点 Start 开始推理 ]")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #222; color: #888; font-size: 16px;")
        self.video_label.setMinimumHeight(480)
        layout.addWidget(self.video_label, 1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪")

    def _build_worker(self) -> None:
        self.worker_thread = QThread(self)
        self.worker = InferWorker()
        self.worker.moveToThread(self.worker_thread)

        self.sink = QtSignalSink()
        self.sink.frame_ready.connect(self._on_frame, Qt.QueuedConnection)
        self.worker.set_sink(self.sink)

        self.request_start.connect(self.worker.start_infer)
        self.worker.progress_changed.connect(self._on_progress)
        self.worker.inference_finished.connect(self._on_finished)
        self.worker.inference_failed.connect(self._on_failed)

        self.worker_thread.start()
        logger.info("worker thread 启动完成")

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Model (*.pt *.engine);;所有文件 (*)")
        if path:
            self.le_model.setText(path)

    def _browse_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频/图片文件",
            "",
            "视频/图片 (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png);;所有文件 (*)",
        )
        if path:
            self.le_source.setText(path)

    def _on_start(self) -> None:
        model = self.le_model.text().strip()
        source = self.le_source.text().strip()
        if not model or not source:
            QMessageBox.warning(self, "缺少参数", "请填写模型和源。")
            return
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setVisible(True)
        self.statusBar().showMessage(f"启动推理: model={model}, source={source}")
        self.request_start.emit(model, source)

    def _on_stop(self) -> None:
        self.statusBar().showMessage("Stop 按下, 等待 pipeline 响应 (< 2s)...")
        self.worker.cancel()
        self.btn_stop.setEnabled(False)

    def _on_frame(self, annotated) -> None:
        if not isinstance(annotated, np.ndarray):
            return
        if annotated.ndim != 3 or annotated.shape[2] != 3:
            return
        try:
            height, width, _ = annotated.shape
            rgb = annotated[..., ::-1]
            if not rgb.flags["C_CONTIGUOUS"]:
                rgb = np.ascontiguousarray(rgb)
            qimg = QImage(rgb.data, width, height, width * 3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(pix)
        except Exception as exc:
            logger.exception("_on_frame 显示失败: %s", exc)

    def _on_progress(self, frame_idx: int, total: int, fps_loop: float) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(frame_idx)
        self.statusBar().showMessage(f"帧 {frame_idx}/{total if total > 0 else '?'} | loop {fps_loop:.1f} FPS")

    def _on_finished(self, result) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setVisible(False)
        if result.success:
            stats = result.stats or {}
            self.statusBar().showMessage(
                f"完成 {stats.get('frames', '?')} 帧 | 检测 {stats.get('detections', '?')} 个 | "
                f"loop {stats.get('fps', {}).get('loop', '?')} FPS"
            )
        else:
            self.statusBar().showMessage(f"失败: {result.error}")
            QMessageBox.critical(self, "推理失败", result.error or "未知错误")

    def _on_failed(self, error: str) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setVisible(False)
        self.statusBar().showMessage(f"失败: {error}")
        QMessageBox.critical(self, "推理失败", error)

    def closeEvent(self, event) -> None:
        try:
            self.worker.cancel()
        except Exception:
            pass
        self.worker_thread.quit()
        if not self.worker_thread.wait(3000):
            logger.warning("worker thread 3s 没退, 强终止")
            self.worker_thread.terminate()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
