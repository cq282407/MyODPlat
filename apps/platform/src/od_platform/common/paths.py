#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : paths.py
# @Time      : 2026/6/29 13:39:33
# @Author    : 小陈同学
# @Project   : ODPlatform
# @Function  : 定义所有的路径变量信息，方便其他模块调用

from pathlib import Path

from typing import List, Tuple


# 找到Workspace根目录
WORKSPACE_MARKER: str = ".odp-workspace"


def _find_workspace_root(
        start: Path,
        markers: Tuple[str, ...] = (WORKSPACE_MARKER,)
) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent

    for parent in [current, *current.parents]:
        for marker in markers:
            if (parent / marker).exists():
                return parent

    raise FileNotFoundError(f"找不到workspace marker文件 ({markers})"
                            f"请确认仓库根目录已存在 {WORKSPACE_MARKER} 文件")


# 计算ROOT_DIR位置
ROOT_DIR: Path = _find_workspace_root(Path(__file__))


# 端的根目录APP_DIR
APP_DIR: Path = ROOT_DIR / "apps" / "platform"


# 共享资产
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
RUNS_DIR: Path = ROOT_DIR / "runs"
VALIDATION_RUNS_DIR: Path = RUNS_DIR / "data_validation"


# 模型的子目录
PRETRAINED_MODELS_DIR: Path = MODELS_DIR / "pretrained"  # 存放哪些预训练模型
TRAINED_MODELS_DIR: Path = MODELS_DIR / "trained"  # 训练好的哪些归档模型


# 数据集的子目录
RAW_DATA_DIR: Path = DATA_DIR / "raw"  # 这个存放用户原始的数据
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"  # 派生的数据集,含冻结后的train/val/test


# 端私有资产
CONFIGS_DIR: Path = APP_DIR / "configs"
DATASET_CONFIGS_DIR: Path = CONFIGS_DIR / "datasets"
RUNTIME_CONFIGS_DIR: Path = CONFIGS_DIR / "runtime"
LOGGING_DIR: Path = APP_DIR / "logging"
UNIT_TEST_DIR: Path = APP_DIR / "tests"


# 顶层的文档目录
DOCS_DIR: Path = ROOT_DIR / "docs"


# 工程基础设置目录共享的
SCRIPTS_DIR: Path = ROOT_DIR / "scripts"


# 兼容 PRD 中出现过的 checkpoints 命名，实际初始化沿用老师演示的 trained 目录
CHECKPOINTS_DIR: Path = TRAINED_MODELS_DIR


# 元工具数据目录 / 工具自身的一些日志
META_DIR: Path = ROOT_DIR / ".odp-meta"
META_LOGGING_DIR: Path = META_DIR / "logging"


# 对外暴露的要初始化的目录列表
def get_dirs_to_initialize() -> List[Path]:
    return [
        DATA_DIR,
        MODELS_DIR,
        RUNS_DIR,
        PRETRAINED_MODELS_DIR,
        TRAINED_MODELS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        CONFIGS_DIR,
        DATASET_CONFIGS_DIR,
        LOGGING_DIR,
        UNIT_TEST_DIR,
        DOCS_DIR,
        SCRIPTS_DIR,
        META_LOGGING_DIR,
    ]


def get_dirs_to_reset() -> List[Path]:
    """返回 reset_project 可以安全清理的目录列表。"""
    return [
        PROCESSED_DATA_DIR,
        RUNS_DIR,
        LOGGING_DIR,
        TRAINED_MODELS_DIR,
    ]


def validation_run_dir(run_id: str) -> Path:
    """Return the output directory for one validation run."""

    return VALIDATION_RUNS_DIR / run_id


def runtime_config_path(name: str) -> Path:
    """返回某个运行配置文件的路径: <CONFIGS_DIR>/runtime/<name>.yaml

    Args:
        name: 配置名 (如 "train" / "val" / "infer"), 不带 .yaml 后缀

    Returns:
        Path 对象 (尚未创建, 调用方自己负责 mkdir)

    用法:
        train_yaml = runtime_config_path("train")
        # → <APP_DIR>/configs/runtime/train.yaml
    """
    return RUNTIME_CONFIGS_DIR / f"{name}.yaml"


# 绝对保护目录: reset 工具永远不能删除这些内容
PROTECTED_DIRS: tuple[Path, ...] = (
    ROOT_DIR,
    APP_DIR,
    SCRIPTS_DIR,
    DOCS_DIR,
    UNIT_TEST_DIR,
    ROOT_DIR / ".git",
    ROOT_DIR / WORKSPACE_MARKER,
    RAW_DATA_DIR,
    PRETRAINED_MODELS_DIR,
    CONFIGS_DIR,
    DATASET_CONFIGS_DIR,
    RUNTIME_CONFIGS_DIR,
    APP_DIR / "src",
    META_DIR,
    META_LOGGING_DIR,
)


def is_protected(path: Path) -> bool:
    """判断路径是否受保护, 防止 reset 工具误删关键内容。"""
    resolved_path = path.resolve(strict=False)
    for protected in PROTECTED_DIRS:
        resolved_protected = protected.resolve(strict=False)
        if resolved_path == resolved_protected:
            return True
        if resolved_protected.is_relative_to(resolved_path):
            return True

    protected_subtrees = (
        ROOT_DIR / ".git",
        RAW_DATA_DIR,
        PRETRAINED_MODELS_DIR,
        CONFIGS_DIR,
        UNIT_TEST_DIR,
        APP_DIR / "src",
        META_DIR,
        META_LOGGING_DIR,
    )
    for protected in protected_subtrees:
        if resolved_path.is_relative_to(protected.resolve(strict=False)):
            return True
    return False


if __name__ == "__main__":
    print(f"ROOT DIR (workspace) = {ROOT_DIR}")
    print(f"APP DIR = {APP_DIR}")
    print(f"DATA DIR = {DATA_DIR}")
    print(f"MODELS DIR = {MODELS_DIR}")
    print(f"RUNS DIR = {RUNS_DIR}")
    print(f"PRETRAINED MODELS DIR = {PRETRAINED_MODELS_DIR}")
    print(f"TRAINED MODELS DIR = {TRAINED_MODELS_DIR}")
    print(f"RAW DATA DIR = {RAW_DATA_DIR}")
    print(f"PROCESSED DATA DIR = {PROCESSED_DATA_DIR}")
    print(f"CONFIGS DIR = {CONFIGS_DIR}")
    print(f"LOGGING DIR = {LOGGING_DIR}")
    print(f"UNIT TEST DIR = {UNIT_TEST_DIR}")
    for d in get_dirs_to_initialize():
        print(f"将要初始化的目录有: {d.relative_to(ROOT_DIR)}")
    for d in get_dirs_to_reset():
        print(f"可重置清理的目录有: {d.relative_to(ROOT_DIR)}")
