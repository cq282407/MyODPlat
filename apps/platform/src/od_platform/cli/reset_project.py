#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : reset_project.py
# @Author    : ODPlatform team
# @Project   : ODPlatform
# @Function  : 项目重置工具(v3 - dry-run + 确认 + 进度)
"""ODPlatform 项目重置工具: 安全地撤销 init_project 创建的运行时产物。"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import (
    META_LOGGING_DIR,
    PRETRAINED_MODELS_DIR,
    RAW_DATA_DIR,
    ROOT_DIR,
    get_dirs_to_reset,
    is_protected,
)
from od_platform.common.string_utils import format_table_row, format_table_separator


logger = get_logger(
    base_path=META_LOGGING_DIR,
    log_type="reset_project",
    temp_log=False,
)

CONFIRM_KEYWORD = "RESET"
LINE_WIDTH = 70


def _format_size(bytes_size: int) -> str:
    """使用 1024 进制格式化文件大小。"""
    if bytes_size >= 1024 ** 3:
        return f"{bytes_size / (1024 ** 3):.2f} GiB"
    if bytes_size >= 1024 ** 2:
        return f"{bytes_size / (1024 ** 2):.2f} MiB"
    if bytes_size >= 1024:
        return f"{bytes_size / 1024:.2f} KiB"
    return f"{bytes_size} B"


def _on_rm_error(func, path, exc_info) -> None:
    """处理 Windows 只读文件导致的 rmtree 删除失败。"""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        raise


def _audit_context() -> dict:
    """收集审计上下文: user / git rev / argv / cwd / pid。"""
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_rev = "(not a git repo)"

    return {
        "user": getpass.getuser(),
        "pid": os.getpid(),
        "git_rev": git_rev,
        "argv": sys.argv,
        "cwd": os.getcwd(),
    }


def _scan_targets() -> tuple[list[tuple[Path, int, int]], list[Path]]:
    """扫描所有目标, 返回 (可删除目录, 跳过目录)。"""
    deletable: list[tuple[Path, int, int]] = []
    skipped: list[Path] = []

    for d in get_dirs_to_reset():
        if is_protected(d):
            logger.warning(f"拒绝处理受保护目录(配置可能有误): {d}")
            skipped.append(d)
            continue
        if not d.exists():
            skipped.append(d)
            continue

        file_count = 0
        total_size = 0
        try:
            for f in d.rglob("*"):
                if f.is_file():
                    file_count += 1
                    try:
                        total_size += f.stat().st_size
                    except OSError:
                        pass
        except OSError as e:
            logger.warning(f"扫描 {d} 时出错: {e}")

        deletable.append((d, file_count, total_size))

    return deletable, skipped


def _print_plan(
    deletable: list[tuple[Path, int, int]],
    skipped: list[Path],
    will_actually_delete: bool,
) -> None:
    if will_actually_delete:
        logger.warning("即将删除以下目录".center(LINE_WIDTH, "="))
    else:
        logger.info("[DRY-RUN] 计划如下(未实际删除)".center(LINE_WIDTH, "="))

    if not deletable:
        logger.info("(没有可删除的目录 - 项目已经是干净状态)")
        return

    widths = [40, 12, 14]
    aligns = ["left", "right", "right"]
    logger.info(format_table_row(["目录", "文件数", "大小"], widths, aligns))
    logger.info(format_table_separator(widths))

    total_files = 0
    total_bytes = 0
    for path, count, size in deletable:
        rel = path.relative_to(ROOT_DIR)
        logger.info(format_table_row(
            [str(rel), str(count), _format_size(size)], widths, aligns,
        ))
        total_files += count
        total_bytes += size

    logger.info(format_table_separator(widths))
    logger.info(format_table_row(
        ["【合计】", str(total_files), _format_size(total_bytes)], widths, aligns,
    ))

    if skipped:
        logger.info("")
        logger.info("以下目录不存在或被保护, 已跳过:")
        for path in skipped:
            try:
                logger.info(f"  - {path.relative_to(ROOT_DIR)}")
            except ValueError:
                logger.info(f"  - {path}")

    logger.info("")
    logger.info("以下重要目录不会被动:")
    logger.info(f"  - 原始数据: {RAW_DATA_DIR.relative_to(ROOT_DIR)}/")
    logger.info(f"  - 预训练权重: {PRETRAINED_MODELS_DIR.relative_to(ROOT_DIR)}/")
    logger.info("  - 所有代码、文档、元工具日志")


def _confirm(deletable_count: int) -> bool:
    """交互式确认。只有精确输入 RESET 才执行删除。"""
    print()
    print("=" * LINE_WIDTH)
    print(f"你正要删除 {deletable_count} 个目录的内容。这个操作不可撤销。")
    print(f"如果确认, 请精确输入大写的 '{CONFIRM_KEYWORD}'(其他任何输入都会取消):")
    print("=" * LINE_WIDTH)
    try:
        user_input = input("> ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    return user_input == CONFIRM_KEYWORD


def _delete_one(path: Path, idx: int, total: int, file_count: int, size: int) -> str | None:
    """删除单个目录, 带进度提示。返回 None 表示成功, 字符串表示失败原因。"""
    if is_protected(path):
        logger.error(f"[{idx}/{total}] 删除前检查失败, 跳过: {path}")
        return "受保护目录"

    rel = path.relative_to(ROOT_DIR)
    size_str = _format_size(size)

    if size > 1024 ** 3:
        logger.warning(
            f"[{idx}/{total}] 正在删除 {rel} ({size_str}, {file_count} 个文件)"
            f" - 这可能需要一会..."
        )
    else:
        logger.info(f"[{idx}/{total}] 删除 {rel} ({size_str}, {file_count} 个文件)")

    try:
        shutil.rmtree(path, onerror=_on_rm_error)
        logger.info(f"[{idx}/{total}] 已删除: {rel}")
        return None
    except OSError as e:
        logger.error(f"[{idx}/{total}] 删除失败 {rel}: {e}")
        return str(e)


def _execute_delete(deletable: list[tuple[Path, int, int]]) -> int:
    """执行删除并返回退出码: 0=全成功, 1=部分失败, 2=全部失败。"""
    total = len(deletable)
    success: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for idx, (path, file_count, size) in enumerate(deletable, 1):
        reason = _delete_one(path, idx, total, file_count, size)
        if reason is None:
            success.append(path)
        else:
            failed.append((path, reason))

    logger.info("=" * LINE_WIDTH)
    if failed:
        logger.warning(f"完成: 成功 {len(success)} 个, 失败 {len(failed)} 个")
        for p, reason in failed:
            logger.warning(f"  - {p.relative_to(ROOT_DIR)}: {reason}")
    else:
        logger.info(f"完成: 成功 {len(success)} 个, 失败 0 个")

    if not failed:
        return 0
    if success:
        return 1
    return 2


def reset_project(yes: bool = False, force: bool = False, dry_run: bool = False) -> int:
    """重置项目运行时产物。默认 dry-run, 不会实际删除。"""
    logger.info("项目重置工具".center(LINE_WIDTH, "="))
    logger.info(f"项目根目录: {ROOT_DIR}")

    ctx = _audit_context()
    logger.info(f"审计: user={ctx['user']}, pid={ctx['pid']}, git={ctx['git_rev']}")
    logger.info(f"审计: cwd={ctx['cwd']}")
    logger.info(f"审计: argv={' '.join(ctx['argv'])}")

    if dry_run and yes:
        logger.warning("同时给了 --dry-run 和 --yes, 以 --dry-run 为准(只打印不删除)")
        yes = False

    deletable, skipped = _scan_targets()
    _print_plan(deletable, skipped, will_actually_delete=yes)

    if not deletable:
        return 0

    if not yes:
        logger.info("")
        if dry_run:
            logger.info("这是显式的 --dry-run。要真正执行删除, 请加 --yes:")
        else:
            logger.info("这是 dry-run(默认行为)。要真正执行删除, 请加 --yes:")
        logger.info("   python scripts/reset_project.py --yes")
        return 0

    if not force and not _confirm(len(deletable)):
        logger.warning("用户取消, 未执行删除")
        return 0

    logger.info("")
    logger.info("开始删除...".center(LINE_WIDTH, "="))
    return _execute_delete(deletable)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="重置 ODPlatform 项目 - 撤销 init_project 创建的运行时产物。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--yes", action="store_true", help="真正执行删除(默认是 dry-run)")
    parser.add_argument("--force", action="store_true", help="跳过交互式确认(仅当 --yes 时有效)")
    parser.add_argument("--dry-run", action="store_true", help="显式声明 dry-run")
    args = parser.parse_args()
    return reset_project(yes=args.yes, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
