#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : generator.py
# @Project   : ODPlatform
# @Function  : Reflect runtime config models into self-documenting YAML templates.

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from od_platform.common.paths import runtime_config_path
from od_platform.runtime_config.registry import CONFIG_REGISTRY

logger = logging.getLogger(__name__)


class ConfigGenerator:
    """Generate YAML templates from Pydantic field metadata."""

    def generate(
        self,
        config_class,
        output_path,
        *,
        overwrite: bool = False,
        backup: bool = True,
        title: str | None = None,
    ) -> bool:
        output_path = Path(output_path)

        if output_path.exists() and not overwrite:
            logger.info("配置文件已存在，跳过生成: %s", output_path)
            return False

        if output_path.exists() and backup:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = output_path.with_name(f"{output_path.name}.bak.{stamp}")
            shutil.copy2(output_path, backup_path)
            logger.warning("覆盖前已备份原配置: %s", backup_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self._generate_yaml(config_class, title), encoding="utf-8")
        logger.info("配置文件已生成: %s", output_path)
        return True

    def _generate_yaml(self, config_class, title: str | None = None) -> str:
        try:
            config = config_class()
        except Exception:
            config = config_class.model_construct()

        lines: list[str] = []
        display_title = title or config_class.__name__
        lines.append(f"#{'=' * 78}")
        lines.append(f"# {display_title}")
        lines.append(f"# 自动生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("# 编辑后保存；重新生成(--overwrite)会覆盖并备份原文件。")
        lines.append(f"#{'=' * 78}")
        lines.append("")

        groups = config.get_field_groups()
        for group_name, field_names in groups.items():
            lines.append(f"#{'-' * 78}")
            lines.append(f"# {group_name}")
            lines.append(f"#{'-' * 78}")
            lines.append("")

            for field_name in field_names:
                meta = config.get_field_metadata(field_name)
                lines.append(f"# {meta['yaml_comment']}")
                examples = meta.get("examples", [])
                if examples:
                    lines.append(
                        f"# 示例: {', '.join(self._format_value(item) for item in examples)}"
                    )
                tips = meta.get("tips", [])
                if tips:
                    lines.append("# 提示:")
                    lines.extend(f"#   - {tip}" for tip in tips)
                default_value = meta.get("default")
                lines.append(f"{field_name}: {self._format_value(default_value)}")
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_value(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            if any(char in value for char in [":", "#", "[", "]", "{", "}"]):
                return f'"{value}"'
            return value
        if isinstance(value, (list, tuple)):
            if not value:
                return "[]"
            return f"[{', '.join(str(item) for item in value)}]"
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="odp-gen-config",
        description="生成 YOLO 运行配置 YAML 模板",
    )
    parser.add_argument("name", choices=list(CONFIG_REGISTRY), help="配置名 (train/val/infer)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出路径")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有文件(默认不覆盖)")
    parser.add_argument("--no-backup", action="store_true", help="覆盖时不备份(默认备份)")
    args = parser.parse_args()

    config_class, title = CONFIG_REGISTRY[args.name]
    output_path = args.output or runtime_config_path(args.name)
    generated = ConfigGenerator().generate(
        config_class,
        output_path,
        overwrite=args.overwrite,
        backup=not args.no_backup,
        title=title,
    )
    if generated:
        print(f"[OK] 已生成: {output_path}")
    else:
        print(
            "[SKIP] 文件已存在，未覆盖。如需重新生成请加 --overwrite"
            f"(默认会自动备份)。\n  路径: {output_path}"
        )


if __name__ == "__main__":
    main()
