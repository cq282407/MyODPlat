#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : reset_project.py
# @Author    : 小陈同学
# @Project   : ODPlatform
# @Function  : reset_project 开发期入口

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLATFORM_SRC = REPO_ROOT / "apps" / "platform" / "src"

sys.path.insert(0, str(PLATFORM_SRC))

from od_platform.cli.reset_project import main


if __name__ == "__main__":
    sys.exit(main())
