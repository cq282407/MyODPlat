#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : validate_data.py
# @Project   : ODPlatform
# @Function  : validate_data development entry point.

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLATFORM_SRC = REPO_ROOT / "apps" / "platform" / "src"

sys.path.insert(0, str(PLATFORM_SRC))

from od_platform.cli.validate_data import main


if __name__ == "__main__":
    sys.exit(main())
