# -*- coding: utf-8 -*-
"""
common/__init__.py
让 `from common.info import banner` 在任意子目录下都能工作。
"""
import sys
from pathlib import Path

# 把项目根目录加入 sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
