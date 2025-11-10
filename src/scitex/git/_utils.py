#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/git/ops.py

"""
Git operations utilities.
"""

import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _in_directory(path: Path):
    cwd_original = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd_original)


__all__ = [
    "_in_directory",
]

# EOF
