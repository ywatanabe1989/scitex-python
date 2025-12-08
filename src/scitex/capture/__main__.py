#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-18 09:55:55 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/capture/__main__.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/capture/__main__.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Entry point for python -m scitex.capture
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())

# EOF
