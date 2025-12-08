#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 07:24:01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/sh/_types.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/sh/_types.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

__FILE__ = __file__

from typing import List, Literal, TypedDict


class ShellResult(TypedDict):
    stdout: str
    stderr: str
    exit_code: int
    success: bool


CommandInput = List[str]
ReturnFormat = Literal["dict", "str"]

# EOF
