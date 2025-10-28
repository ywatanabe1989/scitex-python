#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-09-14 11:29:19 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/_sh.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Backward compatibility wrapper
# Import from new modular structure
from .sh import sh, sh_run, quote

__all__ = ['sh', 'sh_run', 'quote']

# EOF
