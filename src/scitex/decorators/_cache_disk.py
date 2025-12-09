#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/decorators/_cache_disk.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/decorators/_cache_disk.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import functools

from joblib import Memory as _Memory

from scitex.config import get_paths


def cache_disk(func):
    """Disk caching decorator that uses joblib.Memory.

    Usage:
        @cache_disk
        def expensive_function(x):
            return x ** 2
    """
    cache_dir = str(get_paths().function_cache)
    memory = _Memory(cache_dir, verbose=0)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cached_func = memory.cache(func)
        return cached_func(*args, **kwargs)

    return wrapper


# EOF
