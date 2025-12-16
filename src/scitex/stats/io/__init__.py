#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/io/__init__.py

"""
I/O operations for scitex.stats - Statistical results bundles (.statsz).

This module handles:
    - .statsz bundle load/save operations
    - Statistical comparison metadata
    - P-value and effect size validation
"""

from ._bundle import (
    validate_statsz_spec,
    load_statsz_bundle,
    save_statsz_bundle,
    STATSZ_SCHEMA_SPEC,
)

__all__ = [
    "validate_statsz_spec",
    "load_statsz_bundle",
    "save_statsz_bundle",
    "STATSZ_SCHEMA_SPEC",
]

# EOF
