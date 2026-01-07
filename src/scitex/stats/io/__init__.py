#!/usr/bin/env python3
# Timestamp: "2025-12-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/io/__init__.py

"""
I/O operations for scitex.stats - Statistical results bundles (.stats).

This module handles:
    - .stats bundle load/save operations
    - Statistical comparison metadata
    - P-value and effect size validation
"""

from ._bundle import (
    STATS_SCHEMA_SPEC,
    load_stats_bundle,
    save_stats_bundle,
    validate_stats_spec,
)

__all__ = [
    "validate_stats_spec",
    "load_stats_bundle",
    "save_stats_bundle",
    "STATS_SCHEMA_SPEC",
]

# EOF
