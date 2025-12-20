#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/stats/_schema.py
# Timestamp: 2025-12-20
"""
Statistical Result Schema - DEPRECATED

This module is deprecated. Import from scitex.fts._stats instead:
    from scitex.fts._stats import Position, StatStyling, StatPositioning
"""

import warnings

warnings.warn(
    "scitex.stats._schema is deprecated. Import from scitex.fts._stats instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from FTS (new single source of truth)
from scitex.fts._stats import (
    # Type aliases
    PositionMode,
    UnitType,
    SymbolStyle,
    # Position and styling
    Position,
    StatStyling,
    StatPositioning,
)

# StatResult is no longer a dataclass - use dicts for test results
StatResult = dict

def create_stat_result(
    test_type: str,
    statistic_name: str,
    statistic_value: float,
    p_value: float,
    **kwargs,
) -> dict:
    """Create a stat result dict (deprecated, use simple dicts instead)."""
    from scitex.stats.utils import p2stars

    return {
        "test_type": test_type,
        "test_category": kwargs.get("test_category", "other"),
        "statistic": {"name": statistic_name, "value": statistic_value},
        "p_value": p_value,
        "stars": p2stars(p_value, ns_symbol=False),
        **{k: v for k, v in kwargs.items() if k != "test_category"},
    }

__all__ = [
    # Type aliases
    "PositionMode",
    "UnitType",
    "SymbolStyle",
    # Position and styling
    "Position",
    "StatStyling",
    "StatPositioning",
    # Deprecated
    "StatResult",
    "create_stat_result",
]

# EOF
