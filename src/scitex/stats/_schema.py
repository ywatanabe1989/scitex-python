#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/stats/_schema.py
# Time-stamp: "2024-12-09 09:20:00 (ywatanabe)"
"""
Statistical Result Schema - Re-exports from central schema module.

This module re-exports all schema classes from scitex.schema._stats
for backward compatibility. The canonical definitions now live in
scitex.schema._stats as the single source of truth.

Note: New code should import directly from scitex.schema:
    from scitex.schema import StatResult, Position, StatStyling

This module exists for backward compatibility with existing code that
imports from scitex.stats._schema.
"""

# Re-export everything from the central schema module
from scitex.schema._stats import (
    # Type aliases
    PositionMode,
    UnitType,
    SymbolStyle,
    # Position and styling
    Position,
    StatStyling,
    StatPositioning,
    # Main result class
    StatResult,
    # Convenience function
    create_stat_result,
)

__all__ = [
    # Type aliases
    "PositionMode",
    "UnitType",
    "SymbolStyle",
    # Position and styling
    "Position",
    "StatStyling",
    "StatPositioning",
    # Main result class
    "StatResult",
    # Convenience function
    "create_stat_result",
]


# EOF
