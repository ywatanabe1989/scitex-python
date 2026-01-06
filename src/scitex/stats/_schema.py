#!/usr/bin/env python3
# File: ./src/scitex/stats/_schema.py
# Timestamp: 2025-12-20
"""
Statistical Result Schema - DEPRECATED

This module is deprecated. Import from scitex.schema._stats instead:
    from scitex.schema._stats import Position, StatStyling, StatPositioning, StatResult

For the new simplified API, use scitex.io.bundle._stats.
"""

import warnings

warnings.warn(
    "scitex.stats._schema is deprecated. Import from scitex.schema._stats instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from schema module (the OLD API with full StatResult dataclass)
from scitex.schema._stats import (
    # Position and styling
    Position,
    # Type aliases
    PositionMode,
    StatPositioning,
    # StatResult dataclass (old API)
    StatResult,
    StatStyling,
    SymbolStyle,
    UnitType,
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
    # Deprecated but functional
    "StatResult",
    "create_stat_result",
]

# EOF
