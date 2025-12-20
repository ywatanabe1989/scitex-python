#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_stats/__init__.py

"""FTS Stats - Statistics dataclasses with GUI support."""

# Public dataclasses
from ._dataclasses import (
    STATS_VERSION,
    Stats,
    # Type aliases
    PositionMode,
    UnitType,
    SymbolStyle,
    # GUI classes
    Position,
    StatStyling,
    StatPositioning,
    # Core classes
    DataRef,
    EffectSize,
    StatMethod,
    StatResult,
    StatDisplay,
    Analysis,
)

__all__ = [
    "STATS_VERSION",
    "Stats",
    # Type aliases
    "PositionMode",
    "UnitType",
    "SymbolStyle",
    # GUI classes
    "Position",
    "StatStyling",
    "StatPositioning",
    # Core classes
    "DataRef",
    "EffectSize",
    "StatMethod",
    "StatResult",
    "StatDisplay",
    "Analysis",
]

# EOF
