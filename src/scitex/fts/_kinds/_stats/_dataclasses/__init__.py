#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_stats/_dataclasses/__init__.py

"""Statistics-specific dataclasses for FTS with GUI support."""

from ._Stats import (
    STATS_VERSION,
    # Type aliases
    PositionMode,
    UnitType,
    SymbolStyle,
    # GUI classes
    Position,
    StatStyling,
    StatPositioning,
    # Core classes
    Analysis,
    DataRef,
    EffectSize,
    StatDisplay,
    StatMethod,
    StatResult,
    Stats,
)

__all__ = [
    # Version
    "STATS_VERSION",
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
    "Stats",
]

# EOF
