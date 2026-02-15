#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_stats/_dataclasses/__init__.py

"""Statistics-specific dataclasses for FTS with GUI support."""

from ._Stats import (  # Type aliases; GUI classes; Core classes
    STATS_VERSION,
    Analysis,
    DataRef,
    EffectSize,
    Position,
    PositionMode,
    StatDisplay,
    StatMethod,
    StatPositioning,
    StatResult,
    Stats,
    StatStyling,
    SymbolStyle,
    UnitType,
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
