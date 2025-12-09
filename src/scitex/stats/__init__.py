#!/usr/bin/env python3
"""
SciTeX Stats Module - Professional Statistical Testing Framework

Clean, normalized API for statistical analysis with publication-ready outputs.

Organized submodules:
- correct: 3 multiple comparison correction methods
- effect_sizes: Effect size computations and interpretations
- power: Statistical power analysis
- tests: Statistical tests (correlation, etc.)
- utils: Formatters and normalizers (for backward compatibility)
- _schema: Core StatResult schema for standardized test results
"""

# Import new organized submodules
from . import correct
from . import effect_sizes
from . import power
from . import utils
from . import posthoc
from . import descriptive
from . import tests
from . import _schema

# Export commonly used functions and classes for convenience
from .descriptive import describe
from ._schema import (
    StatResult,
    Position,
    StatStyling,
    StatPositioning,
    create_stat_result,
)

__all__ = [
    # Main submodules
    "correct",
    "effect_sizes",
    "power",
    "utils",
    "posthoc",
    "descriptive",
    "tests",
    "_schema",
    # Convenience exports
    "describe",
    "StatResult",
    "Position",
    "StatStyling",
    "StatPositioning",
    "create_stat_result",
]

__version__ = "2.0.0"
