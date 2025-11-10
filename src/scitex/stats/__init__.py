#!/usr/bin/env python3
"""
SciTeX Stats Module - Professional Statistical Testing Framework

Clean, normalized API for statistical analysis with publication-ready outputs.

Organized submodules:
- correct: 3 multiple comparison correction methods
- effect_sizes: Effect size computations and interpretations
- power: Statistical power analysis
- utils: Formatters and normalizers (for backward compatibility)
"""

# Import new organized submodules
from . import correct
from . import effect_sizes
from . import power
from . import utils
from . import posthoc
from . import descriptive

# Export commonly used functions for convenience
from .descriptive import describe

__all__ = [
    # Main submodules
    "correct",
    "effect_sizes",
    "power",
    "utils",
    "posthoc",
    "descriptive",
    # Convenience exports
    "describe",
]

__version__ = "2.0.0"
