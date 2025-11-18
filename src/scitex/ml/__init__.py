#!/usr/bin/env python3
"""
SciTeX ML Module - Alias for AI module.

This module is an alias for scitex.ai and re-exports all its contents.
The 'ml' name is kept for backward compatibility and convenience.

Usage:
    from scitex.ai import classification  # Same as scitex.ai.classification
    from scitex.ai.metrics import calc_bacc  # Same as scitex.ai.metrics
"""

# Re-export everything from ai module
from scitex.ai import *  # noqa: F403,F401

__all__ = [
    "classification",
    "metrics",
    "plt",
    "feature_selection",
    "feature_extraction",
    "training",
    "clustering",
    "loss",
    "optim",
    "activation",
    "sklearn",
    "sk",
    "utils",
]
