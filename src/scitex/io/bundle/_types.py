#!/usr/bin/env python3
# Timestamp: 2026-01-07
# File: src/scitex/io/bundle/_types.py

"""
SciTeX Bundle Types and Constants.

Defines bundle types, extensions, and error classes used across the bundle module.

Extension formats:
    - ZIP: .figure.zip, .plot.zip, .stats.zip
    - Directory: .figure/, .plot/, .stats/
"""

from typing import Tuple

__all__ = [
    "BundleType",
    "BundleError",
    "BundleValidationError",
    "BundleNotFoundError",
    "NestedBundleNotFoundError",
    "EXTENSIONS",
    "DIR_EXTENSIONS",
    "FIGURE",
    "PLOT",
    "STATS",
]

# =============================================================================
# ZIP extensions
# =============================================================================
EXTENSIONS: Tuple[str, ...] = (".figure.zip", ".plot.zip", ".stats.zip")

# =============================================================================
# Directory extensions
# =============================================================================
DIR_EXTENSIONS: Tuple[str, ...] = (".figure", ".plot", ".stats")

# =============================================================================
# Bundle type constants
# =============================================================================
FIGURE = "figure"
PLOT = "plot"
STATS = "stats"


class BundleType:
    """Bundle type constants.

    Usage:
        from scitex.io.bundle import BundleType

        if bundle_type == BundleType.FIGURE:
            ...
    """

    FIGURE = "figure"
    PLOT = "plot"
    STATS = "stats"

    @classmethod
    def normalize(cls, bundle_type: str) -> str:
        """Normalize bundle type to standard name.

        Args:
            bundle_type: Bundle type string.

        Returns:
            Normalized type ('figure', 'plot', 'stats').
        """
        return bundle_type.lower()


class BundleError(Exception):
    """Base exception for bundle operations."""

    pass


class BundleValidationError(BundleError, ValueError):
    """Error raised when bundle validation fails."""

    pass


class BundleNotFoundError(BundleError, FileNotFoundError):
    """Error raised when a bundle is not found."""

    pass


class NestedBundleNotFoundError(BundleNotFoundError):
    """Error raised when a nested bundle or file within it is not found."""

    pass


# EOF
