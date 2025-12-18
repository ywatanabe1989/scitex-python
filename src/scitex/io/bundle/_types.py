#!/usr/bin/env python3
# Timestamp: "2025-12-19 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/bundle/_types.py

"""
SciTeX Bundle Types and Constants.

Defines bundle types, extensions, and error classes used across the bundle module.
Supports both legacy formats (.figz, .pltz, .statsz) and unified .stx format.
"""

from typing import Dict, Tuple

__all__ = [
    # Type classes
    "BundleType",
    "StxType",
    # Error classes
    "BundleError",
    "BundleValidationError",
    "BundleNotFoundError",
    "NestedBundleNotFoundError",
    "CircularReferenceError",
    "DepthLimitError",
    "ConstraintError",
    # Constants
    "EXTENSIONS",
    "LEGACY_EXTENSIONS",
    "STX_EXTENSION",
    "FIGZ",
    "PLTZ",
    "STATSZ",
    "STX",
    # Defaults
    "TYPE_DEFAULTS",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
]

# Unified .stx extension (v2.0.0)
STX_EXTENSION: str = ".stx"

# Legacy bundle extensions (v1.0.0)
LEGACY_EXTENSIONS: Tuple[str, ...] = (".figz", ".pltz", ".statsz")

# All supported extensions
EXTENSIONS: Tuple[str, ...] = (STX_EXTENSION,) + LEGACY_EXTENSIONS

# Bundle type constants (for convenience)
FIGZ = "figz"
PLTZ = "pltz"
STATSZ = "statsz"
STX = "stx"

# Schema constants for unified .stx format
SCHEMA_NAME = "scitex.bundle"
SCHEMA_VERSION = "2.0.0"


class BundleType:
    """Bundle type constants (legacy extension-based).

    Usage:
        from scitex.io.bundle import BundleType

        if bundle_type == BundleType.FIGZ:
            ...
    """

    FIGZ = "figz"
    PLTZ = "pltz"
    STATSZ = "statsz"
    STX = "stx"  # Unified format


class StxType:
    """Type constants for unified .stx bundles.

    In .stx format, the type is stored in spec.json["type"],
    not determined by file extension.

    Usage:
        from scitex.io.bundle import StxType

        if bundle.type == StxType.FIGURE:
            ...
    """

    FIGURE = "figure"
    PLOT = "plot"
    STATS = "stats"
    IMAGE = "image"
    TEXT = "text"
    SHAPE = "shape"
    SYMBOL = "symbol"
    COMMENT = "comment"
    EQUATION = "equation"

    # Mapping from legacy extension-based types
    FROM_LEGACY = {
        "figz": "figure",
        "pltz": "plot",
        "statsz": "stats",
    }

    # Mapping to legacy extension-based types
    TO_LEGACY = {
        "figure": "figz",
        "plot": "pltz",
        "stats": "statsz",
    }


# Type-specific default constraints (per scitex-cloud architecture)
TYPE_DEFAULTS: Dict[str, Dict] = {
    "figure": {"allow_children": True, "max_depth": 3},
    "plot": {"allow_children": False, "max_depth": 1},
    "stats": {"allow_children": False, "max_depth": 1},
    "image": {"allow_children": False, "max_depth": 1},
    "text": {"allow_children": False, "max_depth": 1},
    "shape": {"allow_children": False, "max_depth": 1},
    "symbol": {"allow_children": False, "max_depth": 1},
    "comment": {"allow_children": False, "max_depth": 1},
    "equation": {"allow_children": False, "max_depth": 1},
}


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


class CircularReferenceError(BundleValidationError):
    """Error raised when circular reference is detected in bundle hierarchy.

    This occurs when a bundle references itself directly or indirectly
    through its children, detected via bundle_id tracking.
    """

    pass


class DepthLimitError(BundleValidationError):
    """Error raised when bundle nesting exceeds max_depth constraint.

    The max_depth is defined per bundle type in TYPE_DEFAULTS.
    Default is 3 for figures, 1 for leaf types.
    """

    pass


class ConstraintError(BundleValidationError):
    """Error raised when bundle violates its type constraints.

    Examples:
    - Leaf type (plot, stats) has children
    - Unknown type specified
    """

    pass


# EOF
