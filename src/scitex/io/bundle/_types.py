#!/usr/bin/env python3
# Timestamp: 2026-01-07
# File: src/scitex/io/bundle/_types.py

"""
SciTeX Bundle Types and Constants.

Defines bundle types, extensions, and error classes used across the bundle module.

Extension formats:
    - Legacy: .figz, .pltz, .statsz (compact but not obviously zip)
    - New: .figure.zip, .plot.zip, .stats.zip (hybrid, OS-friendly)

Both formats are fully supported and interchangeable.
"""

from typing import Dict, Tuple

__all__ = [
    "BundleType",
    "BundleError",
    "BundleValidationError",
    "BundleNotFoundError",
    "NestedBundleNotFoundError",
    "EXTENSIONS",
    "EXTENSIONS_LEGACY",
    "EXTENSIONS_NEW",
    "DIR_EXTENSIONS",
    "DIR_EXTENSIONS_LEGACY",
    "DIR_EXTENSIONS_NEW",
    "EXTENSION_MAP",
    "FIGZ",
    "PLTZ",
    "STATSZ",
    "FIGURE",
    "PLOT",
    "STATS",
]

# =============================================================================
# Legacy extensions (still fully supported)
# =============================================================================
EXTENSIONS_LEGACY: Tuple[str, ...] = (".figz", ".pltz", ".statsz")
DIR_EXTENSIONS_LEGACY: Tuple[str, ...] = (".figz.d", ".pltz.d", ".statsz.d")

# =============================================================================
# New hybrid extensions (OS-friendly .zip)
# =============================================================================
EXTENSIONS_NEW: Tuple[str, ...] = (".figure.zip", ".plot.zip", ".stats.zip")
DIR_EXTENSIONS_NEW: Tuple[str, ...] = (".figure", ".plot", ".stats")

# =============================================================================
# All supported extensions
# =============================================================================
EXTENSIONS: Tuple[str, ...] = EXTENSIONS_LEGACY + EXTENSIONS_NEW
DIR_EXTENSIONS: Tuple[str, ...] = DIR_EXTENSIONS_LEGACY + DIR_EXTENSIONS_NEW

# =============================================================================
# Mapping from legacy to new
# =============================================================================
EXTENSION_MAP: Dict[str, str] = {
    # ZIP extensions
    ".pltz": ".plot.zip",
    ".figz": ".figure.zip",
    ".statsz": ".stats.zip",
    # Directory extensions
    ".pltz.d": ".plot",
    ".figz.d": ".figure",
    ".statsz.d": ".stats",
}

# Reverse mapping
EXTENSION_MAP_REVERSE: Dict[str, str] = {v: k for k, v in EXTENSION_MAP.items()}

# =============================================================================
# Bundle type constants (legacy)
# =============================================================================
FIGZ = "figz"
PLTZ = "pltz"
STATSZ = "statsz"

# =============================================================================
# Bundle type constants (new names)
# =============================================================================
FIGURE = "figure"
PLOT = "plot"
STATS = "stats"


class BundleType:
    """Bundle type constants.

    Usage:
        from scitex.io.bundle import BundleType

        if bundle_type == BundleType.FIGZ:
            ...
        # Or use new names:
        if bundle_type == BundleType.FIGURE:
            ...
    """

    # Legacy type names
    FIGZ = "figz"
    PLTZ = "pltz"
    STATSZ = "statsz"

    # New type names (aliases)
    FIGURE = "figure"
    PLOT = "plot"
    STATS = "stats"

    @classmethod
    def normalize(cls, type_name: str) -> str:
        """Normalize type name to new format.

        Args:
            type_name: Type name (legacy or new)

        Returns:
            Normalized type name (new format)
        """
        mapping = {
            "figz": "figure",
            "pltz": "plot",
            "statsz": "stats",
        }
        return mapping.get(type_name, type_name)

    @classmethod
    def to_legacy(cls, type_name: str) -> str:
        """Convert type name to legacy format.

        Args:
            type_name: Type name (legacy or new)

        Returns:
            Legacy type name
        """
        mapping = {
            "figure": "figz",
            "plot": "pltz",
            "stats": "statsz",
        }
        return mapping.get(type_name, type_name)


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
