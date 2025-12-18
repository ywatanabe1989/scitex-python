#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-16 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/bundle/_types.py

"""
SciTeX Bundle Types and Constants.

Defines bundle types, extensions, and error classes used across the bundle module.
"""

from typing import Tuple

__all__ = [
    "BundleType",
    "BundleError",
    "BundleValidationError",
    "BundleNotFoundError",
    "NestedBundleNotFoundError",
    "EXTENSIONS",
    "FIGZ",
    "PLTZ",
    "STATSZ",
]

# Bundle extensions
EXTENSIONS: Tuple[str, ...] = (".figz", ".pltz", ".statsz")

# Bundle type constants (for convenience)
FIGZ = "figz"
PLTZ = "pltz"
STATSZ = "statsz"


class BundleType:
    """Bundle type constants.

    Usage:
        from scitex.io.bundle import BundleType

        if bundle_type == BundleType.FIGZ:
            ...
    """

    FIGZ = "figz"
    PLTZ = "pltz"
    STATSZ = "statsz"


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
