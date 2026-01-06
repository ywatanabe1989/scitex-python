#!/usr/bin/env python3
# Timestamp: 2026-01-07
# File: src/scitex/fts/__init__.py
#
# DEPRECATED: This module is deprecated. Use scitex.io.bundle instead.
# This file provides backward compatibility and will be removed in a future version.

"""
SciTeX FTS (Figure-Table-Statistics) - DEPRECATED.

This module has been merged into scitex.io.bundle.
Please update your imports:

    # Old (deprecated)
    from scitex.fts import FTS, Node

    # New (recommended)
    from scitex.io.bundle import FTS, Node
"""

import warnings

warnings.warn(
    "scitex.fts is deprecated. Use scitex.io.bundle instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Version
__version__ = "1.0.0"

# Re-export from io.bundle
try:
    from scitex.io.bundle import (
        FTS,
        BBox,
        BundleError,
        BundleNotFoundError,
        BundleValidationError,
        DataInfo,
        Node,
        NodeType,
        SizeMM,
        create_bundle,
        from_matplotlib,
        load_bundle,
    )
except ImportError:
    # Fallback to internal imports during transition
    from ._bundle import (
        FTS,
        BBox,
        BundleError,
        BundleNotFoundError,
        BundleValidationError,
        DataInfo,
        Node,
        NodeType,
        SizeMM,
        create_bundle,
        from_matplotlib,
        load_bundle,
    )

# Try to import these from their respective locations
try:
    from scitex.io.bundle.kinds._plot import Encoding, Theme
except ImportError:
    try:
        from ._fig import Encoding, Theme
    except ImportError:
        Encoding = None
        Theme = None

try:
    from scitex.stats import Stats
except ImportError:
    try:
        from ._stats import Stats
    except ImportError:
        Stats = None

# Availability flags
FTS_AVAILABLE = True
FTS_VERSION = __version__

# Legacy aliases for backwards compatibility
FSB = FTS
FSB_AVAILABLE = FTS_AVAILABLE
FSB_VERSION = FTS_VERSION

__all__ = [
    # Version
    "__version__",
    "FTS_AVAILABLE",
    "FTS_VERSION",
    # Legacy aliases
    "FSB_AVAILABLE",
    "FSB_VERSION",
    "FSB",
    # FTS class
    "FTS",
    "load_bundle",
    "create_bundle",
    "from_matplotlib",
    # Core dataclasses
    "Node",
    "Encoding",
    "Theme",
    "Stats",
    "BBox",
    "SizeMM",
    "DataInfo",
    # Types
    "NodeType",
    # Errors
    "BundleError",
    "BundleNotFoundError",
    "BundleValidationError",
]

# EOF
