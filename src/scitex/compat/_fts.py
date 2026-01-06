#!/usr/bin/env python3
# Timestamp: 2026-01-07
# File: src/scitex/compat/_fts.py

"""
DEPRECATED: scitex.fts → scitex.io.bundle

This module provides backward compatibility for code using scitex.fts.
Update your imports to use scitex.io.bundle instead.
"""

import warnings

warnings.warn(
    "scitex.fts is deprecated. Use scitex.io.bundle instead.",
    DeprecationWarning,
    stacklevel=3,  # Adjust for import chain: fts/__init__.py -> compat/_fts.py
)

# Version (for backward compat)
__version__ = "1.0.0"

# Re-export from io.bundle
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

# Try to import Encoding/Theme from kinds
try:
    from scitex.io.bundle.kinds._plot import Encoding, Theme
except ImportError:
    Encoding = None
    Theme = None

# Try to import Stats
try:
    from scitex.stats import Stats
except ImportError:
    Stats = None

# Availability flags
FTS_AVAILABLE = True
FTS_VERSION = __version__

# Legacy aliases
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
