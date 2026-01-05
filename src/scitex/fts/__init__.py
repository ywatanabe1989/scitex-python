#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/__init__.py

"""
SciTeX FTS (Figure-Table-Statistics) - The single source of truth for bundle schemas.

FTS defines a standardized format for reproducible scientific figures and tables:
- Self-contained bundles with data, visualization spec, and stats
- Clear separation: Node (structure), Encoding (data mapping), Theme (aesthetics)
- Full provenance tracking for scientific reproducibility

Usage:
    from scitex.fts import FTS, Node, Encoding, Theme

    # Create new bundle
    bundle = FTS("my_plot.zip", create=True, node_type="plot")
    bundle.encoding = {"traces": [{"trace_id": "t1", "x": {"column": "time"}}]}
    bundle.save()

    # Load existing bundle
    bundle = FTS("my_plot.zip")
    print(bundle.node.type)  # "plot"
"""

# Version
__version__ = "1.0.0"

# =============================================================================
# Public API - What users need
# =============================================================================

# FTS class (main entry point)
from ._bundle import FTS, create_bundle, from_matplotlib, load_bundle

# Core dataclasses users interact with
from ._bundle import Node, BBox, SizeMM, DataInfo
from ._fig import Encoding, Theme
from ._stats import Stats

# Type enumeration
from ._bundle import NodeType

# Error classes for exception handling
from ._bundle import (
    BundleError,
    BundleNotFoundError,
    BundleValidationError,
)

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
