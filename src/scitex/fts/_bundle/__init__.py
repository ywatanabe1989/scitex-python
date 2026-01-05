#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/__init__.py

"""FTS Bundle - Core bundle functionality."""

# FTS class
from ._FTS import FTS, create_bundle, from_matplotlib, load_bundle

# Core dataclasses users need
from ._dataclasses import BBox, DataInfo, Node, SizeMM

# Type enumeration
from ._utils import NodeType

# Error classes
from ._utils import BundleError, BundleNotFoundError, BundleValidationError

__all__ = [
    # FTS
    "FTS",
    "load_bundle",
    "create_bundle",
    "from_matplotlib",
    # Core dataclasses
    "Node",
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
