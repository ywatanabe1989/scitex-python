#!/usr/bin/env python3
# Timestamp: 2026-01-07
# File: src/scitex/io/bundle/__init__.py

"""
SciTeX Bundle I/O - Unified bundle handling for bundles.

This module provides a clean API for working with SciTeX bundles:
    - .figz/.figure.zip - Publication Figure Bundle (panels + layout)
    - .pltz/.plot.zip - Reproducible Plot Bundle (data + spec + exports)
    - .statsz/.stats.zip - Statistical Results Bundle (stats + metadata)

Each bundle can exist in two forms:
    - ZIP archive: Figure1.figz, plot.pltz, or Figure1.figure.zip, plot.plot.zip
    - Directory: Figure1.figz.d/, plot.pltz.d/, or Figure1.figure/, plot.plot/

Usage:
    import scitex.io.bundle as bundle

    # Load a bundle
    data = bundle.load("Figure1.figz")  # or "Figure1.figure.zip"

    # Save a bundle
    bundle.save(data, "output.pltz", as_zip=True)

    # Copy a bundle
    bundle.copy("template.pltz", "my_plot.pltz.d")

    # Access ZIP bundles in-memory
    with bundle.ZipBundle("figure.figz") as zb:
        spec = zb.read_json("spec.json")
        data = zb.read_csv("data.csv")

    # Access nested bundles (pltz inside figz)
    preview = bundle.nested.get_preview("Figure1.figz/A.pltz.d")
    spec = bundle.nested.get_json("Figure1.figz/A.pltz.d/spec.json")

Note: This module now includes functionality from the deprecated scitex.fts module.
"""

# Types and constants
# Nested bundle access as namespace
from . import _nested as nested

# Core operations
from ._core import (
    copy,
    dir_to_zip_path,
    get_type,
    is_bundle,
    load,
    pack,
    save,
    unpack,
    validate,
    validate_spec,
    zip_to_dir_path,
)

# Dataclasses (from deprecated scitex.fts)
from ._dataclasses import BBox, DataInfo, Node, NodeRefs, SizeMM

# FTS class and factory functions (from deprecated scitex.fts)
from ._FTS import FTS, create_bundle, from_matplotlib, load_bundle
from ._types import (
    DIR_EXTENSIONS,
    DIR_EXTENSIONS_LEGACY,
    DIR_EXTENSIONS_NEW,
    EXTENSION_MAP,
    EXTENSIONS,
    EXTENSIONS_LEGACY,
    EXTENSIONS_NEW,
    FIGURE,
    FIGZ,
    PLOT,
    PLTZ,
    STATS,
    STATSZ,
    BundleError,
    BundleNotFoundError,
    BundleType,
    BundleValidationError,
    NestedBundleNotFoundError,
)

# ZipBundle class and functions
from ._zip import ZipBundle, zip_directory
from ._zip import create as create_zip
from ._zip import open as open_zip

# NodeType enum placeholder for backward compat
NodeType = Node  # Alias for backward compatibility

__all__ = [
    # Types
    "BundleType",
    "BundleError",
    "BundleValidationError",
    "BundleNotFoundError",
    "NestedBundleNotFoundError",
    # Constants - Extensions
    "EXTENSIONS",
    "EXTENSIONS_LEGACY",
    "EXTENSIONS_NEW",
    "DIR_EXTENSIONS",
    "DIR_EXTENSIONS_LEGACY",
    "DIR_EXTENSIONS_NEW",
    "EXTENSION_MAP",
    # Constants - Type names (legacy)
    "FIGZ",
    "PLTZ",
    "STATSZ",
    # Constants - Type names (new)
    "FIGURE",
    "PLOT",
    "STATS",
    # Core operations
    "load",
    "save",
    "copy",
    "pack",
    "unpack",
    "validate",
    "validate_spec",
    "is_bundle",
    "get_type",
    "dir_to_zip_path",
    "zip_to_dir_path",
    # ZipBundle
    "ZipBundle",
    "open_zip",
    "create_zip",
    "zip_directory",
    # Nested access namespace
    "nested",
    # FTS class and factory functions (backward compat with scitex.fts)
    "FTS",
    "load_bundle",
    "create_bundle",
    "from_matplotlib",
    # Dataclasses (backward compat with scitex.fts)
    "Node",
    "NodeType",
    "NodeRefs",
    "BBox",
    "SizeMM",
    "DataInfo",
]

# EOF
