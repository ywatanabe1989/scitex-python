#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-16 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/bundle/__init__.py

"""
SciTeX Bundle I/O - Unified bundle handling for .figz, .pltz, .statsz formats.

This module provides a clean API for working with SciTeX bundles:
    - .figz - Publication Figure Bundle (panels + layout)
    - .pltz - Reproducible Plot Bundle (data + spec + exports)
    - .statsz - Statistical Results Bundle (stats + metadata)

Each bundle can exist in two forms:
    - ZIP archive: Figure1.figz, plot.pltz
    - Directory: Figure1.figz.d/, plot.pltz.d/

Usage:
    import scitex.io.bundle as bundle

    # Load a bundle
    data = bundle.load("Figure1.figz")

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
"""

# Types and constants
from ._types import (
    EXTENSIONS,
    FIGZ,
    PLTZ,
    STATSZ,
    BundleError,
    BundleNotFoundError,
    BundleType,
    BundleValidationError,
    NestedBundleNotFoundError,
)

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

# ZipBundle class and functions
from ._zip import ZipBundle
from ._zip import create as create_zip
from ._zip import open as open_zip
from ._zip import zip_directory

# Nested bundle access as namespace
from . import _nested as nested

__all__ = [
    # Types
    "BundleType",
    "BundleError",
    "BundleValidationError",
    "BundleNotFoundError",
    "NestedBundleNotFoundError",
    # Constants
    "EXTENSIONS",
    "FIGZ",
    "PLTZ",
    "STATSZ",
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
]

# EOF
