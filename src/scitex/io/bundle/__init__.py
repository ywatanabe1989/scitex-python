#!/usr/bin/env python3
# Timestamp: "2025-12-19 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/bundle/__init__.py

"""
SciTeX Bundle I/O - Unified bundle handling for .stx and legacy formats.

This module provides a clean API for working with SciTeX bundles:
    - .stx - Unified SciTeX Bundle (v2.0.0) - all types via spec["type"]
    - .figz - Publication Figure Bundle (legacy, auto-converted)
    - .pltz - Reproducible Plot Bundle (legacy, auto-converted)
    - .statsz - Statistical Results Bundle (legacy, auto-converted)

Each bundle can exist in two forms:
    - ZIP archive: Figure1.stx, Figure1.figz
    - Directory: Figure1.stx.d/, Figure1.figz.d/

Migration Strategy: "Save as .stx, read all formats"
    - Loader auto-detects format and converts legacy to .stx in memory
    - Saver produces .stx format (warns on legacy extensions)

Usage:
    import scitex.io.bundle as bundle

    # Load any bundle format (auto-detects)
    data = bundle.load("Figure1.stx")    # Native .stx
    data = bundle.load("Figure1.figz")   # Auto-converts legacy

    # Save as unified .stx
    bundle.save(data, "output.stx", as_zip=True)

    # Access ZIP bundles in-memory
    with bundle.ZipBundle("figure.stx") as zb:
        spec = zb.read_json("spec.json")
        data = zb.read_csv("data.csv")

    # Self-recursive figures (figures can contain figures)
    data = bundle.load_stx("nested_figure.stx")
    print(data["children"])  # Child bundles
"""

# Types and constants
# Nested bundle access as namespace
from . import _nested as nested

# Core operations
from ._core import (
    copy,
    dir_to_zip_path,
    get_content_type,
    get_type,
    is_bundle,
    load,
    load_stx,
    pack,
    save,
    unpack,
    validate,
    validate_spec,
    zip_to_dir_path,
)

# Stx spec handling
from ._stx import (
    create_stx_spec,
    generate_bundle_id,
    get_default_constraints,
    get_stx_type,
    is_stx_format,
    migrate_v1_to_v2,
    normalize_spec,
    validate_stx_bundle,
)
from ._types import (
    # Extensions
    EXTENSIONS,
    # Type constants
    FIGZ,
    LEGACY_EXTENSIONS,
    PLTZ,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    STATSZ,
    STX,
    STX_EXTENSION,
    # Defaults
    TYPE_DEFAULTS,
    # Error classes
    BundleError,
    BundleNotFoundError,
    # Type classes
    BundleType,
    BundleValidationError,
    CircularReferenceError,
    ConstraintError,
    DepthLimitError,
    NestedBundleNotFoundError,
    StxType,
)

# ZipBundle class and functions
from ._zip import ZipBundle, zip_directory
from ._zip import create as create_zip
from ._zip import open as open_zip

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
    "ConstraintError",
    "DepthLimitError",
    # Constants
    "EXTENSIONS",
    "LEGACY_EXTENSIONS",
    "STX_EXTENSION",
    "FIGZ",
    "PLTZ",
    "STATSZ",
    "STX",
    "TYPE_DEFAULTS",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    # Core operations
    "load",
    "load_stx",
    "save",
    "copy",
    "pack",
    "unpack",
    "validate",
    "validate_spec",
    "is_bundle",
    "get_type",
    "get_content_type",
    "dir_to_zip_path",
    "zip_to_dir_path",
    # Stx spec handling
    "normalize_spec",
    "create_stx_spec",
    "migrate_v1_to_v2",
    "validate_stx_bundle",
    "generate_bundle_id",
    "get_stx_type",
    "get_default_constraints",
    "is_stx_format",
    # ZipBundle
    "ZipBundle",
    "open_zip",
    "create_zip",
    "zip_directory",
    # Nested access namespace
    "nested",
]

# EOF
