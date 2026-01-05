#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_bundle/_utils/__init__.py

"""FTS Bundle utilities - types, constants, errors, and helpers."""

# Constants
from ._const import EXTENSIONS, SCHEMA_NAME, SCHEMA_VERSION, ZIP_EXTENSION

# Types
from ._types import (
    TYPE_DEFAULTS,
    BundleType,
    NodeType,
    get_default_constraints,
)

# Errors
from ._errors import (
    BundleError,
    BundleNotFoundError,
    BundleValidationError,
    CircularReferenceError,
    ConstraintError,
    DepthLimitError,
    NestedBundleNotFoundError,
)

# Generation
from ._generate import generate_bundle_id

__all__ = [
    # Constants
    "ZIP_EXTENSION",
    "EXTENSIONS",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    # Types
    "NodeType",
    "BundleType",
    "TYPE_DEFAULTS",
    "get_default_constraints",
    # Errors
    "BundleError",
    "BundleValidationError",
    "BundleNotFoundError",
    "NestedBundleNotFoundError",
    "CircularReferenceError",
    "DepthLimitError",
    "ConstraintError",
    # Generation
    "generate_bundle_id",
]

# EOF
