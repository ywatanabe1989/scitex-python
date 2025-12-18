#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-17
# File: ./src/scitex/fig/io/__init__.py
"""
I/O operations for scitex.fig figure bundles.

.figz Bundle Architecture:
    Figure1.figz.d/
        ├── spec.json       # Figure specification
        ├── style.json      # Figure style
        ├── A.pltz.d/       # Panel A bundle
        ├── B.pltz.d/       # Panel B bundle
        ├── exports/        # Figure exports (PNG, SVG)
        └── cache/          # Cached geometry

Schema Version: 1.0.0
"""

from ._bundle import (
    validate_figz_spec,
    load_figz_bundle,
    save_figz_bundle,
    export_figz_bundle,
    FIGZ_SCHEMA_SPEC,
)

__all__ = [
    # Bundle operations
    "validate_figz_spec",
    "load_figz_bundle",
    "save_figz_bundle",
    "export_figz_bundle",
    "FIGZ_SCHEMA_SPEC",
]

# EOF
