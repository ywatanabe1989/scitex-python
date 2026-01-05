#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/io/__init__.py

"""
I/O operations for scitex.plt - Plot bundles (.pltz).

This module handles:
    - .pltz bundle load/save operations (legacy and layered formats)
    - CSV data export/import
    - Hitmap generation and storage
    - Bundle overview generation

Layered Format (v2.0):
    plot.pltz.d/
        spec.json           # Semantic: WHAT to plot
        style.json          # Appearance: HOW it looks
        data.csv            # Raw data
        exports/            # Preview images
        cache/              # Regenerable geometry data
"""

from ._bundle import (
    validate_pltz_spec,
    load_pltz_bundle,
    save_pltz_bundle,
    generate_bundle_overview,
    PLTZ_SCHEMA_SPEC,
)

from ._layered_bundle import (
    save_layered_pltz_bundle,
    load_layered_pltz_bundle,
    merge_layered_bundle,
    is_layered_bundle,
)

__all__ = [
    # Legacy bundle operations
    "validate_pltz_spec",
    "load_pltz_bundle",
    "save_pltz_bundle",
    "generate_bundle_overview",
    "PLTZ_SCHEMA_SPEC",
    # Layered bundle operations (v2.0)
    "save_layered_pltz_bundle",
    "load_layered_pltz_bundle",
    "merge_layered_bundle",
    "is_layered_bundle",
]

# EOF
