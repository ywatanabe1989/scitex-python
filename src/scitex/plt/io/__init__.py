#!/usr/bin/env python3
# Timestamp: "2025-12-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/io/__init__.py

"""
I/O operations for scitex.plt - Plot bundles (.plot).

This module handles:
    - .plot bundle load/save operations (legacy and layered formats)
    - CSV data export/import
    - Hitmap generation and storage
    - Bundle overview generation

Layered Format (v2.0):
    plot.plot/
        spec.json           # Semantic: WHAT to plot
        style.json          # Appearance: HOW it looks
        data.csv            # Raw data
        exports/            # Preview images
        cache/              # Regenerable geometry data
"""

from ._bundle import (
    PLOT_SCHEMA_SPEC,
    generate_bundle_overview,
    load_plot_bundle,
    save_plot_bundle,
    validate_plot_spec,
)
from ._layered_bundle import (
    is_layered_bundle,
    load_layered_plot_bundle,
    merge_layered_bundle,
    save_layered_plot_bundle,
)

__all__ = [
    # Bundle operations
    "validate_plot_spec",
    "load_plot_bundle",
    "save_plot_bundle",
    "generate_bundle_overview",
    "PLOT_SCHEMA_SPEC",
    # Layered bundle operations (v2.0)
    "save_layered_plot_bundle",
    "load_layered_plot_bundle",
    "merge_layered_bundle",
    "is_layered_bundle",
]

# EOF
