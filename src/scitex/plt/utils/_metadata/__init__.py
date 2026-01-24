#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_metadata/__init__.py

"""
Figure metadata extraction package.

This package provides modular utilities for extracting metadata from
matplotlib figures, split from the original _collect_figure_metadata.py.

Modules:
- _rounding: Precision-controlled rounding utilities
- _detect: Plot type detection
- _csv: CSV column naming and hash computation
- _verification: CSV/JSON consistency verification
- _legend: Legend extraction
- _artists: Artist extraction (lines, collections, patches, images, text)
"""

# Rounding utilities
# Artist extraction
from ._artists import _extract_artists, _extract_traces

# CSV utilities
from ._csv import (
    _compute_csv_hash,
    _compute_csv_hash_from_df,
    _extract_csv_columns_from_history,
    _get_csv_column_names,
    _get_csv_columns_for_method,
    _get_csv_columns_for_method_with_index,
)

# Plot type detection
from ._detect import _detect_plot_type

# Legend extraction
from ._legend import _extract_legend_info
from ._rounding import (
    PRECISION,
    FixedFloat,
    _round_dict,
    _round_list,
    _round_value,
)

# Verification
from ._verification import (
    assert_csv_json_consistency,
    verify_csv_json_consistency,
)

__all__ = [
    # Rounding
    "PRECISION",
    "FixedFloat",
    "_round_value",
    "_round_list",
    "_round_dict",
    # Detection
    "_detect_plot_type",
    # CSV
    "_get_csv_column_names",
    "_extract_csv_columns_from_history",
    "_get_csv_columns_for_method_with_index",
    "_get_csv_columns_for_method",
    "_compute_csv_hash_from_df",
    "_compute_csv_hash",
    # Verification
    "assert_csv_json_consistency",
    "verify_csv_json_consistency",
    # Legend
    "_extract_legend_info",
    # Artists
    "_extract_artists",
    "_extract_traces",
]


# EOF
