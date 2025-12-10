#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scitex/plt/utils/metadata/__init__.py

"""
Figure metadata collection package.

This package provides utilities to collect comprehensive metadata from matplotlib
figures and axes for embedding in saved images.

Public API
----------
collect_figure_metadata : function
    Main function to collect all metadata from a figure
collect_recipe_metadata : function
    Collect metadata with reconstruction recipe
assert_csv_json_consistency : function
    Assert CSV columns match JSON metadata
verify_csv_json_consistency : function
    Verify CSV-JSON consistency and return detailed results
"""

# Import public API from core module
from ._core import collect_figure_metadata
from ._data_linkage import (
    assert_csv_json_consistency,
    verify_csv_json_consistency,
    collect_recipe_metadata,
)

__all__ = [
    "collect_figure_metadata",
    "collect_recipe_metadata",
    "assert_csv_json_consistency",
    "verify_csv_json_consistency",
]
