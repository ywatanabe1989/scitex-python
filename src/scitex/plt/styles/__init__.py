#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 10:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/styles/__init__.py

"""SciTeX plot styling module.

This module centralizes all plot-specific default styling, including:
- Pre-processing: Default kwargs applied before matplotlib method calls
- Post-processing: Styling applied after matplotlib method calls

Usage:
    from scitex.plt.styles import apply_plot_defaults, apply_plot_postprocess

    # In AxisWrapper.__getattr__ wrapper:
    apply_plot_defaults(method_name, kwargs, id_value, ax)
    result = orig_method(*args, **kwargs)
    apply_plot_postprocess(method_name, result, ax, kwargs)
"""

from ._plot_defaults import apply_plot_defaults
from ._plot_postprocess import apply_plot_postprocess

__all__ = [
    'apply_plot_defaults',
    'apply_plot_postprocess',
]


# EOF
