#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: _deprecated.py - Deprecated method aliases

"""Deprecated plot_ method aliases for backward compatibility."""

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)


def add_deprecated_aliases(cls):
    """Add deprecated plot_ method aliases to the given class.

    Parameters
    ----------
    cls : class
        The class to add deprecated aliases to (usually MatplotlibPlotMixin)
    """
    from scitex.decorators import deprecated

    deprecated_methods = [
        ("plot_image", "stx_image"),
        ("plot_kde", "stx_kde"),
        ("plot_conf_mat", "stx_conf_mat"),
        ("plot_rectangle", "stx_rectangle"),
        ("plot_fillv", "stx_fillv"),
        ("plot_box", "stx_box"),
        ("plot_raster", "stx_raster"),
        ("plot_ecdf", "stx_ecdf"),
        ("plot_joyplot", "stx_joyplot"),
        ("plot_line", "stx_line"),
        ("plot_scatter_hist", "stx_scatter_hist"),
        ("plot_heatmap", "stx_heatmap"),
        ("plot_violin", "stx_violin"),
        ("plot_mean_std", "stx_mean_std"),
        ("plot_mean_ci", "stx_mean_ci"),
        ("plot_median_iqr", "stx_median_iqr"),
        ("plot_shaded_line", "stx_shaded_line"),
    ]

    for old_name, new_name in deprecated_methods:
        def make_deprecated_method(target_name):
            @deprecated(reason=f"Use {target_name} instead")
            def method(self, *args, **kwargs):
                return getattr(self, target_name)(*args, **kwargs)
            return method

        setattr(cls, old_name, make_deprecated_method(new_name))


# EOF
