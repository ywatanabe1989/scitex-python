#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 10:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/styles/_plot_defaults.py

"""Pre-processing default kwargs for plot methods.

This module centralizes all default styling applied BEFORE matplotlib
methods are called. Each function modifies kwargs in-place.
"""

from scitex.plt.utils import mm_to_pt


# ============================================================================
# Constants
# ============================================================================
DEFAULT_LINE_WIDTH_MM = 0.2
DEFAULT_MARKER_SIZE_MM = 0.8
DEFAULT_CAP_SIZE_MM = 0.8
DEFAULT_FILL_ALPHA = 1.0  # Solid fill for publication figures


# ============================================================================
# Pre-processing functions
# ============================================================================
def apply_plot_defaults(method_name, kwargs, id_value=None, ax=None):
    """Apply default kwargs for a plot method before calling matplotlib.

    Args:
        method_name: Name of the matplotlib method being called
        kwargs: Keyword arguments dict (modified in-place)
        id_value: Optional id passed to the method
        ax: The matplotlib axes (for methods needing axis setup)

    Returns:
        Modified kwargs dict
    """
    # Dispatch to method-specific defaults
    if method_name == 'plot':
        _apply_plot_line_defaults(kwargs, id_value)
    elif method_name in ('bar', 'barh'):
        _apply_bar_defaults(kwargs)
    elif method_name == 'errorbar':
        _apply_errorbar_defaults(kwargs)
    elif method_name in ('fill_between', 'fill_betweenx'):
        _apply_fill_defaults(kwargs)
    elif method_name in ('quiver', 'streamplot'):
        _apply_vector_field_defaults(method_name, kwargs, ax)
    elif method_name == 'boxplot':
        _apply_boxplot_defaults(kwargs)
    elif method_name == 'violinplot':
        _apply_violinplot_defaults(kwargs)

    return kwargs


def _apply_plot_line_defaults(kwargs, id_value=None):
    """Apply defaults for ax.plot() method."""
    # Default line width: 0.2mm
    if 'linewidth' not in kwargs and 'lw' not in kwargs:
        kwargs['linewidth'] = mm_to_pt(DEFAULT_LINE_WIDTH_MM)

    # KDE-specific styling when id contains "kde"
    if id_value and 'kde' in str(id_value).lower():
        if 'linestyle' not in kwargs and 'ls' not in kwargs:
            kwargs['linestyle'] = '--'
        if 'color' not in kwargs and 'c' not in kwargs:
            kwargs['color'] = 'black'


def _apply_bar_defaults(kwargs):
    """Apply defaults for ax.bar() and ax.barh() methods."""
    # Set error bar line thickness to 0.2mm
    if 'error_kw' not in kwargs:
        kwargs['error_kw'] = {}
    if 'elinewidth' not in kwargs.get('error_kw', {}):
        kwargs['error_kw']['elinewidth'] = mm_to_pt(DEFAULT_LINE_WIDTH_MM)
    if 'capthick' not in kwargs.get('error_kw', {}):
        kwargs['error_kw']['capthick'] = mm_to_pt(DEFAULT_LINE_WIDTH_MM)
    # Set a temporary capsize that will be adjusted in post-processing
    if 'capsize' not in kwargs:
        kwargs['capsize'] = 5  # Placeholder, adjusted later to 33% of bar width


def _apply_errorbar_defaults(kwargs):
    """Apply defaults for ax.errorbar() method."""
    if 'capsize' not in kwargs:
        kwargs['capsize'] = mm_to_pt(DEFAULT_CAP_SIZE_MM)
    if 'capthick' not in kwargs:
        kwargs['capthick'] = mm_to_pt(DEFAULT_LINE_WIDTH_MM)
    if 'elinewidth' not in kwargs:
        kwargs['elinewidth'] = mm_to_pt(DEFAULT_LINE_WIDTH_MM)


def _apply_fill_defaults(kwargs):
    """Apply defaults for ax.fill_between() and ax.fill_betweenx() methods."""
    if 'alpha' not in kwargs:
        kwargs['alpha'] = DEFAULT_FILL_ALPHA  # Transparent to see overlapping data


def _apply_vector_field_defaults(method_name, kwargs, ax):
    """Apply defaults for ax.quiver() and ax.streamplot() methods."""
    # Set equal aspect ratio for proper vector display
    if ax is not None:
        ax.set_aspect('equal', adjustable='datalim')

    if method_name == 'streamplot':
        if 'arrowsize' not in kwargs:
            # arrowsize is a scaling factor; 0.8mm ~ 2.27pt, scale relative to default
            kwargs['arrowsize'] = mm_to_pt(DEFAULT_MARKER_SIZE_MM) / 3
        if 'linewidth' not in kwargs:
            kwargs['linewidth'] = mm_to_pt(DEFAULT_LINE_WIDTH_MM)

    elif method_name == 'quiver':
        if 'width' not in kwargs:
            kwargs['width'] = 0.003  # Narrow arrow shaft (axes fraction)
        if 'headwidth' not in kwargs:
            kwargs['headwidth'] = 3  # Head width relative to shaft
        if 'headlength' not in kwargs:
            kwargs['headlength'] = 4
        if 'headaxislength' not in kwargs:
            kwargs['headaxislength'] = 3.5


def _apply_boxplot_defaults(kwargs):
    """Apply defaults for ax.boxplot() method."""
    # Enable patch_artist for fillable boxes
    if 'patch_artist' not in kwargs:
        kwargs['patch_artist'] = True


def _apply_violinplot_defaults(kwargs):
    """Apply defaults for ax.violinplot() method."""
    # Default to showing boxplot overlay (can be disabled with boxplot=False)
    # Store the boxplot setting for post-processing, then remove from kwargs
    # so it doesn't get passed to matplotlib's violinplot
    if 'boxplot' not in kwargs:
        kwargs['boxplot'] = True  # Default: add boxplot overlay

    # Default to hiding extrema (min/max bars) when boxplot is shown
    if 'showextrema' not in kwargs:
        kwargs['showextrema'] = False


# EOF
