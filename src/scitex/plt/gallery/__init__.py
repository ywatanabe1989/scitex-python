#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-18 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/gallery/__init__.py

"""
SciTeX Plot Gallery

Generate example plots with CSVs organized by visualization purpose.

Usage:
    import scitex as stx

    # Generate full gallery
    stx.plt.gallery.generate("./gallery")

    # Generate specific category
    stx.plt.gallery.generate("./gallery", category="line")

    # List available plots
    stx.plt.gallery.list()

    # Get plot spec and data for bundle creation
    spec = stx.plt.gallery.get_plot_spec("line", "plot")
    data = stx.plt.gallery.get_plot_data("line", "plot")
"""

from ._generate import generate
from ._registry import CATEGORIES, list_plots
from ._plots import PLOT_FUNCTIONS


def get_plot_spec(category: str, plot_name: str) -> dict:
    """Get spec dictionary for a gallery plot.

    Parameters
    ----------
    category : str
        Plot category (e.g., "line", "scatter", "categorical")
    plot_name : str
        Plot name within the category (e.g., "plot", "scatter", "bar")

    Returns
    -------
    dict
        Spec dictionary for the plot type.
    """
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Available: {list(CATEGORIES.keys())}")

    if plot_name not in CATEGORIES[category]["plots"]:
        raise ValueError(f"Unknown plot: {plot_name} in category {category}")

    # Build a minimal spec for this plot type
    return {
        "schema": {"name": "scitex.plt.plot", "version": "1.0.0"},
        "plot_type": plot_name,
        "category": category,
        "axes": {"xlabel": "", "ylabel": ""},
    }


def get_plot_data(category: str, plot_name: str):
    """Get sample data for a gallery plot as DataFrame.

    Parameters
    ----------
    category : str
        Plot category (e.g., "line", "scatter", "categorical")
    plot_name : str
        Plot name within the category (e.g., "plot", "scatter", "bar")

    Returns
    -------
    pandas.DataFrame or None
        Sample data for the plot, or None if plot doesn't use CSV data.
    """
    import io
    import tempfile
    from pathlib import Path

    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Available: {list(CATEGORIES.keys())}")

    if plot_name not in CATEGORIES[category]["plots"]:
        raise ValueError(f"Unknown plot: {plot_name} in category {category}")

    if plot_name not in PLOT_FUNCTIONS:
        return None

    try:
        import scitex as stx
        from scitex.plt.styles.presets import SCITEX_STYLE

        # Create a temporary figure
        style = SCITEX_STYLE.copy()
        style["figsize"] = (4, 3)
        fig, ax = stx.plt.subplots(**style)

        # Generate plot
        plot_func = PLOT_FUNCTIONS[plot_name]
        fig, ax = plot_func(fig, ax, stx)

        # Export to CSV and parse
        csv_buffer = io.StringIO()
        ax.export_as_csv(csv_buffer)
        csv_buffer.seek(0)

        import pandas as pd
        df = pd.read_csv(csv_buffer)

        # Close figure
        stx.plt.close(fig._fig_mpl if hasattr(fig, "_fig_mpl") else fig)

        return df

    except Exception as e:
        # Some plots may not have exportable CSV data
        import warnings
        warnings.warn(f"Could not get data for {category}/{plot_name}: {e}")
        return None


__all__ = ["generate", "list_plots", "CATEGORIES", "get_plot_spec", "get_plot_data"]

# EOF
