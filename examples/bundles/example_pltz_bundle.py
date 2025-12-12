#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-12 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/bundles/example_pltz_bundle.py

"""
Demonstrates .pltz bundle creation and loading.

.pltz bundles contain:
- plot.json: Plot specification (axes, styles, annotations, theme)
- plot.csv: Raw data (immutable)
- plot.png: Raster export
- plot.svg/pdf: Optional vector exports

Features demonstrated:
- mm-based axis dimensions (default: 40mm x 28mm)
- Dark/light theme modes
- Bundle validation and loading
- DataFrame embedding
"""

# Imports
import numpy as np

import scitex as stx
import scitex.io as sio
import scitex.plt as splt
from scitex.io._bundle import validate_bundle


# Functions and Classes
def plot_sine_wave(plt, x, y):
    """Create sine wave plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, "b-", linewidth=2, label="sin(x)")
    ax.set_xyt("x (radians)", "y", "Sine Wave")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_time_series(plt, df):
    """Create time series plot from DataFrame."""
    fig, ax = plt.subplots()
    ax.plot(df["time"], df["value_a"], "o-", label="Group A")
    ax.plot(df["time"], df["value_b"], "s-", label="Group B")
    ax.set_xyt("Time", "Value", "Time Series")
    ax.legend()
    return fig, ax


def plot_bar_chart(plt):
    """Create simple bar chart."""
    fig, ax = plt.subplots()
    ax.bar(["A", "B", "C"], [10, 20, 15])
    ax.set_xyt(None, "Count", "Bar Chart")
    return fig, ax


def plot_scatter(plt):
    """Create scatter plot."""
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [1, 4, 9])
    ax.set_xyt("X", "Y", "Scatter Plot")
    return fig, ax


def plot_dark_mode(plt, x, y):
    """Create plot with dark theme for eye-friendly visualization."""
    # Use theme='dark' for eye-friendly colors on dark backgrounds
    fig, ax = plt.subplots(theme="dark")
    ax.plot(x, y, linewidth=2, label="cos(x)")
    ax.set_xyt("x (radians)", "y", "Dark Mode Demo")
    ax.legend()
    return fig, ax


def plot_custom_axes_size(plt, x, y):
    """Create plot with custom mm-based axis dimensions."""
    # Specify exact axis dimensions in mm (default: 40mm x 28mm)
    fig, ax = plt.subplots(axes_width_mm=60, axes_height_mm=40)
    ax.plot(x, y, linewidth=2, label="sin(2x)")
    ax.set_xyt("x (radians)", "y", "Custom Size (60mm x 40mm)")
    ax.legend()
    return fig, ax


@stx.session
def main(
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
):
    """Demonstrates .pltz bundle functionality."""
    sdir = CONFIG["SDIR_RUN"]

    # 1. Create and save sine wave as directory bundle
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    fig, ax = plot_sine_wave(plt, x, y)
    sio.save(fig, sdir / "sine_wave.pltz.d")
    plt.close(fig)

    # Validate and load bundle
    result = validate_bundle(sdir / "sine_wave.pltz.d")
    assert result['valid'], "Bundle validation failed"
    loaded_fig, loaded_ax, loaded_data = sio.load(sdir / "sine_wave.pltz.d")
    plt.close(loaded_fig.figure)

    # 2. Create bundle with embedded DataFrame
    import pandas as pd
    df = pd.DataFrame(
        {
            "time": np.arange(10),
            "value_a": rng_manager("a").standard_normal(10),
            "value_b": rng_manager("b").standard_normal(10) + 1,
        }
    )
    fig, ax = plot_time_series(plt, df)
    sio.save(fig, sdir / "with_data.pltz.d", data=df)
    plt.close(fig)

    # Verify data was saved
    _, _, loaded_df = sio.load(sdir / "with_data.pltz.d")
    assert loaded_df.shape == df.shape, "DataFrame shape mismatch"

    # 3. Save as ZIP archive
    fig, ax = plot_bar_chart(plt)
    sio.save(fig, sdir / "bar_chart.pltz", as_zip=True)
    plt.close(fig)

    # 4. Save via scitex.io.save (auto-detection)
    fig, ax = plot_scatter(plt)
    sio.save(fig, sdir / "scatter.pltz.d")
    plt.close(fig)

    # 5. Dark mode demo - eye-friendly colors for dark backgrounds
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.cos(x)
    fig, ax = plot_dark_mode(plt, x, y)
    sio.save(fig, sdir / "dark_mode.pltz.d")
    plt.close(fig)

    # 6. Custom axis size demo - mm-based precision
    y2 = np.sin(2 * x)
    fig, ax = plot_custom_axes_size(plt, x, y2)
    sio.save(fig, sdir / "custom_size.pltz.d")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    main()

# EOF
