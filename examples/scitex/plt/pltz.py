#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/bundles/pltz_bundle.py

"""
Demonstrates .pltz bundle creation and loading.

.pltz bundles contain:
- plot.json: Plot specification (axes, styles, annotations, theme)
- plot.csv: Raw data (immutable)
- plot.png/svg/pdf: Exports
- plot_hitmap.png/svg: Element selection maps
- overview.png: Bundle preview

Features demonstrated:
- Bundle creation, validation, loading
- DataFrame embedding
- Dark/light theme modes
- mm-based axis dimensions
"""

import numpy as np
import pandas as pd

import scitex as stx
import scitex.io as sio
from scitex.dev.plt import plot_histogram, plot_multi_line, plot_scatter_sizes
from scitex.io._bundle import validate_bundle


@stx.session
def main(
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Demonstrates .pltz bundle functionality."""
    logger.info("Starting .pltz bundle demo")
    sdir = CONFIG["SDIR_RUN"]
    rng = rng_manager("pltz_demo")

    # 1. Basic plot bundle
    logger.info("Creating basic plot bundle")
    fig, ax = plot_multi_line(plt, rng)
    sio.save(fig, sdir / "multi_line.pltz.d")
    plt.close(fig)

    result = validate_bundle(sdir / "multi_line.pltz.d")
    logger.info(f"Bundle valid: {result['valid']}, type: {result['bundle_type']}")

    # Load and verify
    loaded_fig, loaded_ax, _ = sio.load(sdir / "multi_line.pltz.d")
    plt.close(loaded_fig.figure)
    logger.success("Basic bundle created and loaded")

    # 2. Bundle with embedded DataFrame
    logger.info("Creating bundle with DataFrame")
    df = pd.DataFrame({
        "time": np.arange(10),
        "value_a": rng.standard_normal(10),
        "value_b": rng.standard_normal(10) + 1,
    })
    fig, ax = plt.subplots()
    ax.plot(df["time"], df["value_a"], "o-", label="Group A")
    ax.plot(df["time"], df["value_b"], "s-", label="Group B")
    ax.set_xyt("Time", "Value", "Time Series with Data")
    ax.legend()
    sio.save(fig, sdir / "with_data.pltz.d", data=df)
    plt.close(fig)

    _, _, loaded_df = sio.load(sdir / "with_data.pltz.d")
    assert loaded_df.shape == df.shape
    logger.success("DataFrame bundle created")

    # 3. Scatter plot bundle
    logger.info("Creating scatter bundle")
    fig, ax = plot_scatter_sizes(plt, rng)
    sio.save(fig, sdir / "scatter.pltz.d")
    plt.close(fig)

    # 4. Histogram bundle
    logger.info("Creating histogram bundle")
    fig, ax = plot_histogram(plt, rng)
    sio.save(fig, sdir / "histogram.pltz.d")
    plt.close(fig)

    # 5. Dark mode demo
    logger.info("Creating dark mode bundle")
    fig, ax = plt.subplots(theme="dark")
    x = np.linspace(0, 2 * np.pi, 100)
    ax.plot(x, np.cos(x), linewidth=2, label="cos(x)")
    ax.set_xyt("x", "y", "Dark Mode Demo")
    ax.legend()
    sio.save(fig, sdir / "dark_mode.pltz.d")
    plt.close(fig)

    # 6. Custom axis size (mm-based)
    logger.info("Creating custom size bundle")
    fig, ax = plt.subplots(axes_width_mm=60, axes_height_mm=40)
    ax.plot(x, np.sin(2 * x), linewidth=2)
    ax.set_xyt("x", "y", "Custom Size (60mm x 40mm)")
    sio.save(fig, sdir / "custom_size.pltz.d")
    plt.close(fig)

    # 7. ZIP archive
    logger.info("Creating ZIP archive")
    fig, ax = plt.subplots()
    ax.bar(["A", "B", "C"], [10, 20, 15])
    ax.set_xyt(None, "Count", "Bar Chart")
    sio.save(fig, sdir / "bar_chart.pltz", as_zip=True)
    plt.close(fig)
    logger.success("ZIP bundle created")

    logger.success("Demo completed")
    return 0


if __name__ == "__main__":
    main()

# EOF
