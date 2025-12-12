#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-12 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/bundles/example_figz_bundle.py

"""
Demonstrates .figz bundle creation and loading.

.figz bundles contain:
- figure.json: Figure specification (layout, panels, styles)
- figure.png/svg/pdf: Exported figure
- Nested .pltz bundles for each panel

Hierarchy:
  Figure (.figz) = Publication Figure (e.g., "Figure 1")
  └── Panel(s) (A, B, C...)
      └── Plot(s) (.pltz)
"""

# Imports
import numpy as np

import scitex as stx
import scitex.fig as sfig
import scitex.plt as splt
from scitex.io._bundle import validate_bundle


# Functions and Classes
def plot_time_course(plt):
    """Create time course plot for Panel A."""
    fig, ax = plt.subplots(figsize=(6, 4))
    t = np.linspace(0, 10, 100)
    ax.plot(t, np.sin(t), "b-", label="Control")
    ax.plot(t, np.sin(t + 0.5) * 0.8, "r-", label="Treatment")
    ax.set_xyt("Time (s)", "Response", "Time Course")
    ax.legend()
    return fig, ax


def plot_bar_comparison(plt):
    """Create bar chart for Panel B."""
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = ["Control", "Treatment"]
    values = [5.2, 8.1]
    errors = [0.5, 0.7]
    ax.bar(groups, values, yerr=errors, capsize=5, color=["#3498db", "#e74c3c"])
    ax.set_xyt(None, "Mean Response", "Group Comparison")
    return fig, ax


def plot_scatter_correlation(plt, rng):
    """Create scatter plot for Panel C."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = rng.standard_normal(50)
    y = 0.8 * x + rng.standard_normal(50) * 0.3
    ax.scatter(x, y, alpha=0.6)
    ax.set_xyt("Variable X", "Variable Y", "Correlation")
    return fig, ax


@stx.session
def main(
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Demonstrates .figz bundle functionality."""
    logger.info("Starting .figz bundle demo")

    sdir = CONFIG["SDIR_RUN"]

    # 1. Create individual .pltz panels
    logger.info("Creating panel plots")

    # Panel A: Time series
    fig_a, ax_a = plot_time_course(plt)
    splt.save_pltz(fig_a, sdir / "panel_a.pltz.d")
    plt.close(fig_a)

    # Panel B: Bar chart
    fig_b, ax_b = plot_bar_comparison(plt)
    splt.save_pltz(fig_b, sdir / "panel_b.pltz.d")
    plt.close(fig_b)

    # Panel C: Scatter plot
    fig_c, ax_c = plot_scatter_correlation(plt, rng_manager("scatter"))
    splt.save_pltz(fig_c, sdir / "panel_c.pltz.d")
    plt.close(fig_c)

    logger.success("Panel plots created")

    # 2. Create .figz bundle from panels
    logger.info("Creating .figz publication figure")
    panels = {
        "A": str(sdir / "panel_a.pltz.d"),
        "B": str(sdir / "panel_b.pltz.d"),
        "C": str(sdir / "panel_c.pltz.d"),
    }

    sfig.save_figz(panels, sdir / "Figure1.figz.d")

    # Validate bundle
    result = validate_bundle(sdir / "Figure1.figz.d")
    logger.info(f"Bundle valid: {result['valid']}, type: {result['bundle_type']}")

    # Load and verify
    loaded = sfig.load_figz(sdir / "Figure1.figz.d")
    logger.info(f"Loaded figure with {len(loaded['spec']['panels'])} panels")

    # Close loaded figures
    for panel_id, panel_data in loaded.get("panels", {}).items():
        if isinstance(panel_data, tuple) and panel_data[0] is not None:
            fig_wrapper = panel_data[0]
            if hasattr(fig_wrapper, "figure"):
                plt.close(fig_wrapper.figure)

    logger.success("Figure bundle created and verified")

    # 3. Save as ZIP archive (default for .figz)
    logger.info("Creating ZIP archive")
    sfig.save_figz(panels, sdir / "Figure1.figz")  # as_zip=True by default
    logger.success("ZIP bundle created")

    # 4. Custom figure specification
    logger.info("Creating figure with custom specification")
    custom_spec = {
        "schema": {"name": "scitex.fig.figure", "version": "1.0.0"},
        "figure": {
            "id": "fig2",
            "title": "Neural Response Analysis",
            "caption": "Comparison of neural responses between control and treatment groups.",
            "styles": {
                "size": {"width_mm": 180, "height_mm": 90},
                "background": "#ffffff",
            },
        },
        "panels": [
            {
                "id": "A",
                "label": "A",
                "caption": "Time course of response",
                "position": {"x_mm": 5, "y_mm": 5},
                "size": {"width_mm": 55, "height_mm": 40},
            },
            {
                "id": "B",
                "label": "B",
                "caption": "Mean response comparison",
                "position": {"x_mm": 65, "y_mm": 5},
                "size": {"width_mm": 55, "height_mm": 40},
            },
            {
                "id": "C",
                "label": "C",
                "caption": "Correlation analysis",
                "position": {"x_mm": 125, "y_mm": 5},
                "size": {"width_mm": 55, "height_mm": 40},
            },
        ],
    }

    sfig.save_figz(panels, sdir / "Figure2.figz.d", spec=custom_spec)
    logger.success("Custom specification figure created")

    logger.success("Demo completed")
    return 0


if __name__ == "__main__":
    main()

# EOF
