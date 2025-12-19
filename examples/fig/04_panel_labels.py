#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/examples/fig/04_panel_labels.py

"""
Example 04: Panel Labels and Descriptions

Demonstrates:
- Assigning panel letters (A, B, C, ...)
- Setting panel descriptions for caption generation
- Auto-assigning panel letters with different styles
"""

import numpy as np

import scitex as stx
from scitex import INJECTED
from scitex.dev.plt import plot_stx_line, plot_stx_scatter
from scitex.fig import Figz


@stx.session(verbose=True, agg=True)
def main(plt=INJECTED, CONFIG=INJECTED, logger=INJECTED):
    """Demonstrate panel label assignment."""
    logger.info("Example 04: Panel Labels and Descriptions")

    out_dir = CONFIG["SDIR_OUT"]

    fig = Figz(
        out_dir / "labeled_panels.zip.d",
        name="Analysis Results",
        size_mm={"width": 170, "height": 70},
    )

    # Create random number generator
    rng = np.random.default_rng(42)

    # Create and add plots
    fig_a, ax_a = plot_stx_line(plt, rng)
    ax_a.set_title("Time Series")
    fig.add_element(
        "plot_A",
        "plot",
        fig_a,
        position={"x_mm": 5, "y_mm": 5},
        size={"width_mm": 80, "height_mm": 60},
    )
    plt.close(fig_a)

    fig_b, ax_b = plot_stx_scatter(plt, rng)
    ax_b.set_title("Correlation")
    fig.add_element(
        "plot_B",
        "plot",
        fig_b,
        position={"x_mm": 88, "y_mm": 5},
        size={"width_mm": 80, "height_mm": 60},
    )
    plt.close(fig_b)

    # === Method 1: Manual panel info ===
    logger.info("Setting panel info manually...")
    fig.set_panel_info(
        "plot_A", panel_letter="A", description="Time-series analysis of signal"
    )
    fig.set_panel_info(
        "plot_B", panel_letter="B", description="Correlation between variables"
    )

    # Verify
    info_a = fig.get_panel_info("plot_A")
    info_b = fig.get_panel_info("plot_B")
    logger.info(f"Panel A: {info_a}")
    logger.info(f"Panel B: {info_b}")

    # === Method 2: Auto-assign letters ===
    logger.info("\nAuto-assigning panel letters...")

    # Uppercase (default)
    fig.auto_assign_panel_letters(style="uppercase")
    logger.info("uppercase: A, B, C, ...")

    # Show other styles (just for demo)
    from scitex.schema import PanelLabels

    labels = PanelLabels(style="lowercase")
    logger.info(f"lowercase: {labels.format_letter(0)}, {labels.format_letter(1)}")

    labels = PanelLabels(style="roman")
    logger.info(f"roman: {labels.format_letter(0)}, {labels.format_letter(1)}")

    labels = PanelLabels(style="Roman")
    logger.info(f"Roman: {labels.format_letter(0)}, {labels.format_letter(1)}")

    # Save
    fig.save()
    logger.info(f"\nSaved: {fig.path}")

    logger.success("Example 04 completed!")


if __name__ == "__main__":
    main()

# EOF
