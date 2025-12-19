#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/examples/fig/02_adding_plots.py

"""
Example 02: Adding Plot Panels

Demonstrates:
- Adding matplotlib plots as elements
- Positioning plots in a grid layout
- Using ax.stx_* methods for plotting
"""

import numpy as np

import scitex as stx
from scitex import INJECTED
from scitex.fig import Figz


@stx.session(verbose=True, agg=True)
def main(plt=INJECTED, CONFIG=INJECTED, logger=INJECTED):
    """Create multi-panel figure with plots."""
    logger.info("Example 02: Adding Plot Panels")

    out_dir = CONFIG["SDIR_OUT"]

    # Create figure bundle (2x2 grid layout)
    fig = Figz(
        out_dir / "multi_panel.stx.d",
        name="Multi-Panel Figure",
        size_mm={"width": 170, "height": 130},
    )

    # Panel A: Line plot (top-left)
    x = np.linspace(0, 10, 100)
    fig_a, ax_a = plt.subplots(figsize=(3, 2.5))
    ax_a.plot(x, np.sin(x), label="sin(x)")
    ax_a.plot(x, np.cos(x), label="cos(x)")
    ax_a.legend()
    ax_a.set_title("Trigonometric Functions")

    fig.add_element(
        "plot_A",
        "plot",
        fig_a,
        position={"x_mm": 5, "y_mm": 5},
        size={"width_mm": 80, "height_mm": 60},
    )
    plt.close(fig_a)
    logger.info("Added plot_A (line)")

    # Panel B: Scatter plot (top-right)
    np.random.seed(42)
    fig_b, ax_b = plt.subplots(figsize=(3, 2.5))
    ax_b.scatter(np.random.randn(50), np.random.randn(50))
    ax_b.set_title("Random Scatter")

    fig.add_element(
        "plot_B",
        "plot",
        fig_b,
        position={"x_mm": 88, "y_mm": 5},
        size={"width_mm": 80, "height_mm": 60},
    )
    plt.close(fig_b)
    logger.info("Added plot_B (scatter)")

    # Panel C: Bar chart (bottom-left)
    fig_c, ax_c = plt.subplots(figsize=(3, 2.5))
    ax_c.bar(["A", "B", "C", "D"], [4, 7, 2, 8])
    ax_c.set_title("Category Values")

    fig.add_element(
        "plot_C",
        "plot",
        fig_c,
        position={"x_mm": 5, "y_mm": 68},
        size={"width_mm": 80, "height_mm": 60},
    )
    plt.close(fig_c)
    logger.info("Added plot_C (bar)")

    # Panel D: Box plot (bottom-right)
    fig_d, ax_d = plt.subplots(figsize=(3, 2.5))
    data = [np.random.randn(50) + i for i in range(4)]
    ax_d.boxplot(data)
    ax_d.set_title("Distribution Comparison")

    fig.add_element(
        "plot_D",
        "plot",
        fig_d,
        position={"x_mm": 88, "y_mm": 68},
        size={"width_mm": 80, "height_mm": 60},
    )
    plt.close(fig_d)
    logger.info("Added plot_D (box)")

    # Save
    fig.save()
    logger.info(f"Saved: {fig.path}")
    logger.info(f"Elements: {fig.list_element_ids()}")

    logger.success("Example 02 completed!")


if __name__ == "__main__":
    main()

# EOF
