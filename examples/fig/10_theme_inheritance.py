#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/examples/fig/10_theme_inheritance.py

"""
Example 10: Theme Inheritance

Demonstrates:
- Figure-level theme applies to all child plots
- Child plots can override specific theme fields
- Resolved theme = parent merged with child overrides
"""

import json

import numpy as np

import scitex as stx
from scitex import INJECTED
from scitex.fig import Figz


@stx.session(verbose=True, agg=True)
def main(plt=INJECTED, CONFIG=INJECTED, logger=INJECTED):
    """Demonstrate theme inheritance from figure to child plots."""
    logger.info("Example 10: Theme Inheritance Demo")

    out_dir = CONFIG["SDIR_OUT"]

    # Create figure with global theme
    fig = Figz(
        out_dir / "theme_inheritance.zip.d",
        name="Theme Inheritance Demo",
        size_mm={"width": 170, "height": 100},
    )

    # Generate data
    x = np.linspace(0, 10, 50)
    np.random.seed(42)

    # Panel A: Inherits figure theme (no override)
    fig_a, ax_a = plt.subplots(figsize=(3, 2.5))
    ax_a.plot(x, np.sin(x), linewidth=1.5)
    ax_a.set_title("Panel A (inherits)")
    fig.add_element(
        "plot_A",
        "plot",
        fig_a,
        position={"x_mm": 5, "y_mm": 5},
        size={"width_mm": 80, "height_mm": 45},
    )
    plt.close(fig_a)

    # Panel B: Overrides trace color
    fig_b, ax_b = plt.subplots(figsize=(3, 2.5))
    ax_b.plot(x, np.cos(x), linewidth=1.5, color="red")
    ax_b.set_title("Panel B (color override)")
    fig.add_element(
        "plot_B",
        "plot",
        fig_b,
        position={"x_mm": 88, "y_mm": 5},
        size={"width_mm": 80, "height_mm": 45},
    )
    plt.close(fig_b)

    # Panel C: Overrides multiple fields
    fig_c, ax_c = plt.subplots(figsize=(3, 2.5))
    ax_c.plot(x, x**0.5, linewidth=2.0, color="green", linestyle="--")
    ax_c.set_title("Panel C (multi override)")
    fig.add_element(
        "plot_C",
        "plot",
        fig_c,
        position={"x_mm": 5, "y_mm": 52},
        size={"width_mm": 80, "height_mm": 45},
    )
    plt.close(fig_c)

    # Panel D: Uses dashed lines
    fig_d, ax_d = plt.subplots(figsize=(3, 2.5))
    ax_d.bar(["X", "Y", "Z"], [3, 7, 5])
    ax_d.set_title("Panel D (inherits)")
    fig.add_element(
        "plot_D",
        "plot",
        fig_d,
        position={"x_mm": 88, "y_mm": 52},
        size={"width_mm": 80, "height_mm": 45},
    )
    plt.close(fig_d)

    # Set panel info
    fig.set_panel_info("plot_A", panel_letter="A", description="Inherits all theme")
    fig.set_panel_info("plot_B", panel_letter="B", description="Overrides trace color")
    fig.set_panel_info("plot_C", panel_letter="C", description="Overrides multiple")
    fig.set_panel_info("plot_D", panel_letter="D", description="Inherits all theme")

    # Save
    fig.save()
    logger.info(f"Saved: {fig.path}")

    # === Theme Inheritance Demo ===
    logger.info("\n" + "=" * 60)
    logger.info("THEME INHERITANCE ANALYSIS")
    logger.info("=" * 60)

    # Load and show figure theme
    theme_path = fig.path / "theme.json"
    with open(theme_path) as f:
        figure_theme = json.load(f)

    logger.info("\nFigure-level theme:")
    logger.info(f"  colors.mode: {figure_theme.get('colors', {}).get('mode')}")
    logger.info(
        f"  typography.family: {figure_theme.get('typography', {}).get('family')}"
    )
    logger.info(f"  grid: {figure_theme.get('grid')}")

    # Show resolved theme per child
    logger.info("\nResolved theme per panel:")
    logger.info("-" * 40)

    children_dir = fig.path / "children"
    if children_dir.exists():
        for child in sorted(children_dir.iterdir()):
            if child.is_dir() or child.suffix in (".zip", ".stx"):
                child_id = child.stem.replace(".zip", "").replace(".stx", "")
                logger.info(f"\n{child_id}:")

                # Check for child theme override
                if child.is_dir():
                    child_theme_path = child / "theme.json"
                else:
                    child_theme_path = None

                if child_theme_path and child_theme_path.exists():
                    with open(child_theme_path) as f:
                        child_theme = json.load(f)
                    logger.info("  Has local theme.json (overrides)")
                    # Show what's overridden
                    if "traces" in child_theme and child_theme["traces"]:
                        logger.info(f"  traces: {child_theme['traces']}")
                else:
                    logger.info("  Inherits from figure (no local override)")

    # Show inheritance rules
    logger.info("\n" + "-" * 40)
    logger.info("Inheritance Rules:")
    logger.info("  1. Figure theme.json is the default")
    logger.info("  2. Child theme.json overrides specific fields")
    logger.info("  3. Unspecified fields inherit from parent")
    logger.info("  4. Merge: child_theme = {**parent, **child_overrides}")

    logger.info("\n" + "=" * 60)
    logger.success("Example 10 completed!")


if __name__ == "__main__":
    main()

# EOF
