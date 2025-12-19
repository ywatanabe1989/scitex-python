#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/examples/fig/05_caption_generation.py

"""
Example 05: Caption Generation

Demonstrates:
- Setting figure title with number
- Auto-generating captions from panel descriptions
- Getting captions in different formats (text, LaTeX, Markdown)
"""

import numpy as np

import scitex as stx
from scitex import INJECTED
from scitex.fig import Figz


@stx.session(verbose=True, agg=True)
def main(plt=INJECTED, CONFIG=INJECTED, logger=INJECTED):
    """Demonstrate auto-caption generation."""
    logger.info("Example 05: Caption Generation")

    out_dir = CONFIG["SDIR_OUT"]

    fig = Figz(
        out_dir / "captioned_figure.zip.d",
        name="Experimental Results",
        size_mm={"width": 170, "height": 130},
    )

    # Add plots with descriptions
    x = np.linspace(0, 10, 100)

    # Plot A
    fig_a, ax_a = plt.subplots(figsize=(3, 2.5))
    ax_a.plot(x, np.sin(x))
    ax_a.set_title("Raw Signal")
    fig.add_element(
        "plot_A",
        "plot",
        fig_a,
        position={"x_mm": 5, "y_mm": 5},
        size={"width_mm": 80, "height_mm": 55},
    )
    plt.close(fig_a)

    # Plot B
    np.random.seed(42)
    fig_b, ax_b = plt.subplots(figsize=(3, 2.5))
    ax_b.scatter(np.random.randn(50), np.random.randn(50))
    ax_b.set_title("Feature Space")
    fig.add_element(
        "plot_B",
        "plot",
        fig_b,
        position={"x_mm": 88, "y_mm": 5},
        size={"width_mm": 80, "height_mm": 55},
    )
    plt.close(fig_b)

    # Plot C
    fig_c, ax_c = plt.subplots(figsize=(3, 2.5))
    ax_c.bar(["Ctrl", "Drug", "Wash"], [1.0, 2.5, 1.2])
    ax_c.set_title("Treatment Effect")
    fig.add_element(
        "plot_C",
        "plot",
        fig_c,
        position={"x_mm": 45, "y_mm": 65},
        size={"width_mm": 80, "height_mm": 55},
    )
    plt.close(fig_c)

    # === Set figure title ===
    logger.info("Setting figure title...")
    fig.set_figure_title("Experimental Results", prefix="Figure", number=1)

    # === Set panel descriptions ===
    logger.info("Setting panel descriptions...")
    fig.set_panel_info(
        "plot_A",
        panel_letter="A",
        description="Raw signal traces from electrode recordings",
    )
    fig.set_panel_info(
        "plot_B", panel_letter="B", description="Feature space visualization using PCA"
    )
    fig.set_panel_info(
        "plot_C",
        panel_letter="C",
        description="Mean response amplitude across conditions",
    )

    # === Generate captions ===
    logger.info("\n" + "=" * 60)
    logger.info("Generated Captions:")
    logger.info("=" * 60)

    # Plain text
    caption_text = fig.get_caption()
    logger.info(f"\n[Plain Text]\n{caption_text}")

    # LaTeX
    caption_latex = fig.get_caption_latex()
    logger.info(f"\n[LaTeX]\n{caption_latex}")

    # Markdown
    caption_md = fig.get_caption_markdown()
    logger.info(f"\n[Markdown]\n{caption_md}")

    # === Add visual caption to figure ===
    logger.info("\n" + "=" * 60)
    logger.info("Adding Visual Caption:")
    logger.info("=" * 60)

    visual_caption = fig.add_visual_caption(fontsize=9.0, margin_mm=5.0)
    logger.info(f"Visual caption added: {visual_caption[:50]}...")

    # Save
    fig.save()
    logger.info(f"\nSaved: {fig.path}")

    # Show theme.json content
    import json

    theme_path = fig.path / "theme.json"
    with open(theme_path) as f:
        theme = json.load(f)
    logger.info(f"\ntheme.json figure_title: {theme.get('figure_title')}")

    logger.success("Example 05 completed!")


if __name__ == "__main__":
    main()

# EOF
