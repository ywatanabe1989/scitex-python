#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-08 17:40:30 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/fig/demo_canvas.py

# Timestamp: 2025-12-08
"""
Demo: scitex.fig - Canvas-Based Figure Composition

Demonstrates composing publication figures from multiple panels.

API:
    canvas = stx.fig.Canvas("fig1", width_mm=180, height_mm=150)
    canvas.add_panel("panel_a", "plot.png", position=(10, 10), size=(80, 60), label="A")
    stx.io.save(canvas, "/output/fig1.canvas")  # Auto-exports PNG/PDF/SVG
"""

from pathlib import Path
import scitex as stx
from scitex.dev.plt import (
    plot_multi_line,
    plot_bar_simple,
    plot_scatter_sizes,
    plot_histogram,
)


def create_sample_plots(output_dir: Path, plt, rng) -> dict:
    """Create sample plots for panels using scitex.dev.plt plotters."""
    paths = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    # Line plot (trig functions)
    fig, ax = plot_multi_line(plt, rng)
    paths["trig"] = output_dir / "plot_trig.png"
    stx.io.save(fig, paths["trig"])
    fig.close()

    # Bar chart
    fig, ax = plot_bar_simple(plt, rng)
    paths["bar"] = output_dir / "plot_bar.png"
    stx.io.save(fig, paths["bar"])
    fig.close()

    # Scatter plot
    fig, ax = plot_scatter_sizes(plt, rng)
    paths["scatter"] = output_dir / "plot_scatter.png"
    stx.io.save(fig, paths["scatter"])
    fig.close()

    # Histogram
    fig, ax = plot_histogram(plt, rng)
    paths["hist"] = output_dir / "plot_hist.png"
    stx.io.save(fig, paths["hist"])
    fig.close()

    return paths


def demo_basic_canvas(CONFIG, paths):
    """Basic canvas workflow: create Canvas object, add panels, save."""
    out = Path(CONFIG.SDIR_OUT)

    # Create Canvas object
    canvas = stx.fig.Canvas("fig1_demo", width_mm=180, height_mm=150)

    # Add panels
    canvas.add_panel(
        "panel_a", paths["trig"], position=(10, 10), size=(80, 60), label="A"
    )
    canvas.add_panel(
        "panel_b", paths["bar"], position=(100, 10), size=(80, 60), label="B"
    )
    canvas.add_panel(
        "panel_c",
        paths["scatter"],
        position=(10, 80),
        size=(80, 60),
        label="C",
    )
    canvas.add_panel(
        "panel_d", paths["hist"], position=(100, 80), size=(80, 60), label="D"
    )

    # Save (auto-exports PNG/PDF/SVG)
    stx.io.save(canvas, out / "fig1_demo.canvas")


def demo_panel_transforms(CONFIG, paths):
    """Panel transformations: rotation, opacity, flip."""
    out = Path(CONFIG.SDIR_OUT)

    canvas = stx.fig.Canvas("fig2_transforms", width_mm=180, height_mm=130)

    canvas.add_panel(
        "normal",
        paths["trig"],
        position=(10, 10),
        size=(70, 50),
        label="Normal",
    )
    canvas.add_panel(
        "rotated",
        paths["bar"],
        position=(100, 10),
        size=(70, 50),
        label="Rotated",
        rotation_deg=10,
    )
    canvas.add_panel(
        "transparent",
        paths["scatter"],
        position=(10, 70),
        size=(70, 50),
        label="Transparent",
        opacity=0.5,
    )
    canvas.add_panel(
        "flipped",
        paths["hist"],
        position=(100, 70),
        size=(70, 50),
        label="Flipped",
        flip_h=True,
    )

    stx.io.save(canvas, out / "fig2_transforms.canvas")


def demo_update_panel(CONFIG, paths):
    """Update panel properties using Canvas object."""
    out = Path(CONFIG.SDIR_OUT)

    canvas = stx.fig.Canvas("fig3_update", width_mm=180, height_mm=130)
    canvas.add_panel(
        "panel_a", paths["trig"], position=(10, 10), size=(80, 60), label="A"
    )

    # Update panel properties
    canvas.update_panel(
        "panel_a",
        {
            "position": {"x_mm": 50, "y_mm": 50},
            "opacity": 0.8,
            "border": {"visible": True, "color": "#FF0000", "width_mm": 0.5},
        },
    )

    stx.io.save(canvas, out / "fig3_update.canvas")


def demo_symlink_canvas(CONFIG, paths):
    """Save canvas with symlinks (default, bundle=False)."""
    out = Path(CONFIG.SDIR_OUT)

    canvas = stx.fig.Canvas("fig4_symlink", width_mm=180, height_mm=120)
    canvas.add_panel(
        "panel_a", paths["trig"], position=(10, 10), size=(160, 100), label="A"
    )

    # Save with symlinks (default) - panels directory contains symlinks to source
    canvas.save(out / "fig4_symlink.canvas")  # bundle=False is default


def demo_bundle_canvas(CONFIG, paths):
    """Save canvas with bundled files (bundle=True)."""
    out = Path(CONFIG.SDIR_OUT)

    canvas = stx.fig.Canvas("fig5_bundle", width_mm=180, height_mm=120)
    canvas.add_panel(
        "panel_a", paths["trig"], position=(10, 10), size=(160, 100), label="A"
    )

    # Save with bundled files - panels directory contains actual copies
    # This makes the canvas portable (can be moved/shared without source files)
    canvas.save(out / "fig5_bundle.canvas", bundle=True)


def demo_canvas_caption(CONFIG, paths):
    """Add figure caption (scientific legend) to canvas."""
    out = Path(CONFIG.SDIR_OUT)

    canvas = stx.fig.Canvas("fig6_caption", width_mm=180, height_mm=150)

    # Add panels
    canvas.add_panel(
        "panel_a", paths["trig"], position=(10, 10), size=(75, 50), label="A"
    )
    canvas.add_panel(
        "panel_b", paths["bar"], position=(95, 10), size=(75, 50), label="B"
    )

    # Add figure caption (rendered below figure)
    canvas.set_caption(
        "Figure 1. Demonstration of canvas composition with scientific caption. "
        "(A) Trigonometric functions showing sine and cosine waves over one period. "
        "(B) Bar chart comparing values across five categories."
    )

    canvas.save(out / "fig6_caption.canvas")


def demo_copy_canvas(CONFIG, paths):
    """Copy canvas using stx.io.save/load."""
    out = Path(CONFIG.SDIR_OUT)

    # Create and save original canvas
    canvas = stx.fig.Canvas("fig7_original", width_mm=180, height_mm=120)
    canvas.add_panel(
        "panel_a", paths["trig"], position=(10, 10), size=(160, 100), label="A"
    )
    canvas.save(out / "fig7_original.canvas")

    # Load and save to new location (copies panels + exports figures)
    loaded = stx.io.load(out / "fig7_original.canvas")
    loaded.save(out / "fig7_copy.canvas")


@stx.session
def main(
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Run all canvas demos."""
    out = Path(CONFIG.SDIR_OUT)
    rng = rng_manager("canvas_demo")
    paths = create_sample_plots(out / "plots", plt, rng)

    demo_basic_canvas(CONFIG, paths)
    demo_panel_transforms(CONFIG, paths)
    demo_update_panel(CONFIG, paths)
    demo_symlink_canvas(CONFIG, paths)
    demo_bundle_canvas(CONFIG, paths)
    demo_canvas_caption(CONFIG, paths)
    demo_copy_canvas(CONFIG, paths)


if __name__ == "__main__":
    main()

# EOF
