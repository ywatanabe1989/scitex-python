#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/bundles/figz_bundle.py

"""
Comprehensive .figz bundle demonstration with various plot types.

Demonstrates:
1. .figz bundle hierarchy (Figure → Panels → Plots)
2. Various matplotlib plot types with hitmap support
3. Bundle validation and loading

Plot types: lines, bars, scatter, histogram, errorbar, boxplot, violin,
            contour, heatmap, fill_between, pie, annotations, step/stem
"""

import scitex as stx
import scitex.fig as sfig
import scitex.io as sio
from scitex.io._bundle import validate_bundle

from scitex.dev.plt import (
    plot_bar_grouped,
    plot_bar_simple,
    plot_boxplot,
    plot_contour,
    plot_errorbar,
    plot_fill_between,
    plot_heatmap,
    plot_histogram_multiple,
    plot_multi_line,
    plot_scatter_sizes,
    plot_violin,
)


@stx.session
def main(
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Comprehensive .figz bundle demonstration."""
    logger.info("Starting .figz bundle demo with various plot types")

    sdir = CONFIG["SDIR_RUN"]
    rng = rng_manager("figz_demo")

    # -------------------------------------------------------------------------
    # Part 1: Create individual .pltz panels with various plot types
    # -------------------------------------------------------------------------
    logger.info("Creating panel plots (.pltz bundles)")

    plot_configs = [
        (plot_multi_line, "panel_lines", "A"),
        (plot_bar_grouped, "panel_bars", "B"),
        (plot_scatter_sizes, "panel_scatter", "C"),
        (plot_histogram_multiple, "panel_histogram", "D"),
        (plot_errorbar, "panel_errorbar", "E"),
        (plot_boxplot, "panel_boxplot", "F"),
        (plot_contour, "panel_contour", "G"),
        (plot_heatmap, "panel_heatmap", "H"),
        (plot_fill_between, "panel_fill", "I"),
        (plot_bar_simple, "panel_bar_simple", "J"),
        (plot_violin, "panel_violin", "K"),
    ]

    panels = {}
    for plot_func, name, panel_id in plot_configs:
        try:
            fig, ax = plot_func(plt, rng)
            panel_path = sdir / f"{name}.pltz.d"
            sio.save(fig, panel_path, dpi=150)
            plt.close(fig)
            panels[panel_id] = str(panel_path)
            logger.info(f"  Created {name}.pltz.d (Panel {panel_id})")
        except Exception as e:
            logger.warning(f"  Failed {name}: {e}")

    logger.success(f"Created {len(panels)} panel plots")

    # -------------------------------------------------------------------------
    # Part 2: Create .figz bundle from panels (3-panel figure)
    # -------------------------------------------------------------------------
    logger.info("Creating .figz publication figure (3 panels)")

    figure1_panels = {k: panels[k] for k in ["A", "B", "C"] if k in panels}
    sfig.save_figz(figure1_panels, sdir / "Figure1.figz.d")

    result = validate_bundle(sdir / "Figure1.figz.d")
    logger.info(f"Figure1 valid: {result['valid']}, type: {result['bundle_type']}")

    loaded = sfig.load_figz(sdir / "Figure1.figz.d")
    logger.info(f"Loaded figure with {len(loaded['spec']['panels'])} panels")

    for panel_id, panel_data in loaded.get("panels", {}).items():
        if isinstance(panel_data, tuple) and panel_data[0] is not None:
            fig_wrapper = panel_data[0]
            if hasattr(fig_wrapper, "figure"):
                plt.close(fig_wrapper.figure)

    logger.success("Figure1.figz.d created and verified")

    # -------------------------------------------------------------------------
    # Part 3: Create larger figure with custom specification
    # -------------------------------------------------------------------------
    logger.info("Creating Figure2 with custom specification (6 panels)")

    figure2_panels = {k: panels[k] for k in ["D", "E", "F", "G", "H", "I"] if k in panels}

    custom_spec = {
        "schema": {"name": "scitex.fig.figure", "version": "1.0.0"},
        "figure": {
            "id": "fig2",
            "title": "Comprehensive Plot Type Demo",
            "caption": "Demonstration of various plot types with hitmap support.",
            "styles": {"size": {"width_mm": 180, "height_mm": 120}},
        },
        "panels": [{"id": pid, "label": pid} for pid in figure2_panels.keys()],
    }

    sfig.save_figz(figure2_panels, sdir / "Figure2.figz.d", spec=custom_spec)
    logger.success("Figure2.figz.d created with custom spec")

    # -------------------------------------------------------------------------
    # Part 4: Create ZIP archives
    # -------------------------------------------------------------------------
    logger.info("Creating ZIP archives")
    sfig.save_figz(figure1_panels, sdir / "Figure1.figz")
    sfig.save_figz(figure2_panels, sdir / "Figure2.figz")
    logger.success("ZIP bundles created")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("Bundle Demo Summary:")
    logger.info(f"  Output: {sdir}")
    logger.info(f"  Panels (.pltz): {len(panels)}")
    logger.info("  Figures (.figz): 2")
    logger.success("Demo completed")

    return 0


if __name__ == "__main__":
    main()

# EOF
