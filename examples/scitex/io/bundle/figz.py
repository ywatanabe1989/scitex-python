#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/examples/scitex/io/bundle/figz.py
"""
Comprehensive FTS figure bundle demonstration with various plot types.

Demonstrates:
1. FTS bundle hierarchy (Figure → Panels → Plots)
2. Various matplotlib plot types
3. Bundle creation and loading using FTS

Uses PLOTTERS registry from scitex.dev.plt for all plot types.
"""

import scitex as stx
import scitex.io as sio
from scitex.fts import FTS
from scitex.dev.plt import PLOTTERS_STX, PLOTTERS_MPL


@stx.session
def main(
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Comprehensive FTS figure bundle demonstration."""
    logger.info("Starting FTS figure bundle demo with various plot types")

    sdir = CONFIG["SDIR_OUT"]
    rng = rng_manager("figz_demo")

    # -------------------------------------------------------------------------
    # Part 1: Create individual plot bundles with various plot types
    # -------------------------------------------------------------------------
    logger.info("Creating panel plots (FTS bundles)")

    # Select a variety of plot types from registries
    plot_configs = [
        (PLOTTERS_STX["stx_line"], "panel_line", "A"),
        (PLOTTERS_STX["stx_bar"], "panel_bar", "B"),
        (PLOTTERS_STX["stx_scatter"], "panel_scatter", "C"),
        (PLOTTERS_MPL["mpl_hist"], "panel_histogram", "D"),
        (PLOTTERS_STX["stx_errorbar"], "panel_errorbar", "E"),
        (PLOTTERS_STX["stx_boxplot"], "panel_boxplot", "F"),
    ]

    panels = {}
    for plot_func, name, panel_id in plot_configs:
        try:
            fig, ax = plot_func(plt, rng)
            panel_path = sdir / f"{name}.stx"
            sio.save(fig, panel_path, dpi=150)
            plt.close(fig)
            panels[panel_id] = str(panel_path)
            logger.info(f"  Created {name}.stx (Panel {panel_id})")
        except Exception as e:
            logger.warning(f"  Failed {name}: {e}")

    logger.success(f"Created {len(panels)} panel plots")

    # -------------------------------------------------------------------------
    # Part 2: Create composite figure bundle
    # -------------------------------------------------------------------------
    logger.info("Creating composite figure bundle (3 panels)")

    # Create a composite bundle using FTS
    figure_path = sdir / "Figure1.stx"
    bundle = FTS(figure_path, create=True, node_type="figure")
    bundle.node.title = "Composite Figure Demo"
    bundle.node.description = "Demonstration of FTS figure bundles"
    bundle.save()

    logger.success("Figure1.stx created")

    # -------------------------------------------------------------------------
    # Part 3: Create ZIP archive
    # -------------------------------------------------------------------------
    logger.info("Creating ZIP archive")
    zip_path = sdir / "Figure1.zip"
    bundle_zip = FTS(zip_path, create=True, node_type="figure")
    bundle_zip.node.title = "ZIP Archive Demo"
    bundle_zip.save()
    logger.success("ZIP bundle created")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("Bundle Demo Summary:")
    logger.info(f"  Output: {sdir}")
    logger.info(f"  Panels (.stx): {len(panels)}")
    logger.info("  Figure bundles: 2")
    logger.success("Demo completed")

    return 0


if __name__ == "__main__":
    main()

# EOF
