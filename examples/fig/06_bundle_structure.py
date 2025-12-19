#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/examples/fig/06_bundle_structure.py

"""
Example 06: Bundle Structure Inspection

Demonstrates:
- Understanding .zip bundle directory structure
- Inspecting spec.json, theme.json, encoding.json
- Viewing cache files (geometry_px.json, hitmap)
- Understanding canonical vs cache vs export files
"""

import json

import numpy as np

import scitex as stx
from scitex import INJECTED
from scitex.dev.plt import plot_stx_line
from scitex.fig import Figz


@stx.session(verbose=True, agg=True)
def main(plt=INJECTED, CONFIG=INJECTED, logger=INJECTED):
    """Inspect .zip bundle structure."""
    logger.info("Example 06: Bundle Structure Inspection")

    out_dir = CONFIG["SDIR_OUT"]

    # Create a figure with content
    fig = Figz(
        out_dir / "inspectable.zip.d",
        name="Inspection Demo",
        size_mm={"width": 170, "height": 80},
    )

    # Create random number generator
    rng = np.random.default_rng(42)

    # Add a plot
    fig_a, ax_a = plot_stx_line(plt, rng)
    ax_a.legend()
    fig.add_element(
        "plot_A",
        "plot",
        fig_a,
        position={"x_mm": 45, "y_mm": 5},
        size={"width_mm": 80, "height_mm": 70},
    )
    plt.close(fig_a)

    fig.set_figure_title("Demo Figure", number=1)
    fig.set_panel_info("plot_A", description="Sinusoidal signal")
    fig.save()

    logger.info(f"\nBundle path: {fig.path}")
    logger.info("=" * 60)

    # === Directory structure ===
    logger.info("\n📁 BUNDLE STRUCTURE:")
    for f in sorted(fig.path.rglob("*")):
        if f.is_file():
            rel = f.relative_to(fig.path)
            size = f.stat().st_size
            logger.info(f"  {rel} ({size} bytes)")

    # === Canonical files ===
    logger.info("\n📄 CANONICAL FILES (source of truth):")

    # spec.json
    with open(fig.path / "spec.json") as f:
        spec = json.load(f)
    logger.info("\nspec.json:")
    logger.info(f"  type: {spec.get('type')}")
    logger.info(f"  title: {spec.get('title')}")
    logger.info(f"  elements: {len(spec.get('elements', []))}")

    # encoding.json
    with open(fig.path / "encoding.json") as f:
        encoding = json.load(f)
    logger.info("\nencoding.json (data→visual mapping):")
    logger.info(f"  traces: {len(encoding.get('traces', []))}")

    # theme.json
    with open(fig.path / "theme.json") as f:
        theme = json.load(f)
    logger.info("\ntheme.json (aesthetics):")
    logger.info(f"  colors.mode: {theme.get('colors', {}).get('mode')}")
    logger.info(f"  figure_title: {theme.get('figure_title', {}).get('text')}")

    # === Cache files ===
    logger.info("\n🔄 CACHE FILES (regenerable):")

    cache_dir = fig.path / "cache"
    if cache_dir.exists():
        for f in cache_dir.iterdir():
            logger.info(f"  {f.name}")

    # === Export files ===
    logger.info("\n📤 EXPORT FILES (derived):")

    exports_dir = fig.path / "exports"
    if exports_dir.exists():
        for f in exports_dir.iterdir():
            logger.info(f"  {f.name}")

    # === Children ===
    logger.info("\n👶 CHILDREN (embedded bundles):")
    children_dir = fig.path / "children"
    if children_dir.exists():
        for f in children_dir.iterdir():
            logger.info(f"  {f.name}")

    logger.info("\n" + "=" * 60)
    logger.info("File categories:")
    logger.info("  CANONICAL: spec.json, encoding.json, theme.json, data/*")
    logger.info("  CACHE: cache/* (can delete and regenerate)")
    logger.info("  EXPORT: exports/* (derived outputs)")

    logger.success("Example 06 completed!")


if __name__ == "__main__":
    main()

# EOF
