#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/examples/fig/13_layout_edit_bbox_norm.py

"""
Example 13: Layout Edit at Figure Level (bbox_norm editing)

Demonstrates:
- Canonical geometry (position in mm) is the real layout
- Pixel coordinates (geometry_px.json) are derived
- Programmatic layout adjustment
"""

import json
import shutil

import numpy as np

import scitex as stx
from scitex import INJECTED
from scitex.fig import Figz


@stx.session(verbose=True, agg=True)
def main(plt=INJECTED, CONFIG=INJECTED, logger=INJECTED):
    """Demonstrate layout editing via spec.json positions."""
    logger.info("Example 13: Layout Edit Demo")

    out_dir = CONFIG["SDIR_OUT"]
    bundle_path = out_dir / "layout_edit.zip.d"

    # === Create initial multi-panel figure ===
    logger.info("\n" + "=" * 60)
    logger.info("Creating initial multi-panel figure...")
    logger.info("=" * 60)

    fig = Figz(
        bundle_path, name="Layout Edit Demo", size_mm={"width": 170, "height": 70}
    )

    x = np.linspace(0, 10, 50)

    # Panel A
    fig_a, ax_a = plt.subplots(figsize=(3, 2))
    ax_a.plot(x, np.sin(x))
    ax_a.set_title("Panel A")
    fig.add_element(
        "plot_A",
        "plot",
        fig_a,
        position={"x_mm": 5, "y_mm": 5},
        size={"width_mm": 50, "height_mm": 60},
    )
    plt.close(fig_a)

    # Panel B
    fig_b, ax_b = plt.subplots(figsize=(3, 2))
    ax_b.plot(x, np.cos(x), color="red")
    ax_b.set_title("Panel B")
    fig.add_element(
        "plot_B",
        "plot",
        fig_b,
        position={"x_mm": 60, "y_mm": 5},  # Original position
        size={"width_mm": 50, "height_mm": 60},
    )
    plt.close(fig_b)

    # Panel C
    fig_c, ax_c = plt.subplots(figsize=(3, 2))
    ax_c.plot(x, x**0.5, color="green")
    ax_c.set_title("Panel C")
    fig.add_element(
        "plot_C",
        "plot",
        fig_c,
        position={"x_mm": 115, "y_mm": 5},
        size={"width_mm": 50, "height_mm": 60},
    )
    plt.close(fig_c)

    fig.set_panel_info("plot_A", panel_letter="A")
    fig.set_panel_info("plot_B", panel_letter="B")
    fig.set_panel_info("plot_C", panel_letter="C")
    fig.save()

    # Record initial positions
    logger.info("\nInitial positions (from spec.json):")
    for elem in fig.elements:
        if elem.get("type") == "plot":
            pos = elem.get("position", {})
            logger.info(f"  {elem['id']}: x={pos.get('x_mm')}mm, y={pos.get('y_mm')}mm")

    # Load initial geometry_px
    geometry_path = bundle_path / "cache" / "geometry_px.json"
    if geometry_path.exists():
        with open(geometry_path) as f:
            initial_geometry = json.load(f)
        logger.info("\nInitial geometry_px (derived):")
        for elem in initial_geometry.get("elements", []):
            bbox = elem.get("bbox_px", {})
            logger.info(f"  {elem['id']}: x={bbox.get('x', 0):.0f}px")

    # === Edit layout: Shift Panel B right ===
    logger.info("\n" + "=" * 60)
    logger.info("Shifting Panel B right by +10mm...")
    logger.info("=" * 60)

    # Modify spec.json directly
    spec_path = bundle_path / "spec.json"
    with open(spec_path) as f:
        spec = json.load(f)

    # Find and modify Panel B position
    for elem in spec.get("elements", []):
        if elem.get("id") == "plot_B":
            old_x = elem["position"]["x_mm"]
            elem["position"]["x_mm"] = old_x + 10  # Shift right by 10mm
            logger.info(
                f"  plot_B: x_mm changed from {old_x} to {elem['position']['x_mm']}"
            )

    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    # === Delete cache and exports, re-render ===
    logger.info("\n" + "=" * 60)
    logger.info("Deleting cache/exports and re-rendering...")
    logger.info("=" * 60)

    cache_dir = bundle_path / "cache"
    exports_dir = bundle_path / "exports"

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    if exports_dir.exists():
        shutil.rmtree(exports_dir)

    # Reload and save to regenerate
    fig2 = Figz(bundle_path)
    fig2.save()

    # === Verify changes ===
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION")
    logger.info("=" * 60)

    logger.info("\nFinal positions (from spec.json):")
    for elem in fig2.elements:
        if elem.get("type") == "plot":
            pos = elem.get("position", {})
            logger.info(f"  {elem['id']}: x={pos.get('x_mm')}mm, y={pos.get('y_mm')}mm")

    # Load final geometry_px
    if geometry_path.exists():
        with open(geometry_path) as f:
            final_geometry = json.load(f)
        logger.info("\nFinal geometry_px (derived):")
        for elem in final_geometry.get("elements", []):
            bbox = elem.get("bbox_px", {})
            logger.info(f"  {elem['id']}: x={bbox.get('x', 0):.0f}px")

    # Show what changed
    logger.info("\n" + "-" * 40)
    logger.info("Summary:")
    logger.info("  Panel A: UNCHANGED (x=5mm)")
    logger.info("  Panel B: MOVED (x=60mm -> 70mm)")
    logger.info("  Panel C: UNCHANGED (x=115mm)")
    logger.info("  geometry_px.json: REGENERATED")
    logger.info("  exports/*: REGENERATED")

    logger.info("\n" + "=" * 60)
    logger.info("Key takeaway:")
    logger.info("  spec.json positions (mm) are canonical")
    logger.info("  geometry_px.json (px) is derived on render")
    logger.info("  Edit canonical → delete cache → re-render")
    logger.info("=" * 60)

    logger.success("Example 13 completed!")


if __name__ == "__main__":
    main()

# EOF
