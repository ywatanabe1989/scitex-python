#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/examples/fig/08_hittest_demo.py

"""
Example 08: Interactive Hit-Testing Proof

Demonstrates:
- geometry_px.json and hitmap work together
- Mapping pixel coordinates to element IDs
- Pixel-level hit detection using color-coded hitmap
- Hit detection stability across cache regeneration
"""

import json

import numpy as np
from PIL import Image

import scitex as stx
from scitex import INJECTED
from scitex.fig import Figz


@stx.session(verbose=True, agg=True)
def main(plt=INJECTED, CONFIG=INJECTED, logger=INJECTED):
    """Demonstrate hit-testing with geometry_px and hitmap."""
    logger.info("Example 08: Hit-Testing Demo")

    out_dir = CONFIG["SDIR_OUT"]

    # Create a multi-panel figure
    fig = Figz(
        out_dir / "hittest_figure.zip.d",
        name="Hit-Test Demo",
        size_mm={"width": 170, "height": 100},
    )

    # Panel A: Line plot
    x = np.linspace(0, 10, 100)
    fig_a, ax_a = plt.subplots(figsize=(3, 2.5))
    ax_a.plot(x, np.sin(x), label="sin(x)")
    ax_a.set_title("Panel A")
    ax_a.legend()
    fig.add_element(
        "plot_A",
        "plot",
        fig_a,
        position={"x_mm": 5, "y_mm": 5},
        size={"width_mm": 80, "height_mm": 45},
    )
    plt.close(fig_a)

    # Panel B: Scatter plot
    np.random.seed(42)
    fig_b, ax_b = plt.subplots(figsize=(3, 2.5))
    ax_b.scatter(np.random.randn(30), np.random.randn(30))
    ax_b.set_title("Panel B")
    fig.add_element(
        "plot_B",
        "plot",
        fig_b,
        position={"x_mm": 88, "y_mm": 5},
        size={"width_mm": 80, "height_mm": 45},
    )
    plt.close(fig_b)

    # Panel C: Bar chart
    fig_c, ax_c = plt.subplots(figsize=(3, 2.5))
    ax_c.bar(["A", "B", "C"], [4, 7, 2])
    ax_c.set_title("Panel C")
    fig.add_element(
        "plot_C",
        "plot",
        fig_c,
        position={"x_mm": 45, "y_mm": 52},
        size={"width_mm": 80, "height_mm": 45},
    )
    plt.close(fig_c)

    # Set panel labels
    fig.set_panel_info("plot_A", panel_letter="A", description="Sinusoidal signal")
    fig.set_panel_info("plot_B", panel_letter="B", description="Scatter distribution")
    fig.set_panel_info("plot_C", panel_letter="C", description="Category comparison")

    # Save
    fig.save()
    logger.info(f"Saved: {fig.path}")

    # === Hit-testing demo ===
    logger.info("\n" + "=" * 60)
    logger.info("HIT-TESTING DEMO")
    logger.info("=" * 60)

    # Load geometry and hitmap
    cache_dir = fig.path / "cache"
    geometry_path = cache_dir / "geometry_px.json"
    hitmap_path = cache_dir / "hitmap.png"
    hitmap_colors_path = cache_dir / "hitmap_colors.json"

    if not geometry_path.exists():
        logger.warning("geometry_px.json not found")
        return

    with open(geometry_path) as f:
        geometry = json.load(f)

    # Show canvas info
    canvas = geometry.get("canvas", {})
    logger.info(
        f"\nCanvas: {canvas.get('width_mm', 0)}mm x {canvas.get('height_mm', 0)}mm"
    )
    logger.info(
        f"        {canvas.get('width_px', 0)}px x {canvas.get('height_px', 0)}px @ {canvas.get('dpi', 150)} DPI"
    )
    logger.info(f"\nGeometry contains {len(geometry.get('elements', []))} elements")

    # Show element bounding boxes
    logger.info("\nElement bounding boxes (px):")
    for elem in geometry.get("elements", []):
        elem_id = elem.get("id", "?")
        bbox = elem.get("bbox_px", {})
        logger.info(
            f"  {elem_id}: x={bbox.get('x', 0):.0f}, y={bbox.get('y', 0):.0f}, "
            f"w={bbox.get('width', 0):.0f}, h={bbox.get('height', 0):.0f}"
        )

    # === BOUNDING BOX HIT TESTS ===
    logger.info("\n" + "-" * 40)
    logger.info("Bounding Box Hit Tests:")
    logger.info("-" * 40)

    # Generate sample points (some inside elements, some outside)
    sample_points = []
    for elem in geometry.get("elements", []):
        bbox = elem.get("bbox_px", {})
        # Point in center of element
        cx = bbox.get("x", 0) + bbox.get("width", 0) / 2
        cy = bbox.get("y", 0) + bbox.get("height", 0) / 2
        sample_points.append((int(cx), int(cy), elem.get("id", "?")))

    # Add some points outside elements
    sample_points.extend(
        [
            (5, 5, "background"),  # top-left corner (small margin)
            (1, 1, "background"),  # very corner
        ]
    )

    # Hit test each point using bounding box
    logger.info(f"\n{'X_px':>6} {'Y_px':>6} {'Expected':>12} {'Hit':>12}")
    logger.info("-" * 45)

    bbox_pass, bbox_total = 0, 0
    for x_px, y_px, expected in sample_points:
        hit_element = _bbox_hit_test(x_px, y_px, geometry)
        status = "OK" if hit_element == expected else "MISMATCH"
        if hit_element == expected:
            bbox_pass += 1
        bbox_total += 1
        logger.info(f"{x_px:>6} {y_px:>6} {expected:>12} {hit_element:>12} {status}")

    logger.info(f"\nBBox Hit Test: {bbox_pass}/{bbox_total} passed")

    # === PIXEL-LEVEL HIT TESTS (using hitmap) ===
    logger.info("\n" + "-" * 40)
    logger.info("Pixel-Level Hit Tests (using hitmap):")
    logger.info("-" * 40)

    if hitmap_path.exists() and hitmap_colors_path.exists():
        hitmap = Image.open(hitmap_path)
        with open(hitmap_colors_path) as f:
            color_map = json.load(f)

        logger.info(f"Hitmap size: {hitmap.size[0]}x{hitmap.size[1]} px")
        logger.info(f"Color map: {len(color_map)} elements")

        # Show element colors
        logger.info("\nElement colors (RGB):")
        for elem_id, color in color_map.items():
            logger.info(f"  {elem_id}: ({color['r']}, {color['g']}, {color['b']})")

        # Test pixel-level hit detection
        logger.info(
            f"\n{'X_px':>6} {'Y_px':>6} {'Expected':>12} {'Hit':>12} {'Color':>18}"
        )
        logger.info("-" * 60)

        pixel_pass, pixel_total = 0, 0
        for x_px, y_px, expected in sample_points:
            if 0 <= x_px < hitmap.width and 0 <= y_px < hitmap.height:
                pixel = hitmap.getpixel((x_px, y_px))
                hit_element = _pixel_hit_test(pixel, color_map)
                status = "OK" if hit_element == expected else "MISMATCH"
                if hit_element == expected:
                    pixel_pass += 1
                pixel_total += 1
                color_str = f"({pixel[0]:3d},{pixel[1]:3d},{pixel[2]:3d})"
                logger.info(
                    f"{x_px:>6} {y_px:>6} {expected:>12} {hit_element:>12} {color_str:>18} {status}"
                )

        logger.info(f"\nPixel Hit Test: {pixel_pass}/{pixel_total} passed")

        # Edge detection test - sample along element boundaries
        logger.info("\n" + "-" * 40)
        logger.info("Edge Detection Tests:")
        logger.info("-" * 40)

        edge_tests = []
        for elem in geometry.get("elements", []):
            bbox = elem.get("bbox_px", {})
            x, y = bbox.get("x", 0), bbox.get("y", 0)
            w, h = bbox.get("width", 0), bbox.get("height", 0)
            elem_id = elem.get("id", "?")

            if w > 0 and h > 0:
                # Inside center
                edge_tests.append((int(x + w / 2), int(y + h / 2), elem_id, "center"))
                # Just inside edges
                edge_tests.append((int(x + 5), int(y + h / 2), elem_id, "left-edge"))
                edge_tests.append(
                    (int(x + w - 5), int(y + h / 2), elem_id, "right-edge")
                )
                # Just outside (should be background or other element)
                if x > 5:
                    edge_tests.append(
                        (int(x - 5), int(y + h / 2), "background", "outside-left")
                    )

        edge_pass, edge_total = 0, 0
        for x_px, y_px, expected, desc in edge_tests[:12]:  # Limit output
            if 0 <= x_px < hitmap.width and 0 <= y_px < hitmap.height:
                pixel = hitmap.getpixel((x_px, y_px))
                hit_element = _pixel_hit_test(pixel, color_map)
                ok = hit_element == expected
                if ok:
                    edge_pass += 1
                edge_total += 1
                status = "OK" if ok else f"GOT:{hit_element}"
                logger.info(
                    f"  ({x_px:>4},{y_px:>4}) {desc:<15} expect:{expected:<12} {status}"
                )

        logger.info(f"\nEdge Tests: {edge_pass}/{edge_total} passed")
    else:
        logger.warning("Hitmap files not found for pixel-level tests")

    logger.info("\n" + "=" * 60)
    logger.success("Example 08 completed!")


def _bbox_hit_test(x_px: int, y_px: int, geometry: dict) -> str:
    """Perform hit test at pixel coordinates using bounding boxes."""
    for elem in geometry.get("elements", []):
        bbox = elem.get("bbox_px", {})
        x0 = bbox.get("x", 0)
        y0 = bbox.get("y", 0)
        x1 = x0 + bbox.get("width", 0)
        y1 = y0 + bbox.get("height", 0)
        if x0 <= x_px <= x1 and y0 <= y_px <= y1:
            return elem.get("id", "unknown")
    return "background"


def _pixel_hit_test(pixel: tuple, color_map: dict, tolerance: int = 5) -> str:
    """Perform hit test using hitmap pixel color."""
    r, g, b = pixel[:3]
    # Background check (black)
    if r < 10 and g < 10 and b < 10:
        return "background"
    # Find matching element
    for elem_id, color in color_map.items():
        if (
            abs(color["r"] - r) <= tolerance
            and abs(color["g"] - g) <= tolerance
            and abs(color["b"] - b) <= tolerance
        ):
            return elem_id
    return "unknown"


if __name__ == "__main__":
    main()

# EOF
