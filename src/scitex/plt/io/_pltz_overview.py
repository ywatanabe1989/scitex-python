#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/io/_pltz_overview.py

"""Overview and README generation for pltz bundles."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from scitex.schema import PltzGeometry, PltzRenderManifest, PltzSpec, PltzStyle

from scitex import logging
from scitex.plt.styles import get_default_dpi, get_preview_dpi
from scitex.schema import PLOT_SPEC_VERSION

logger = logging.getLogger()

__all__ = [
    "generate_pltz_overview",
    "generate_pltz_readme",
    "draw_bbox",
    "format_json_summary",
]


def draw_bbox(ax, bbox: List, color: str, label: str, lw: float = 2) -> None:
    """Draw a bounding box on an axes with label inside."""
    import matplotlib.patches as patches

    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    rect = patches.Rectangle(
        (x0, y0), width, height, linewidth=lw, edgecolor=color, facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(
        x0 + 2,
        y0 + 2,
        label,
        fontsize=6,
        color="white",
        va="top",
        ha="left",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.8),
    )


def format_json_summary(data: Dict, max_depth: int = 2, current_depth: int = 0) -> str:
    """Format JSON data as summary text with limited depth."""
    lines = []

    def _format_value(key: str, value, depth: int, prefix: str = "") -> None:
        indent = "  " * depth
        if depth >= max_depth:
            if isinstance(value, dict):
                lines.append(f"{prefix}{indent}{key}: {{...}} ({len(value)} keys)")
            elif isinstance(value, list):
                lines.append(f"{prefix}{indent}{key}: [...] ({len(value)} items)")
            else:
                val_str = str(value)[:30]
                if len(str(value)) > 30:
                    val_str += "..."
                lines.append(f"{prefix}{indent}{key}: {val_str}")
        elif isinstance(value, dict):
            lines.append(f"{prefix}{indent}{key}:")
            for k, v in list(value.items())[:8]:
                _format_value(k, v, depth + 1, prefix)
            if len(value) > 8:
                lines.append(f"{prefix}{indent}  ... ({len(value) - 8} more)")
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], dict):
                lines.append(f"{prefix}{indent}{key}: [{len(value)} items]")
            else:
                val_str = str(value)[:50]
                if len(str(value)) > 50:
                    val_str += "..."
                lines.append(f"{prefix}{indent}{key}: {val_str}")
        else:
            val_str = str(value)[:40]
            if len(str(value)) > 40:
                val_str += "..."
            lines.append(f"{prefix}{indent}{key}: {val_str}")

    for key, value in data.items():
        _format_value(key, value, current_depth)

    return "\n".join(lines[:40])


def generate_pltz_overview(
    exports_dir: Path, basename: str, cache_dir: Path = None
) -> None:
    """Generate comprehensive overview with plot, hitmap, overlay, bboxes, and JSON.

    Args:
        exports_dir: Path to exports directory.
        basename: Base filename for the bundle.
        cache_dir: Path to cache directory (for hitmap). Defaults to exports_dir.
    """
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    bundle_dir = exports_dir.parent
    if cache_dir is None:
        cache_dir = bundle_dir / "cache"

    png_path = exports_dir / f"{basename}.png"
    hitmap_path = cache_dir / "hitmap.png"

    if not png_path.exists():
        return

    try:
        main_img = Image.open(png_path)
        img_width, img_height = main_img.size
        has_hitmap = hitmap_path.exists()

        spec_data, style_data, geometry_data, manifest_data = {}, {}, {}, {}

        spec_path = bundle_dir / "spec.json"
        style_path = bundle_dir / "style.json"
        geometry_path = cache_dir / "geometry_px.json"
        manifest_path = cache_dir / "render_manifest.json"

        if spec_path.exists():
            with open(spec_path) as f:
                spec_data = json.load(f)
        if style_path.exists():
            with open(style_path) as f:
                style_data = json.load(f)
        if geometry_path.exists():
            with open(geometry_path) as f:
                geometry_data = json.load(f)
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest_data = json.load(f)

        dpi = manifest_data.get("dpi", get_default_dpi())
        panel_size_mm = manifest_data.get("panel_size_mm", [80, 68])

        fig = plt.figure(figsize=(18, 12), facecolor="white")
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)

        # Row 1: Plot | Hitmap | Overlay
        ax_plot = fig.add_subplot(gs[0, 0])
        ax_plot.set_title("Plot", fontweight="bold", fontsize=11)
        ax_plot.imshow(main_img)
        ax_plot.axis("off")

        ax_hitmap = fig.add_subplot(gs[0, 1])
        ax_hitmap.set_title("Hit Regions", fontweight="bold", fontsize=11)
        if has_hitmap:
            hitmap_img = Image.open(hitmap_path)
            ax_hitmap.imshow(hitmap_img)

            color_map = geometry_data.get("hit_regions", {}).get("color_map", {})
            artists = geometry_data.get("artists", [])

            for idx, artist in enumerate(artists):
                bbox = artist.get("bbox_px", {})
                if bbox:
                    x0 = bbox.get("x0", 0)
                    y0 = bbox.get("y0", 0)
                    width = bbox.get("width", 0)
                    height = bbox.get("height", 0)
                    cx, cy = x0 + width / 2, y0 + height / 2

                    color_map_id = str(idx + 1)
                    label = f"artist_{idx}"
                    if color_map_id in color_map:
                        label = color_map[color_map_id].get("label", label)

                    ax_hitmap.text(
                        cx,
                        cy,
                        label,
                        fontsize=8,
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.2", facecolor="black", alpha=0.7
                        ),
                    )
        else:
            ax_hitmap.text(
                0.5,
                0.5,
                "No hitmap",
                ha="center",
                va="center",
                transform=ax_hitmap.transAxes,
            )
        ax_hitmap.axis("off")

        ax_overlay = fig.add_subplot(gs[0, 2])
        ax_overlay.set_title("Overlay (Plot + Hit)", fontweight="bold", fontsize=11)
        ax_overlay.imshow(main_img)
        if has_hitmap:
            hitmap_img = Image.open(hitmap_path).convert("RGBA")
            hitmap_array = np.array(hitmap_img)
            hitmap_array[:, :, 3] = (hitmap_array[:, :, 3] * 0.5).astype(np.uint8)
            ax_overlay.imshow(hitmap_array, alpha=0.5)
        ax_overlay.axis("off")

        # Row 2: Bboxes | JSON | mm Scaler
        ax_bboxes = fig.add_subplot(gs[1, 0])
        ax_bboxes.set_title("Element Bboxes", fontweight="bold", fontsize=11)
        ax_bboxes.imshow(main_img)

        colors = ["red", "blue", "green", "orange", "purple", "cyan"]
        selectable = geometry_data.get("selectable_regions", {})

        for ax_idx, ax_region in enumerate(selectable.get("axes", [])):
            color = colors[ax_idx % len(colors)]

            for key in ["title", "xlabel", "ylabel"]:
                if key in ax_region:
                    bbox = ax_region[key].get("bbox_px", [])
                    if len(bbox) == 4:
                        draw_bbox(ax_bboxes, bbox, color, key)

            if "xaxis" in ax_region and "spine" in ax_region["xaxis"]:
                bbox = ax_region["xaxis"]["spine"].get("bbox_px", [])
                if len(bbox) == 4:
                    draw_bbox(ax_bboxes, bbox, "gray", "xaxis", lw=1)

            if "yaxis" in ax_region and "spine" in ax_region["yaxis"]:
                bbox = ax_region["yaxis"]["spine"].get("bbox_px", [])
                if len(bbox) == 4:
                    draw_bbox(ax_bboxes, bbox, "gray", "yaxis", lw=1)

            if "legend" in ax_region:
                bbox = ax_region["legend"].get("bbox_px", [])
                if len(bbox) == 4:
                    draw_bbox(ax_bboxes, bbox, "magenta", "legend")

        ax_bboxes.axis("off")

        ax_json = fig.add_subplot(gs[1, 1])
        ax_json.set_title("Bundle Info (depth=2)", fontweight="bold", fontsize=11)
        ax_json.axis("off")

        json_text = format_json_summary(
            {"spec": spec_data, "style": style_data}, max_depth=2
        )
        ax_json.text(
            0.02,
            0.98,
            json_text,
            transform=ax_json.transAxes,
            fontsize=7,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax_scale = fig.add_subplot(gs[1, 2])
        ax_scale.set_title("Size & Scale (mm)", fontweight="bold", fontsize=11)
        ax_scale.imshow(main_img, extent=[0, panel_size_mm[0], panel_size_mm[1], 0])

        for x in range(0, int(panel_size_mm[0]) + 1, 10):
            ax_scale.axvline(x, color="gray", linewidth=0.5, alpha=0.5)
            if x > 0:
                ax_scale.text(x, -1, f"{x}", ha="center", fontsize=7)
        for y in range(0, int(panel_size_mm[1]) + 1, 10):
            ax_scale.axhline(y, color="gray", linewidth=0.5, alpha=0.5)
            if y > 0:
                ax_scale.text(-1, y, f"{y}", ha="right", va="center", fontsize=7)

        ax_scale.set_xlabel("mm", fontsize=9)
        ax_scale.set_ylabel("mm", fontsize=9)
        ax_scale.set_xlim(-3, panel_size_mm[0] + 1)
        ax_scale.set_ylim(panel_size_mm[1] + 1, -3)

        size_text = (
            f"Panel: {panel_size_mm[0]:.1f} x {panel_size_mm[1]:.1f} mm\n"
            f"DPI: {dpi}\nPixels: {img_width} x {img_height}"
        )
        ax_scale.text(
            panel_size_mm[0] * 0.95,
            panel_size_mm[1] * 0.95,
            size_text,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        fig.suptitle(f"Overview: {basename}", fontsize=14, fontweight="bold", y=0.98)

        overview_path = exports_dir / f"{basename}_overview.png"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*tight_layout.*")
            fig.savefig(
                overview_path,
                dpi=get_preview_dpi(),
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(fig)

    except Exception as e:
        logger.debug(f"Could not generate pltz overview: {e}")
        import traceback

        logger.debug(traceback.format_exc())


def generate_pltz_readme(
    bundle_dir: Path,
    basename: str,
    spec: "PltzSpec",
    style: "PltzStyle",
    geometry: "PltzGeometry",
    manifest: "PltzRenderManifest",
) -> None:
    """Generate a dynamic README.md describing the bundle."""
    from datetime import datetime

    n_axes = len(spec.axes) if spec.axes else 0
    n_traces = len(spec.traces) if spec.traces else 0

    width_mm = style.size.width_mm if style.size else 0
    height_mm = style.size.height_mm if style.size else 0
    dpi = manifest.dpi
    render_px = manifest.render_px

    readme_content = f"""# {basename}.pltz.d

> SciTeX Layered Plot Bundle - Auto-generated README

## Overview

![Plot Overview](exports/{basename}_overview.png)

## Bundle Structure

```
{basename}.pltz.d/
+-- spec.json           # WHAT to plot (semantic, editable)
+-- style.json          # HOW it looks (appearance, editable)
+-- {basename}.csv      # Raw data (immutable)
+-- exports/
|   +-- {basename}.png          # Main plot image
|   +-- {basename}.svg          # Vector version
|   +-- {basename}_overview.png # Visual summary
+-- cache/
|   +-- geometry_px.json       # Pixel coordinates (regenerable)
|   +-- render_manifest.json   # Render metadata
|   +-- hitmap.png             # Hit detection image
|   +-- hitmap.svg             # Vector hit detection
+-- README.md           # This file
```

## Plot Information

| Property | Value |
|----------|-------|
| Plot ID | `{spec.plot_id}` |
| Axes | {n_axes} |
| Traces | {n_traces} |
| Size | {width_mm:.1f} x {height_mm:.1f} mm |
| DPI | {dpi} |
| Pixels | {render_px[0]} x {render_px[1]} |
| Theme | {style.theme.mode if style.theme else "light"} |

## Coordinate System

The bundle uses a layered coordinate system:

1. **spec.json + style.json** = Source of truth (edit these)
2. **cache/** = Derived data (can be deleted and regenerated)

### Coordinate Transformation Pipeline

```
Original Figure (at export DPI)
         |
         v crop_box offset
    +-------------------+
    |  Final PNG        |  <- bbox_px coordinates are in this space
    |  ({render_px[0]} x {render_px[1]})  |
    +-------------------+
```

**Formula**: `final_coords = original_coords - crop_offset`

## Usage

### Python

```python
import scitex as stx

# Load the bundle
bundle = stx.plt.io.load_layered_pltz_bundle("{bundle_dir}")

# Access components
spec = bundle["spec"]       # What to plot
style = bundle["style"]     # How it looks
geometry = bundle["geometry"]  # Where in pixels
```

### Editing

Edit `spec.json` to change:
- Axis labels, titles, limits
- Trace data columns
- Data source

Edit `style.json` to change:
- Colors, line widths
- Font sizes
- Theme (light/dark)

After editing, regenerate cache with:
```python
stx.plt.io.regenerate_cache("{bundle_dir}")
```

---

*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*Schema: scitex.plt v{PLOT_SPEC_VERSION}*
"""

    readme_path = bundle_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)


# EOF
