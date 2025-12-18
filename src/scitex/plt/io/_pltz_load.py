#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/io/_pltz_load.py

"""Load and merge utilities for pltz bundles."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from scitex.plt.styles import get_default_dpi

__all__ = [
    "load_layered_pltz_bundle",
    "merge_layered_bundle",
]


def load_layered_pltz_bundle(bundle_dir: Path) -> Dict[str, Any]:
    """Load layered .pltz bundle and return merged spec for editor.

    Parameters
    ----------
    bundle_dir : Path
        Path to .pltz.d bundle.

    Returns
    -------
    dict
        Merged bundle data compatible with editor.
    """
    bundle_dir = Path(bundle_dir)

    result = {
        "spec": None,
        "style": None,
        "geometry": None,
        "merged": None,
        "basename": "plot",
    }

    # Load spec.json
    spec_path = bundle_dir / "spec.json"
    if spec_path.exists():
        with open(spec_path) as f:
            result["spec"] = json.load(f)
            result["basename"] = result["spec"].get("plot_id", "plot")

    # Load style.json
    style_path = bundle_dir / "style.json"
    if style_path.exists():
        with open(style_path) as f:
            result["style"] = json.load(f)

    # Load geometry from cache
    geometry_path = bundle_dir / "cache" / "geometry_px.json"
    if geometry_path.exists():
        with open(geometry_path) as f:
            result["geometry"] = json.load(f)

    # Create merged view for backward compatibility with editor
    result["merged"] = merge_layered_bundle(
        result["spec"], result["style"], result["geometry"]
    )

    return result


def merge_layered_bundle(
    spec: Optional[Dict],
    style: Optional[Dict],
    geometry: Optional[Dict],
) -> Dict[str, Any]:
    """Merge spec/style/geometry into old-format compatible dict for editor.

    This provides backward compatibility with editors expecting the old format.
    """
    if spec is None:
        return {}

    merged = {
        "schema": {"name": "scitex.plt.plot", "version": "2.0.0"},
        "backend": "mpl",
    }

    # Merge data section
    if "data" in spec:
        merged["data"] = {
            "source": spec["data"].get("csv", "data.csv"),
            "path": spec["data"].get("csv", "data.csv"),
            "hash": spec["data"].get("hash"),
        }

    # Merge size from style
    if style and "size" in style:
        merged["size"] = {
            "width_mm": style["size"].get("width_mm", 80),
            "height_mm": style["size"].get("height_mm", 68),
            "dpi": geometry.get("dpi", get_default_dpi())
            if geometry
            else get_default_dpi(),
        }

    # Merge axes from spec + style + geometry
    merged["axes"] = []
    for ax_spec in spec.get("axes", []):
        ax_merged = {
            "id": ax_spec.get("id"),
            "xlabel": ax_spec.get("labels", {}).get("xlabel"),
            "ylabel": ax_spec.get("labels", {}).get("ylabel"),
            "title": ax_spec.get("labels", {}).get("title"),
            "xlim": ax_spec.get("limits", {}).get("x"),
            "ylim": ax_spec.get("limits", {}).get("y"),
            "bbox": ax_spec.get("bbox", {}),
        }

        # Add geometry bbox_px if available
        if geometry:
            for ax_geom in geometry.get("axes", []):
                if ax_geom.get("id") == ax_spec.get("id"):
                    ax_merged["bbox_px"] = ax_geom.get("bbox_px", {})
                    break

        merged["axes"].append(ax_merged)

    # Merge traces with styles
    merged["traces"] = []
    trace_style_map = {}
    if style and "traces" in style:
        for ts in style.get("traces", []):
            if isinstance(ts, dict):
                trace_style_map[ts.get("trace_id", "")] = ts

    for trace in spec.get("traces", []):
        trace_merged = dict(trace)
        trace_id = trace.get("id", "")
        if trace_id in trace_style_map:
            trace_merged.update(trace_style_map[trace_id])
        merged["traces"].append(trace_merged)

    # Merge theme from style
    if style and "theme" in style:
        merged["theme"] = style["theme"]

    # Merge legend from style (for editor compatibility)
    if style and "legend" in style:
        legend_style = style["legend"]
        merged["legend"] = {
            "visible": legend_style.get("visible", True),
            "loc": legend_style.get("location", "best"),
            "location": legend_style.get("location", "best"),
            "frameon": legend_style.get("frameon", False),
            "fontsize": legend_style.get("fontsize"),
            "ncols": legend_style.get("ncols", 1),
            "title": legend_style.get("title"),
        }

    # Merge hit_regions, selectable_regions, and figure_px from geometry
    if geometry:
        if "hit_regions" in geometry:
            merged["hit_regions"] = geometry["hit_regions"]
        if "selectable_regions" in geometry:
            merged["selectable_regions"] = geometry["selectable_regions"]
        if "figure_px" in geometry:
            merged["figure_px"] = geometry["figure_px"]
        if "artists" in geometry:
            merged["artists"] = geometry["artists"]

    return merged


# EOF
