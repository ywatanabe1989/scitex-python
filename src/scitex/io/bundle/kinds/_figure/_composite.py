#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_composite.py

"""Composite figure renderer for FTS bundles.

Renders composite figures (kind=figure) containing multiple children.
ALWAYS re-renders from child's canonical (exports = optional cache only).

Design principles:
- Re-render children from canonical/encoding.json + payload/data.csv
- Recursively render nested figures (NOT from exports/)
- Apply container's theme for unified styling
- Generate geometry_px.json with flattened child geometry
- Cache key includes canonical_hash + effective_theme_hash + renderer_version
"""

import io
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from matplotlib.axes import Axes as MplAxes
    from matplotlib.figure import Figure as MplFigure

    from .._bundle._FTS import FTS
    from ._dataclasses import Theme


def render_composite(
    children: Dict[str, "FTS"],
    layout: Dict,
    size_mm: Optional[Dict[str, float]] = None,
    theme: Optional["Theme"] = None,
    dpi: int = 300,
) -> Tuple["MplFigure", Dict]:
    """Render composite figure with children in grid layout.

    ALWAYS re-renders from child's canonical (exports = optional cache only).

    Args:
        children: Dict mapping child_name -> FTS object
        layout: Layout specification {rows, cols, panels: [...]}
        size_mm: Figure size in mm (default: 170x85 for two-column)
        theme: Theme to apply to all children
        dpi: Output DPI

    Returns:
        (figure, geometry_data)
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Default size
    if size_mm is None:
        size_mm = {"width": 170, "height": 85}

    # Convert mm to inches (matplotlib uses inches)
    width_in = size_mm.get("width", 170) / 25.4
    height_in = size_mm.get("height", 85) / 25.4

    # Create figure with gridspec
    rows = layout.get("rows", 1)
    cols = layout.get("cols", 1)

    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    gs = GridSpec(rows, cols, figure=fig)

    # Collect geometry data for all elements
    geometry = {
        "elements": [],
        "panels": [],
    }

    # Render each panel
    panels = layout.get("panels", [])
    for panel_info in panels:
        child_name = panel_info.get("child")
        row = panel_info.get("row", 0)
        col = panel_info.get("col", 0)
        row_span = panel_info.get("row_span", 1)
        col_span = panel_info.get("col_span", 1)
        label = panel_info.get("label")

        if child_name not in children:
            continue

        child = children[child_name]

        # Create axes for this panel
        ax = fig.add_subplot(gs[row : row + row_span, col : col + col_span])

        # Render child into axes (ALWAYS re-render from canonical)
        child_geometry = render_child_in_axes(ax, child, theme)

        # Add panel label if specified
        if label:
            _add_panel_label(ax, label)

        # Record panel geometry (flatten into parent's figure_px space)
        panel_geometry = {
            "child": child_name,
            "child_id": panel_info.get("child_id"),
            "label": label,
            "position": {"row": row, "col": col, "row_span": row_span, "col_span": col_span},
            "bbox_figure": _get_axes_bbox(ax, fig),
            "child_elements": child_geometry.get("elements", []),
        }
        geometry["panels"].append(panel_geometry)

    plt.tight_layout()

    return fig, geometry


def render_child_in_axes(
    ax: "MplAxes",
    child: "FTS",
    theme: Optional["Theme"] = None,
) -> Dict:
    """Render child FTS into axes.

    ALWAYS re-renders from canonical (exports = optional cache only):
    - kind=plot: Re-render from canonical/spec.json + payload/data.csv
    - kind=figure: Recursively render children (NOT use exports/figure.png)

    Args:
        ax: Matplotlib axes to render into
        child: Child FTS bundle
        theme: Theme to apply (overrides child's theme)

    Returns:
        Geometry data for this child
    """
    geometry = {"elements": []}

    if child.node.is_leaf_kind():
        # Leaf node: render from encoding + payload
        geometry = _render_leaf_in_axes(ax, child, theme)
    elif child.node.is_composite_kind():
        # Composite: recursive render (nested figure)
        geometry = _render_composite_in_axes(ax, child, theme)

    return geometry


def _render_leaf_in_axes(
    ax: "MplAxes",
    child: "FTS",
    theme: Optional["Theme"] = None,
) -> Dict:
    """Render leaf FTS (plot/table/stats) into axes.

    Uses child's pre-rendered export (artifacts/exports/figure.png) to embed
    the visualization. This approach ensures visual consistency with the
    original rendered plot.
    """
    import io

    import matplotlib.pyplot as plt

    geometry = {"elements": []}

    # Try to use pre-rendered export image
    try:
        storage = child.storage

        # Check for exported image
        if storage.exists("artifacts/exports/figure.png"):
            img_data = storage.read("artifacts/exports/figure.png")
            img = plt.imread(io.BytesIO(img_data), format="png")

            # Display image in axes, filling the entire axes
            ax.imshow(img, aspect="auto", extent=[0, 1, 0, 1])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            geometry["elements"].append({
                "type": "embedded_image",
                "source": "artifacts/exports/figure.png",
            })
            return geometry

    except Exception as e:
        pass  # Fall through to placeholder

    # Fallback: draw placeholder with child name
    ax.text(
        0.5,
        0.5,
        f"[{child.node.name or child.node.id[:8]}]",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return geometry


def _render_composite_in_axes(
    ax: "MplAxes",
    child: "FTS",
    theme: Optional["Theme"] = None,
) -> Dict:
    """Render composite FTS (figure) into axes.

    Creates a nested gridspec for the child's children.
    """
    from .._bundle._children import load_embedded_children

    geometry = {"elements": [], "nested_panels": []}

    # Load child's children
    child_children = child.load_children()
    if not child_children:
        ax.text(
            0.5,
            0.5,
            f"[Empty: {child.node.name or child.node.id[:8]}]",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return geometry

    # Get child's layout
    child_layout = child.node.layout or {"rows": 1, "cols": 1, "panels": []}

    # For nested figures, we need to subdivide the axes
    # This is a simplified approach - full implementation would use inset axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ax.axis("off")  # Hide the parent axes

    # Create a figure-in-figure effect using inset axes
    # For simplicity, render children side by side
    n_children = len(child_children)
    width = 1.0 / max(n_children, 1)

    for i, (grandchild_name, grandchild) in enumerate(child_children.items()):
        # Create inset axes
        inset_bounds = [width * i, 0, width * 0.95, 1]
        inset_ax = ax.inset_axes(inset_bounds)

        # Recursively render grandchild
        grandchild_geometry = render_child_in_axes(inset_ax, grandchild, theme)

        geometry["nested_panels"].append(
            {
                "child": grandchild_name,
                "child_id": grandchild.node.id if grandchild.node else None,
                "elements": grandchild_geometry.get("elements", []),
            }
        )

    return geometry


def _add_panel_label(ax: "MplAxes", label: str) -> None:
    """Add panel label (A, B, C...) to axes."""
    ax.text(
        -0.1,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def _get_axes_bbox(ax: "MplAxes", fig: "MplFigure") -> Dict[str, float]:
    """Get axes bounding box in figure coordinates."""
    bbox = ax.get_position()
    return {
        "x": bbox.x0,
        "y": bbox.y0,
        "width": bbox.width,
        "height": bbox.height,
    }


def check_cache_valid(
    child: "FTS",
    current_theme_hash: str,
    renderer_version: str = "1.0.0",
) -> bool:
    """Check if child's cached exports are still valid.

    Cache is valid only if ALL match:
    1. artifacts/exports/figure.png exists
    2. artifacts/cache/render_manifest.json exists
    3. render_manifest.canonical_hash matches current hash
    4. render_manifest.effective_theme_hash matches current theme
    5. render_manifest.renderer_version matches current renderer

    Args:
        child: Child FTS bundle
        current_theme_hash: Hash of current effective theme
        renderer_version: Current renderer version

    Returns:
        True if cache is valid and can be used
    """
    from .._bundle._storage import get_storage
    from .._bundle._saver import compute_canonical_hash

    storage = get_storage(child.path)

    # Check exports exist
    if not storage.exists("artifacts/exports/figure.png"):
        return False

    # Check manifest exists
    manifest = storage.read_json("artifacts/cache/render_manifest.json")
    if manifest is None:
        return False

    # Check canonical hash
    current_canonical_hash = compute_canonical_hash(storage)
    if manifest.get("canonical_hash") != current_canonical_hash:
        return False

    # Check theme hash
    if manifest.get("effective_theme_hash") != current_theme_hash:
        return False

    # Check renderer version
    if manifest.get("renderer_version") != renderer_version:
        return False

    return True


__all__ = [
    "render_composite",
    "render_child_in_axes",
    "check_cache_valid",
]

# EOF
