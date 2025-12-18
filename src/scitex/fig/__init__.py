#!/usr/bin/env python3
# Timestamp: 2025-12-17
# File: ./src/scitex/fig/__init__.py
"""
SciTeX Figure Module (scitex.fig)

Composition of publication-quality figures using .figz bundles.

Terminology:
- Figure: A publication figure (e.g., "Figure 1")
- Panel: A single plot component (.pltz bundle)
- Bundle: Self-contained directory with spec, style, data, exports

Quick Start:
-----------
>>> import scitex as stx
>>>
>>> # Create figure from panels
>>> panels = {"A": "plot_a.pltz.d", "B": "plot_b.pltz.d"}
>>> stx.fig.save_figz(panels, "Figure1.figz.d")
>>>
>>> # Load figure
>>> figure = stx.fig.load_figz("Figure1.figz.d")

Directory Structure (.figz.d):
-----------------------------
Figure1.figz.d/
    ├── spec.json       # Figure specification
    ├── style.json      # Figure style
    ├── A.pltz.d/       # Panel A bundle
    ├── B.pltz.d/       # Panel B bundle
    ├── exports/        # Figure exports (PNG, SVG)
    └── cache/          # Cached geometry
"""

# Submodules for advanced use
from . import backend, editor, io, layout, layout_viz, model, utils

# OOP Bundle API (Unified Element Model)
from ._bundle import Figz

# Editor
from .editor import edit

# Layout utilities
from .layout import (
    auto_crop_layout,
    auto_layout_grid,
    content_bounds,
    element_bounds,
    normalize_position,
    normalize_size,
    to_absolute,
    to_relative,
)

# Layout visualization (blueprint-style)
from .layout_viz import (
    BLUEPRINT_STYLE,
    plot_auto_crop_comparison,
    plot_layout,
)

# =============================================================================
# .figz Bundle Support
# =============================================================================


def save_figz(
    panels,
    path,
    spec=None,
    as_zip=None,
):
    """
    Save panels as a .figz publication figure bundle.

    Parameters
    ----------
    panels : dict
        Dictionary mapping panel IDs to .pltz bundle paths or data.
        Example: {"A": "timecourse.pltz.d", "B": "barplot.pltz.d"}
    path : str or Path
        Output path (e.g., "Figure1.figz" or "Figure1.figz.d").
        - Path ending with ".figz" creates ZIP archive (default behavior)
        - Path ending with ".figz.d" creates directory bundle
    spec : dict, optional
        Figure specification. Auto-generated if None.
    as_zip : bool, optional
        If True, save as ZIP archive. If False, save as directory.
        Default: auto-detect from path (ZIP for .figz, directory for .figz.d).

    Returns
    -------
    Path
        Path to saved bundle.

    Examples
    --------
    >>> import scitex.fig as sfig
    >>> panels = {
    ...     "A": "timecourse.pltz.d",
    ...     "B": "barplot.pltz.d"
    ... }
    >>> sfig.save_figz(panels, "Figure1.figz")    # Creates ZIP
    >>> sfig.save_figz(panels, "Figure1.figz.d")  # Creates directory
    """
    from pathlib import Path

    from scitex.io.bundle import BundleType, save

    p = Path(path)
    spath = str(path)

    # Auto-detect as_zip from path suffix if not specified
    if as_zip is None:
        as_zip = not spath.endswith(".d")

    # Auto-generate spec if not provided
    if spec is None:
        spec = _generate_figure_spec(panels)

    # Build bundle data - pass source paths directly for file copying
    bundle_data = {
        "spec": spec,
        "plots": {},
    }

    # Pass source paths directly (not loaded data) to preserve all files
    for panel_id, pltz_source in panels.items():
        pltz_path = Path(pltz_source)
        if pltz_path.exists():
            # Store source path for direct copying
            bundle_data["plots"][panel_id] = str(pltz_path)

    return save(bundle_data, p, bundle_type=BundleType.FIGZ, as_zip=as_zip)


def load_figz(path):
    """
    Load a .figz bundle.

    Parameters
    ----------
    path : str or Path
        Path to .figz bundle (directory or ZIP).

    Returns
    -------
    dict
        Figure data with:
        - 'spec': Figure specification
        - 'panels': Dict mapping panel IDs to {'spec': ..., 'data': ...}

    Examples
    --------
    >>> figure = scitex.fig.load_figz("Figure1.figz.d")
    >>> print(figure['spec']['figure']['title'])
    >>> panel_a = figure['panels']['A']
    >>> print(panel_a['spec'], panel_a['data'])
    """
    from scitex.io.bundle import load

    bundle = load(path)

    if bundle["type"] != "figz":
        raise ValueError(f"Not a .figz bundle: {path}")

    result = {
        "spec": bundle.get("spec", {}),
        "panels": {},
    }

    # Return spec and data for each panel (reconstruction is optional)
    for panel_id, plot_bundle in bundle.get("plots", {}).items():
        result["panels"][panel_id] = {
            "spec": plot_bundle.get("spec", {}),
            "data": plot_bundle.get("data"),
        }

    return result


def _generate_figure_spec(panels):
    """Generate figure.json spec from panels."""
    from pathlib import Path

    spec = {
        "schema": {"name": "scitex.fig.figure", "version": "1.0.0"},
        "figure": {
            "id": "figure",
            "title": "",
            "caption": "",
            "styles": {
                "size": {"width_mm": 180, "height_mm": 120},
                "background": "#ffffff",
            },
        },
        "panels": [],
    }

    # Auto-layout panels
    panel_ids = sorted(panels.keys())
    n_panels = len(panel_ids)

    if n_panels == 0:
        return spec

    # Simple grid layout
    cols = min(n_panels, 2)
    rows = (n_panels + cols - 1) // cols

    panel_w = 80
    panel_h = 50
    margin = 5

    for i, panel_id in enumerate(panel_ids):
        row = i // cols
        col = i % cols

        x = margin + col * (panel_w + margin)
        y = margin + row * (panel_h + margin)

        # Note: save_bundle uses panel_id for the directory name (e.g., A.pltz.d)
        spec["panels"].append(
            {
                "id": panel_id,
                "label": panel_id,
                "caption": "",
                "plot": f"{panel_id}.pltz.d",
                "position": {"x_mm": x, "y_mm": y},
                "size": {"width_mm": panel_w, "height_mm": panel_h},
            }
        )

    return spec


__all__ = [
    # Submodules (advanced)
    "io",
    "model",
    "backend",
    "utils",
    "editor",
    "layout",
    "layout_viz",
    # Editor
    "edit",
    # OOP Bundle API (Unified Element Model)
    "Figz",
    # Layout utilities
    "to_absolute",
    "to_relative",
    "normalize_position",
    "normalize_size",
    "element_bounds",
    "content_bounds",
    "auto_layout_grid",
    "auto_crop_layout",
    # Layout visualization
    "plot_layout",
    "plot_auto_crop_comparison",
    "BLUEPRINT_STYLE",
    # Legacy functions (deprecated)
    "save_figz",
    "load_figz",
]

# EOF
