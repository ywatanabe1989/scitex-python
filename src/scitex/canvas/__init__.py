#!/usr/bin/env python3
# Timestamp: 2025-12-08
# File: ./src/scitex/vis/__init__.py
"""
SciTeX Visualization Module (scitex.canvas)

Canvas-based composition of publication-quality figures.

Terminology:
- Canvas: A paper figure workspace (e.g., "Figure 1" in publication)
- Panel: A single component on canvas (stx.plt output or image)
- Figure: Reserved for matplotlib's fig object (see scitex.plt)

Quick Start:
-----------
>>> import scitex as stx
>>>
>>> # Create canvas and add panels
>>> stx.vis.create_canvas("/output", "fig1")
>>> stx.vis.add_panel("/output", "fig1", "panel_a", source="plot.png",
...                   xy_mm=(10, 10), size_mm=(80, 60), label="A")
>>>
>>> # Save with stx.io (auto-exports PNG/PDF/SVG)
>>> canvas = stx.io.load("/output/fig1.canvas")
>>> stx.io.save(canvas, "/output/fig1_copy.canvas")

Directory Structure:
-------------------
{parent_dir}/{canvas_name}.canvas/
    ├── canvas.json     # Layout, panels, composition
    ├── panels/         # Panel directories
    └── exports/        # canvas.png, canvas.pdf, canvas.svg
"""

# Submodules for advanced use
from . import backend, editor, io, model, utils

# Canvas class
from .canvas import Canvas

# Editor
from .editor import edit

# Data integrity
# Export (usually handled by stx.io.save, but available for explicit use)
# =============================================================================
# Primary API (minimal, reusable, flexible)
# =============================================================================
# Canvas operations
# Panel operations
from .io import add_panel_from_image, add_panel_from_scitex
from .io import canvas_directory_exists as canvas_exists
from .io import delete_canvas_directory as delete_canvas
from .io import ensure_canvas_directory as create_canvas
from .io import export_canvas_to_file as export_canvas
from .io import get_canvas_directory_path as get_canvas_path
from .io import list_canvas_directories as list_canvases
from .io import list_panels, remove_panel, update_panel
from .io import verify_all_data_hashes as verify_data


# =============================================================================
# Convenience wrapper for add_panel
# =============================================================================
def add_panel(
    parent_dir,
    canvas_name,
    panel_name,
    source,
    xy_mm=(0, 0),
    size_mm=(50, 50),
    label="",
    bundle=False,
    **kwargs,
):
    """
    Add a panel to canvas (auto-detects scitex vs image type).

    Parameters
    ----------
    parent_dir : str or Path
        Parent directory containing canvas
    canvas_name : str
        Canvas name
    panel_name : str
        Name for the panel
    source : str or Path
        Source file (PNG, JPG, SVG)
    xy_mm : tuple
        (x_mm, y_mm) position on canvas in millimeters
    size_mm : tuple
        (width_mm, height_mm) panel size in millimeters
    label : str
        Panel label (A, B, C...)
    bundle : bool
        If True, copy files. If False (default), use symlinks.
    **kwargs
        Additional panel properties (rotation_deg, opacity, flip_h, etc.)
    """
    from pathlib import Path

    source = Path(source)
    panel_properties = {
        "position": {"x_mm": xy_mm[0], "y_mm": xy_mm[1]},
        "size": {"width_mm": size_mm[0], "height_mm": size_mm[1]},
        **kwargs,
    }
    if label:
        panel_properties["label"] = {"text": label, "position": "top-left"}

    # Check if scitex output (has .json/.csv siblings)
    json_sibling = source.parent / f"{source.stem}.json"
    if json_sibling.exists():
        return add_panel_from_scitex(
            project_dir=parent_dir,
            canvas_name=canvas_name,
            panel_name=panel_name,
            source_png=source,
            panel_properties=panel_properties,
            bundle=bundle,
        )
    else:
        return add_panel_from_image(
            project_dir=parent_dir,
            canvas_name=canvas_name,
            panel_name=panel_name,
            source_image=source,
            panel_properties=panel_properties,
            bundle=bundle,
        )


# =============================================================================
# .figure Bundle Support
# =============================================================================


def save_figure(
    panels,
    path,
    spec=None,
    as_zip=None,
):
    """
    Save panels as a .figure publication figure bundle.

    Parameters
    ----------
    panels : dict
        Dictionary mapping panel IDs to .plot bundle paths or data.
        Example: {"A": "timecourse.plot", "B": "barplot.plot"}
    path : str or Path
        Output path (e.g., "Figure1.figure.zip" or "Figure1.figure").
        - Path ending with ".zip" creates ZIP archive
        - Path ending with ".figure" creates directory bundle
    spec : dict, optional
        Figure specification. Auto-generated if None.
    as_zip : bool, optional
        If True, save as ZIP archive. If False, save as directory.
        Default: auto-detect from path.

    Returns
    -------
    Path
        Path to saved bundle.

    Examples
    --------
    >>> import scitex.canvas as sfig
    >>> panels = {
    ...     "A": "timecourse.plot",
    ...     "B": "barplot.plot"
    ... }
    >>> sfig.save_figure(panels, "Figure1.figure.zip")  # Creates ZIP
    >>> sfig.save_figure(panels, "Figure1.figure")      # Creates directory
    """
    import shutil
    from pathlib import Path

    from scitex.io.bundle import BundleType, save

    p = Path(path)
    spath = str(path)

    # Auto-detect as_zip from path suffix if not specified
    if as_zip is None:
        as_zip = spath.endswith(".zip")

    # Auto-generate spec if not provided
    if spec is None:
        spec = _generate_figure_spec(panels)

    # Build bundle data - pass source paths directly for file copying
    bundle_data = {
        "spec": spec,
        "plots": {},
    }

    # Pass source paths directly (not loaded data) to preserve all files
    for panel_id, plot_source in panels.items():
        plot_path = Path(plot_source)
        if plot_path.exists():
            # Store source path for direct copying
            bundle_data["plots"][panel_id] = str(plot_path)

    return save(bundle_data, p, bundle_type=BundleType.FIGURE, as_zip=as_zip)


def load_figure(path):
    """
    Load a .figure bundle.

    Parameters
    ----------
    path : str or Path
        Path to .figure bundle (directory or ZIP).

    Returns
    -------
    dict
        Figure data with:
        - 'spec': Figure specification
        - 'panels': Dict mapping panel IDs to {'spec': ..., 'data': ...}

    Examples
    --------
    >>> figure = scitex.canvas.load_figure("Figure1.figure")
    >>> print(figure['spec']['figure']['title'])
    >>> panel_a = figure['panels']['A']
    >>> print(panel_a['spec'], panel_a['data'])
    """
    from scitex.io.bundle import load

    bundle = load(path)

    if bundle["type"] != "figure":
        raise ValueError(f"Not a .figure bundle: {path}")

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
        "schema": {"name": "scitex.canvas.figure", "version": "1.0.0"},
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

        # Note: save_bundle uses panel_id for the directory name (e.g., A.plot)
        spec["panels"].append(
            {
                "id": panel_id,
                "label": panel_id,
                "caption": "",
                "plot": f"{panel_id}.plot",
                "position": {"x_mm": x, "y_mm": y},
                "size": {"width_mm": panel_w, "height_mm": panel_h},
            }
        )

    return spec


__all__ = [
    # Canvas class
    "Canvas",
    # Submodules (advanced)
    "io",
    "model",
    "backend",
    "utils",
    "editor",
    # Canvas operations
    "create_canvas",
    "get_canvas_path",
    "canvas_exists",
    "list_canvases",
    "delete_canvas",
    # Panel operations
    "add_panel",
    "update_panel",
    "remove_panel",
    "list_panels",
    # Export
    "export_canvas",
    # Data integrity
    "verify_data",
    # Editor
    "edit",
    # .figure bundle
    "save_figure",
    "load_figure",
]

# EOF
