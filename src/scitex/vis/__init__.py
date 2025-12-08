#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: ./src/scitex/vis/__init__.py
"""
SciTeX Visualization Module (scitex.vis)

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
...                   position=(10, 10), size=(80, 60), label="A")
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
from . import io
from . import model
from . import backend
from . import utils
from . import editor

# Canvas class
from .canvas import Canvas

# =============================================================================
# Primary API (minimal, reusable, flexible)
# =============================================================================

# Canvas operations
from .io import (
    ensure_canvas_directory as create_canvas,
    get_canvas_directory_path as get_canvas_path,
    canvas_directory_exists as canvas_exists,
    list_canvas_directories as list_canvases,
    delete_canvas_directory as delete_canvas,
)

# Panel operations
from .io import (
    add_panel_from_scitex,
    add_panel_from_image,
    update_panel,
    remove_panel,
    list_panels,
)

# Export (usually handled by stx.io.save, but available for explicit use)
from .io import export_canvas_to_file as export_canvas

# Data integrity
from .io import verify_all_data_hashes as verify_data

# Editor
from .editor import edit


# =============================================================================
# Convenience wrapper for add_panel
# =============================================================================
def add_panel(
    parent_dir,
    canvas_name,
    panel_name,
    source,
    position=(0, 0),
    size=(50, 50),
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
    position : tuple
        (x_mm, y_mm) position on canvas
    size : tuple
        (width_mm, height_mm) panel size
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
        "position": {"x_mm": position[0], "y_mm": position[1]},
        "size": {"width_mm": size[0], "height_mm": size[1]},
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
]

# EOF
