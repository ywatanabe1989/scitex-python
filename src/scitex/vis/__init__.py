#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/__init__.py
"""
SciTeX Visualization Module (scitex.vis)

A structured approach to creating publication-quality figures through JSON specifications.
This module completes the SciTeX ecosystem as the third pillar alongside scholar and writer.

Architecture:
- model: JSON data models (FigureModel, AxesModel, PlotModel, etc.)
- backend: JSON → matplotlib rendering engine
- io: Load/save figure JSON with project structure support
- utils: Validation and publication format templates

Quick Start:
-----------
>>> import scitex as stx
>>>
>>> # Using a template
>>> fig_json = stx.vis.utils.get_template("nature_single", height_mm=100)
>>> fig_json["axes"] = [...]  # Add axes configurations
>>>
>>> # Save the spec
>>> stx.vis.io.save_figure_json(fig_json, "figure.json")
>>>
>>> # Render to matplotlib
>>> fig, axes = stx.vis.backend.build_figure_from_json(fig_json)
>>>
>>> # Export to image
>>> stx.vis.backend.export_figure(fig_json, "figure.png", dpi=300)

With Project Structure:
----------------------
>>> # Save to project
>>> stx.vis.io.save_figure_json_to_project(
...     project_dir="/path/to/project",
...     figure_id="fig-001",
...     fig_json=fig_json
... )
>>>
>>> # Load from project
>>> loaded = stx.vis.io.load_figure_json_from_project(
...     project_dir="/path/to/project",
...     figure_id="fig-001"
... )
>>>
>>> # Export from project
>>> stx.vis.backend.export_figure(loaded, "output/fig-001.png")
"""

# Import submodules
from . import model
from . import backend
from . import io
from . import utils

# Convenient top-level imports for common use cases
from .model import FigureModel, AxesModel, PlotModel, GuideModel, AnnotationModel

from .backend import (
    build_figure_from_json,
    export_figure,
    export_figure_from_file,
    export_multiple_formats,
)

from .io import (
    load_figure_json,
    save_figure_json,
    load_figure_json_from_project,
    save_figure_json_to_project,
)

from .utils import (
    get_template,
    list_templates,
    NATURE_SINGLE_COLUMN_MM,
    NATURE_DOUBLE_COLUMN_MM,
)

__all__ = [
    # Submodules
    "model",
    "backend",
    "io",
    "utils",
    # Models
    "FigureModel",
    "AxesModel",
    "PlotModel",
    "GuideModel",
    "AnnotationModel",
    # Backend
    "build_figure_from_json",
    "export_figure",
    "export_figure_from_file",
    "export_multiple_formats",
    # I/O
    "load_figure_json",
    "save_figure_json",
    "load_figure_json_from_project",
    "save_figure_json_to_project",
    # Utils
    "get_template",
    "list_templates",
    "NATURE_SINGLE_COLUMN_MM",
    "NATURE_DOUBLE_COLUMN_MM",
]

# EOF
