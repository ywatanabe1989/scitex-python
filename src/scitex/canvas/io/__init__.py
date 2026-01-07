#!/usr/bin/env python3
# Timestamp: 2025-12-08
# File: ./src/scitex/vis/io/__init__.py
"""
I/O operations for scitex.canvas canvas management.

Canvas Architecture:
    project/scitex/vis/canvases/{canvas_name}.canvas/
        ├── canvas.json     # Layout, panels, composition settings
        ├── panels/         # Panel directories
        │   ├── panel_a/    # scitex type (PNG + JSON + CSV)
        │   └── panel_b/    # image type (PNG/JPG/SVG only)
        └── exports/        # Composed outputs (PNG, PDF, SVG)

The .canvas extension makes directories self-documenting, portable, and detectable.

Schema Version: 2.0.0
"""

from ._bundle import (
    FIGURE_SCHEMA_SPEC,
    load_figure_bundle,
    save_figure_bundle,
    validate_figure_spec,
)
from ._canvas import (
    get_canvas_schema_version,
    load_canvas_json,
    save_canvas_json,
    update_canvas_json,
)
from ._data import (
    HashMismatchError,
    compute_file_hash,
    list_data_files,
    update_data_hash,
    verify_all_data_hashes,
    verify_data_hash,
)
from ._directory import (
    CANVAS_EXTENSION,
    SCHEMA_VERSION,
    canvas_directory_exists,
    delete_canvas_directory,
    ensure_canvas_directory,
    get_canvas_directory_path,
    list_canvas_directories,
)
from ._export import (
    export_canvas_to_file,
    export_canvas_to_multiple_formats,
    list_canvas_exports,
)
from ._panel import (
    add_panel_from_image,
    add_panel_from_scitex,
    get_panel,
    list_panels,
    remove_panel,
    reorder_panels,
    update_panel,
)

__all__ = [
    # Schema
    "SCHEMA_VERSION",
    "CANVAS_EXTENSION",
    # Directory operations
    "ensure_canvas_directory",
    "get_canvas_directory_path",
    "list_canvas_directories",
    "delete_canvas_directory",
    "canvas_directory_exists",
    # Canvas operations
    "save_canvas_json",
    "load_canvas_json",
    "update_canvas_json",
    "get_canvas_schema_version",
    # Panel operations
    "add_panel_from_scitex",
    "add_panel_from_image",
    "remove_panel",
    "update_panel",
    "list_panels",
    "get_panel",
    "reorder_panels",
    # Data operations
    "HashMismatchError",
    "compute_file_hash",
    "verify_data_hash",
    "verify_all_data_hashes",
    "update_data_hash",
    "list_data_files",
    # Export operations
    "export_canvas_to_file",
    "export_canvas_to_multiple_formats",
    "list_canvas_exports",
    # Bundle operations
    "validate_figure_spec",
    "load_figure_bundle",
    "save_figure_bundle",
    "FIGURE_SCHEMA_SPEC",
]

# EOF
