#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: ./src/scitex/vis/io/directory.py
"""
Directory operations for canvas storage.

Canvas directories use .canvas extension for portability and distinguishability:
    {parent_dir}/{canvas_name}.canvas/
        ├── canvas.json
        ├── panels/
        └── exports/

The .canvas extension makes canvas directories:
- Self-documenting (clearly a canvas bundle)
- Portable (can be moved/copied as a unit)
- Detectable by scitex.io
"""

from pathlib import Path
from typing import List, Union
import shutil


SCHEMA_VERSION = "2.0.0"
CANVAS_EXTENSION = ".canvas"


def _normalize_canvas_name(canvas_name: str) -> str:
    """Ensure canvas_name has .canvas extension."""
    if not canvas_name.endswith(CANVAS_EXTENSION):
        return canvas_name + CANVAS_EXTENSION
    return canvas_name


def _strip_canvas_extension(canvas_name: str) -> str:
    """Remove .canvas extension from canvas_name."""
    if canvas_name.endswith(CANVAS_EXTENSION):
        return canvas_name[: -len(CANVAS_EXTENSION)]
    return canvas_name


def ensure_canvas_directory(
    parent_dir: Union[str, Path],
    canvas_name: str,
) -> Path:
    """
    Create canvas directory structure if not exists.

    Creates:
        - {parent_dir}/{canvas_name}.canvas/
        - {parent_dir}/{canvas_name}.canvas/panels/
        - {parent_dir}/{canvas_name}.canvas/exports/
        - {parent_dir}/{canvas_name}.canvas/canvas.json (empty template if not exists)

    Parameters
    ----------
    parent_dir : str or Path
        Parent directory where canvas will be created
    canvas_name : str
        Descriptive canvas name (e.g., "fig1_neural_results")
        .canvas extension is added automatically if not present

    Returns
    -------
    Path
        Path to canvas directory (e.g., .../fig1_neural_results.canvas/)
    """
    canvas_dir = get_canvas_directory_path(parent_dir, canvas_name)

    # Create directory structure (exports go directly in canvas dir)
    canvas_dir.mkdir(parents=True, exist_ok=True)
    (canvas_dir / "panels").mkdir(exist_ok=True)

    # Create empty canvas.json if not exists
    json_path = canvas_dir / "canvas.json"
    if not json_path.exists():
        from ._canvas import _get_empty_canvas_template
        import json

        # Use the name without extension for canvas_name in JSON
        base_name = _strip_canvas_extension(canvas_name)
        template = _get_empty_canvas_template(base_name)
        with open(json_path, "w") as f:
            json.dump(template, f, indent=2)

    return canvas_dir


def get_canvas_directory_path(
    parent_dir: Union[str, Path],
    canvas_name: str,
) -> Path:
    """
    Get path to canvas directory.

    Parameters
    ----------
    parent_dir : str or Path
        Parent directory containing the canvas
    canvas_name : str
        Descriptive canvas name (with or without .canvas extension)

    Returns
    -------
    Path
        Path to {parent_dir}/{canvas_name}.canvas/
    """
    normalized_name = _normalize_canvas_name(canvas_name)
    return Path(parent_dir) / normalized_name


def list_canvas_directories(
    parent_dir: Union[str, Path],
    include_extension: bool = False,
) -> List[str]:
    """
    List all canvas directory names in parent directory.

    Parameters
    ----------
    parent_dir : str or Path
        Directory to search for canvas directories
    include_extension : bool, optional
        If True, return names with .canvas extension.
        If False (default), return names without extension.

    Returns
    -------
    List[str]
        List of canvas_names
    """
    parent_dir = Path(parent_dir)

    if not parent_dir.exists():
        return []

    # Find .canvas directories that contain canvas.json
    canvas_names = []
    for item in sorted(parent_dir.iterdir()):
        if (
            item.is_dir()
            and item.name.endswith(CANVAS_EXTENSION)
            and (item / "canvas.json").exists()
        ):
            if include_extension:
                canvas_names.append(item.name)
            else:
                canvas_names.append(_strip_canvas_extension(item.name))

    return canvas_names


def delete_canvas_directory(
    parent_dir: Union[str, Path],
    canvas_name: str,
) -> bool:
    """
    Delete canvas directory and all contents.

    Parameters
    ----------
    parent_dir : str or Path
        Parent directory containing the canvas
    canvas_name : str
        Descriptive canvas name

    Returns
    -------
    bool
        True if deleted successfully, False if directory didn't exist
    """
    canvas_dir = get_canvas_directory_path(parent_dir, canvas_name)

    if not canvas_dir.exists():
        return False

    shutil.rmtree(canvas_dir)
    return True


def canvas_directory_exists(
    parent_dir: Union[str, Path],
    canvas_name: str,
) -> bool:
    """
    Check if canvas directory exists.

    Parameters
    ----------
    parent_dir : str or Path
        Parent directory containing the canvas
    canvas_name : str
        Descriptive canvas name

    Returns
    -------
    bool
        True if canvas directory exists with canvas.json
    """
    canvas_dir = get_canvas_directory_path(parent_dir, canvas_name)
    return canvas_dir.exists() and (canvas_dir / "canvas.json").exists()


# EOF
