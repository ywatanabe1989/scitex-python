#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_load_modules/_canvas.py
"""
Load canvas directory (.canvas) for scitex.canvas.

Canvas directories are portable figure bundles containing:
    - canvas.json: Layout, panels, composition settings
    - panels/: Panel directories (scitex or image type)
    - exports/: Composed outputs (PNG, PDF, SVG)

Usage:
    >>> import scitex as stx
    >>> # Load canvas from directory
    >>> canvas = stx.io.load("/path/to/fig1_results.canvas")
    >>> # Access canvas properties
    >>> print(canvas["canvas_name"])
    >>> print(canvas["panels"])
"""

import json
from pathlib import Path
from typing import Any, Dict, Union


def _load_canvas(
    lpath: Union[str, Path],
    verbose: bool = False,
    load_panels: bool = False,
    as_dict: bool = False,
    **kwargs,
) -> Any:
    """
    Load a canvas from a .canvas directory.

    Parameters
    ----------
    lpath : str or Path
        Path to the .canvas directory.
    verbose : bool, optional
        If True, print verbose output. Default is False.
    load_panels : bool, optional
        If True, also load panel images as numpy arrays.
        If False (default), only load canvas.json metadata.
    as_dict : bool, optional
        If True, return raw dict instead of Canvas object.
        Default is False (returns Canvas object).
    **kwargs
        Additional arguments (reserved for future use).

    Returns
    -------
    Canvas or Dict[str, Any]
        Canvas object (default) or dictionary if as_dict=True.
        Contains:
        - All fields from canvas.json
        - '_canvas_dir': Path to the canvas directory
        - If load_panels=True, panel images are loaded into memory

    Raises
    ------
    FileNotFoundError
        If the .canvas directory or canvas.json doesn't exist.
    ValueError
        If the path doesn't appear to be a valid canvas directory.
    """
    lpath = Path(lpath)

    # Validate path
    if not str(lpath).endswith(".canvas"):
        raise ValueError(f"Canvas path must end with .canvas extension: {lpath}")

    if not lpath.exists():
        raise FileNotFoundError(f"Canvas directory not found: {lpath}")

    if not lpath.is_dir():
        raise ValueError(f"Canvas path must be a directory: {lpath}")

    json_path = lpath / "canvas.json"
    if not json_path.exists():
        raise FileNotFoundError(f"canvas.json not found in canvas directory: {lpath}")

    # Load canvas.json
    with open(json_path, "r") as f:
        canvas_dict = json.load(f)

    # Add reference to the canvas directory
    canvas_dict["_canvas_dir"] = str(lpath)

    if verbose:
        print(f"Loaded canvas: {canvas_dict.get('canvas_name', 'unknown')}")
        print(f"  Schema version: {canvas_dict.get('schema_version', 'unknown')}")
        print(f"  Panels: {len(canvas_dict.get('panels', []))}")

    # Optionally load panel images
    if load_panels:
        _load_panel_images(lpath, canvas_dict, verbose=verbose)

    # Return Canvas object by default
    if not as_dict:
        try:
            from scitex.canvas.canvas import Canvas

            canvas_obj = Canvas.from_dict(canvas_dict)
            # Store reference to original directory for copying
            canvas_obj._canvas_dir = str(lpath)
            return canvas_obj
        except ImportError:
            # Fall back to dict if Canvas class unavailable
            pass

    return canvas_dict


def _load_panel_images(
    canvas_dir: Path,
    canvas_dict: Dict[str, Any],
    verbose: bool = False,
) -> None:
    """
    Load panel images into canvas_dict.

    Modifies canvas_dict in place, adding '_image' key to each panel
    containing the loaded numpy array.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        if verbose:
            print("PIL/numpy not available, skipping panel image loading")
        return

    panels_dir = canvas_dir / "panels"

    for panel in canvas_dict.get("panels", []):
        panel_name = panel.get("name", "")
        if not panel_name:
            continue

        panel_dir = panels_dir / panel_name
        if not panel_dir.exists():
            continue

        # Try to find panel image
        panel_type = panel.get("type", "image")
        if panel_type == "scitex":
            img_path = panel_dir / "panel.png"
        else:
            # For image type, use source filename
            source = panel.get("source", "panel.png")
            img_path = panel_dir / source

        if img_path.exists():
            try:
                img = Image.open(img_path)
                panel["_image"] = np.array(img)
                if verbose:
                    print(f"  Loaded panel image: {panel_name}")
            except Exception as e:
                if verbose:
                    print(f"  Failed to load panel image {panel_name}: {e}")


# EOF
