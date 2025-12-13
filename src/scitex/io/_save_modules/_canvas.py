#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_canvas.py
"""
Save canvas directory (.canvas) for scitex.fig.

Canvas directories are portable figure bundles containing:
    - canvas.json: Layout, panels, composition settings
    - panels/: Panel directories (scitex or image type)
    - exports/: Composed outputs (PNG, PDF, SVG)

Usage:
    >>> import scitex as stx
    >>> # Create canvas object
    >>> canvas = stx.vis.Canvas(name="fig1_results")
    >>> canvas.add_panel("panel_a", "plot.png", ...)
    >>> # Save canvas to directory
    >>> stx.io.save(canvas, "/path/to/fig1_results.canvas")
    >>>
    >>> # Or save existing canvas directory to new location
    >>> stx.io.save(canvas_json_dict, "/path/to/new_location.canvas")
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Union


def save_canvas(
    obj: Any,
    spath: Union[str, Path],
    **kwargs,
) -> Path:
    """
    Save a canvas object or dictionary to a .canvas directory.

    Parameters
    ----------
    obj : Any
        Canvas object or dictionary containing canvas data.
        Can be:
        - Dict with canvas.json structure
        - Canvas object with to_dict() method
        - Path to existing .canvas directory (for copy/move)
    spath : str or Path
        Path where the .canvas directory should be created.
        Must end with .canvas extension.
    **kwargs
        Additional arguments (reserved for future use).

    Returns
    -------
    Path
        Path to the created .canvas directory.

    Raises
    ------
    ValueError
        If path doesn't end with .canvas extension.
    """
    spath = Path(spath)

    # Validate extension
    if not str(spath).endswith(".canvas"):
        raise ValueError(f"Canvas path must end with .canvas extension: {spath}")

    # Handle different object types
    if isinstance(obj, (str, Path)):
        # Source is an existing canvas directory - copy it
        _copy_canvas_directory(Path(obj), spath)
    elif isinstance(obj, dict):
        # Object is a canvas JSON dictionary
        _save_canvas_from_dict(obj, spath)
    elif hasattr(obj, "to_dict"):
        # Object has to_dict method (Canvas object)
        canvas_dict = obj.to_dict()
        # Check if this Canvas was loaded from disk (has _canvas_dir)
        if hasattr(obj, "_canvas_dir") and obj._canvas_dir:
            canvas_dict["_canvas_dir"] = obj._canvas_dir
        # Pass bundle option
        if "bundle" in kwargs:
            canvas_dict["_bundle"] = kwargs.pop("bundle")
        _save_canvas_from_dict(canvas_dict, spath)
    elif hasattr(obj, "_canvas_json"):
        # Object has internal canvas JSON (Canvas object variant)
        _save_canvas_from_dict(obj._canvas_json, spath)
    else:
        raise TypeError(
            f"Cannot save object of type {type(obj).__name__} as canvas. "
            "Expected dict, Canvas object, or path to existing canvas."
        )

    # Export figures to exports/ directory
    _export_canvas_figures(spath, **kwargs)

    return spath


def _export_canvas_figures(
    canvas_dir: Path,
    formats: list = None,
    dpi: int = 300,
    **kwargs,
) -> None:
    """
    Export canvas figures directly to canvas directory.

    Automatically exports PNG, PDF, and SVG formats.
    """
    if formats is None:
        formats = ["png", "pdf", "svg"]

    try:
        from scitex.fig.io.export import _compose_and_export
        import json

        # Load canvas.json
        json_path = canvas_dir / "canvas.json"
        if not json_path.exists():
            return

        with open(json_path, "r") as f:
            canvas_json = json.load(f)

        # Export directly to canvas directory (no exports/ subdirectory)
        for fmt in formats:
            output_path = canvas_dir / f"canvas.{fmt}"
            _compose_and_export(
                canvas_dir=canvas_dir,
                canvas_json=canvas_json,
                output_path=output_path,
                output_format=fmt,
                dpi=dpi,
                transparent=False,
            )
    except ImportError:
        # scitex.fig not available
        pass
    except Exception as e:
        # Log but don't fail save if export fails
        import sys

        print(f"Warning: Canvas export failed: {e}", file=sys.stderr)


def _copy_canvas_directory(source: Path, dest: Path) -> None:
    """Copy an existing canvas directory to a new location."""
    if not source.exists():
        raise FileNotFoundError(f"Source canvas directory not found: {source}")

    if not (source / "canvas.json").exists():
        raise ValueError(f"Invalid canvas directory (missing canvas.json): {source}")

    # Remove destination if exists
    if dest.exists():
        shutil.rmtree(dest)

    # Copy entire directory tree
    shutil.copytree(source, dest)


def _save_canvas_from_dict(canvas_dict: Dict[str, Any], dest: Path) -> None:
    """Create a canvas directory from a dictionary."""
    import json

    # Create directory structure (no exports/ - files go directly in canvas dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "panels").mkdir(exist_ok=True)

    # Check if this dict was loaded from an existing canvas (has _canvas_dir)
    source_canvas_dir = canvas_dict.get("_canvas_dir")

    # Check if this dict has source files (from Canvas object)
    source_files = canvas_dict.get("_source_files", {})

    # Get bundle option
    bundle = canvas_dict.get("_bundle", False)

    # Create a clean copy of canvas_dict without internal keys for saving
    save_dict = {k: v for k, v in canvas_dict.items() if not k.startswith("_")}

    # Save canvas.json
    json_path = dest / "canvas.json"
    with open(json_path, "w") as f:
        json.dump(save_dict, f, indent=2, default=str)

    # If canvas_dict was loaded from an existing canvas, copy panel files
    if source_canvas_dir:
        _copy_panels_from_source(Path(source_canvas_dir), dest, canvas_dict)

    # If canvas_dict has source files (from Canvas object), create panel dirs
    if source_files:
        _create_panels_from_source_files(source_files, dest, canvas_dict, bundle=bundle)

    # If canvas_dict contains embedded panel data, extract it
    _extract_embedded_panels(canvas_dict, dest)


def _copy_panels_from_source(
    source_canvas_dir: Path,
    dest: Path,
    canvas_dict: Dict[str, Any],
) -> None:
    """
    Copy panel files from source canvas directory to destination.

    When a canvas dict was loaded from an existing canvas directory,
    this function copies the panel files to the new location.
    Skips copying if source and destination are the same.
    """
    source_panels_dir = source_canvas_dir / "panels"
    dest_panels_dir = dest / "panels"

    if not source_panels_dir.exists():
        return

    # Skip if source and dest are the same (saving back to same location)
    try:
        if source_canvas_dir.resolve() == dest.resolve():
            return
    except (OSError, ValueError):
        pass

    for panel in canvas_dict.get("panels", []):
        panel_name = panel.get("name", "")
        if not panel_name:
            continue

        source_panel_dir = source_panels_dir / panel_name
        dest_panel_dir = dest_panels_dir / panel_name

        if source_panel_dir.exists() and source_panel_dir.is_dir():
            # Copy entire panel directory (follow symlinks to get actual content)
            if dest_panel_dir.exists():
                shutil.rmtree(dest_panel_dir)
            shutil.copytree(source_panel_dir, dest_panel_dir, symlinks=False)


def _create_panels_from_source_files(
    source_files: Dict[str, str],
    dest: Path,
    canvas_dict: Dict[str, Any],
    bundle: bool = False,
) -> None:
    """
    Create panel directories from source files.

    When a Canvas object is saved, this creates the panel directories
    with symlinks (default) or copies of the source files.

    Parameters
    ----------
    source_files : Dict[str, str]
        Mapping of panel_name -> source_file_path
    dest : Path
        Destination canvas directory
    canvas_dict : Dict[str, Any]
        Canvas dictionary (to get panel types)
    bundle : bool
        If True, copy files. If False (default), create symlinks.
    """
    dest_panels_dir = dest / "panels"

    for panel in canvas_dict.get("panels", []):
        panel_name = panel.get("name", "")
        if not panel_name or panel_name not in source_files:
            continue

        source_path = Path(source_files[panel_name])
        if not source_path.exists():
            continue

        panel_type = panel.get("type", "image")
        panel_dir = dest_panels_dir / panel_name
        panel_dir.mkdir(parents=True, exist_ok=True)

        # Symlink or copy panel files
        if panel_type == "scitex":
            # Scitex panel: PNG, JSON, CSV
            _link_or_copy(source_path, panel_dir / "panel.png", bundle)
            json_sibling = source_path.parent / f"{source_path.stem}.json"
            if json_sibling.exists():
                _link_or_copy(json_sibling, panel_dir / "panel.json", bundle)
            csv_sibling = source_path.parent / f"{source_path.stem}.csv"
            if csv_sibling.exists():
                _link_or_copy(csv_sibling, panel_dir / "panel.csv", bundle)
        else:
            # Image panel: just the image
            dest_name = f"panel{source_path.suffix}"
            _link_or_copy(source_path, panel_dir / dest_name, bundle)


def _link_or_copy(source: Path, dest: Path, bundle: bool = False) -> None:
    """Create relative symlink or copy file based on bundle flag."""
    if dest.exists() or dest.is_symlink():
        dest.unlink()

    if bundle:
        shutil.copy2(source, dest)
    else:
        try:
            # Use relative symlink for portability
            import os

            rel_path = os.path.relpath(source.resolve(), dest.parent.resolve())
            dest.symlink_to(rel_path)
        except (OSError, ValueError):
            # Fallback to copy if symlink fails
            shutil.copy2(source, dest)


def _extract_embedded_panels(canvas_dict: Dict[str, Any], dest: Path) -> None:
    """
    Extract embedded panel data from canvas dictionary.

    Some Canvas objects may embed panel image data (base64) in the dict.
    This function extracts them to the panels/ directory.
    """
    import base64

    for panel in canvas_dict.get("panels", []):
        panel_name = panel.get("name", "")
        if not panel_name:
            continue

        panel_dir = dest / "panels" / panel_name
        panel_dir.mkdir(parents=True, exist_ok=True)

        # Check for embedded image data
        if "image_data" in panel:
            # Decode base64 image data
            img_data = base64.b64decode(panel["image_data"])
            img_ext = panel.get("image_ext", "png")
            img_path = panel_dir / f"panel.{img_ext}"
            with open(img_path, "wb") as f:
                f.write(img_data)

        # Check for embedded JSON data (scitex type panels)
        if "panel_json" in panel:
            import json

            json_path = panel_dir / "panel.json"
            with open(json_path, "w") as f:
                json.dump(panel["panel_json"], f, indent=2, default=str)

        # Check for embedded CSV data (scitex type panels)
        if "panel_csv" in panel:
            csv_path = panel_dir / "panel.csv"
            with open(csv_path, "w") as f:
                f.write(panel["panel_csv"])


# EOF
