#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: ./src/scitex/vis/io/panel.py
"""
Panel operations for scitex.canvas.

Handles adding, removing, updating, and listing panels within a canvas.
Panels can be either 'scitex' type (full stx.plt output) or 'image' type (static image).
"""

from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import shutil

from ._directory import get_canvas_directory_path
from ._canvas import load_canvas_json, save_canvas_json
from ._data import compute_file_hash


def _symlink_or_copy(source: Path, dest: Path, bundle: bool = False) -> None:
    """Create relative symlink or copy file based on bundle flag."""
    import os

    if dest.exists() or dest.is_symlink():
        dest.unlink()

    if bundle:
        shutil.copy2(source, dest)
    else:
        try:
            # Use relative symlink for portability
            rel_path = os.path.relpath(source.resolve(), dest.parent.resolve())
            dest.symlink_to(rel_path)
        except (OSError, ValueError):
            # Fallback to copy if symlink fails (e.g., Windows without admin)
            shutil.copy2(source, dest)


def _get_default_panel_properties() -> Dict[str, Any]:
    """Get default panel properties."""
    return {
        "position": {"x_mm": 0, "y_mm": 0},
        "size": {"width_mm": 50, "height_mm": 50},
        "z_index": 0,
        "rotation_deg": 0,
        "clip": {
            "enabled": False,
            "x_mm": 0,
            "y_mm": 0,
            "width_mm": None,
            "height_mm": None,
        },
        "opacity": 1.0,
        "flip_h": False,
        "flip_v": False,
        "visible": True,
        "label": {
            "text": "",
            "position": "top-left",
            "fontsize": 12,
            "fontweight": "bold",
        },
        "border": {
            "visible": False,
            "color": "#000000",
            "width_mm": 0.2,
        },
    }


def add_panel_from_scitex(
    project_dir: Union[str, Path],
    canvas_name: str,
    panel_name: str,
    source_png: Union[str, Path],
    source_json: Optional[Union[str, Path]] = None,
    source_csv: Optional[Union[str, Path]] = None,
    panel_properties: Optional[Dict[str, Any]] = None,
    bundle: bool = False,
) -> Path:
    """
    Add a panel from stx.plt output (scitex type).

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Canvas name
    panel_name : str
        Name for the new panel
    source_png : str or Path
        Path to source PNG file
    source_json : str or Path, optional
        Path to source JSON file (auto-detected if not provided)
    source_csv : str or Path, optional
        Path to source CSV file (auto-detected if not provided)
    panel_properties : Dict, optional
        Panel properties (position, size, etc.)
    bundle : bool, optional
        If True, copy files. If False (default), use symlinks.

    Returns
    -------
    Path
        Path to panel directory
    """
    source_png = Path(source_png)
    base_name = source_png.stem

    # Auto-detect json and csv if not provided
    if source_json is None:
        source_json = source_png.parent / f"{base_name}.json"
    if source_csv is None:
        source_csv = source_png.parent / f"{base_name}.csv"

    source_json = Path(source_json)
    source_csv = Path(source_csv)

    # Create panel directory
    canvas_dir = get_canvas_directory_path(project_dir, canvas_name)
    panel_dir = canvas_dir / "panels" / panel_name
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Use symlinks (default) or copy files based on bundle flag
    _symlink_or_copy(source_png.resolve(), panel_dir / "panel.png", bundle=bundle)
    if source_json.exists():
        _symlink_or_copy(source_json.resolve(), panel_dir / "panel.json", bundle=bundle)
    if source_csv.exists():
        _symlink_or_copy(source_csv.resolve(), panel_dir / "panel.csv", bundle=bundle)

    # Build panel entry
    panel_entry = _get_default_panel_properties()
    panel_entry["name"] = panel_name
    panel_entry["type"] = "scitex"

    if panel_properties:
        _deep_merge(panel_entry, panel_properties)

    # Update canvas.json
    canvas_json = load_canvas_json(project_dir, canvas_name, verify_data_hashes=False)

    # Remove existing panel with same name
    canvas_json["panels"] = [
        p for p in canvas_json["panels"] if p.get("name") != panel_name
    ]

    # Add new panel
    canvas_json["panels"].append(panel_entry)

    # Add data file reference with hash
    if source_csv.exists():
        csv_path = f"panels/{panel_name}/panel.csv"
        csv_hash = compute_file_hash(panel_dir / "panel.csv")
        # Remove existing reference
        canvas_json["data_files"] = [
            d for d in canvas_json.get("data_files", []) if d.get("path") != csv_path
        ]
        canvas_json["data_files"].append(
            {
                "path": csv_path,
                "hash": csv_hash,
            }
        )

    save_canvas_json(project_dir, canvas_name, canvas_json)

    return panel_dir


def add_panel_from_image(
    project_dir: Union[str, Path],
    canvas_name: str,
    panel_name: str,
    source_image: Union[str, Path],
    panel_properties: Optional[Dict[str, Any]] = None,
    bundle: bool = False,
) -> Path:
    """
    Add a panel from an image file (image type).

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Canvas name
    panel_name : str
        Name for the new panel
    source_image : str or Path
        Path to source image file (PNG, JPG, SVG)
    panel_properties : Dict, optional
        Panel properties (position, size, etc.)
    bundle : bool, optional
        If True, copy files. If False (default), use symlinks.

    Returns
    -------
    Path
        Path to panel directory
    """
    source_image = Path(source_image)
    suffix = source_image.suffix.lower()

    # Create panel directory
    canvas_dir = get_canvas_directory_path(project_dir, canvas_name)
    panel_dir = canvas_dir / "panels" / panel_name
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Use symlink (default) or copy based on bundle flag
    dest_name = f"panel{suffix}"
    _symlink_or_copy(source_image.resolve(), panel_dir / dest_name, bundle=bundle)

    # Build panel entry
    panel_entry = _get_default_panel_properties()
    panel_entry["name"] = panel_name
    panel_entry["type"] = "image"
    panel_entry["source"] = dest_name

    if panel_properties:
        _deep_merge(panel_entry, panel_properties)

    # Update canvas.json
    canvas_json = load_canvas_json(project_dir, canvas_name, verify_data_hashes=False)

    # Remove existing panel with same name
    canvas_json["panels"] = [
        p for p in canvas_json["panels"] if p.get("name") != panel_name
    ]

    # Add new panel
    canvas_json["panels"].append(panel_entry)

    save_canvas_json(project_dir, canvas_name, canvas_json)

    return panel_dir


def remove_panel(
    project_dir: Union[str, Path],
    canvas_name: str,
    panel_name: str,
) -> bool:
    """
    Remove a panel from canvas.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Canvas name
    panel_name : str
        Name of panel to remove

    Returns
    -------
    bool
        True if removed, False if panel didn't exist
    """
    canvas_dir = get_canvas_directory_path(project_dir, canvas_name)
    panel_dir = canvas_dir / "panels" / panel_name

    # Remove from canvas.json
    canvas_json = load_canvas_json(project_dir, canvas_name, verify_data_hashes=False)

    original_count = len(canvas_json["panels"])
    canvas_json["panels"] = [
        p for p in canvas_json["panels"] if p.get("name") != panel_name
    ]

    # Remove data file references for this panel
    canvas_json["data_files"] = [
        d
        for d in canvas_json.get("data_files", [])
        if not d.get("path", "").startswith(f"panels/{panel_name}/")
    ]

    if len(canvas_json["panels"]) < original_count:
        save_canvas_json(project_dir, canvas_name, canvas_json)

        # Remove panel directory
        if panel_dir.exists():
            shutil.rmtree(panel_dir)

        return True

    return False


def update_panel(
    project_dir: Union[str, Path],
    canvas_name: str,
    panel_name: str,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Update panel properties.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Canvas name
    panel_name : str
        Name of panel to update
    updates : Dict[str, Any]
        Properties to update

    Returns
    -------
    Dict[str, Any]
        Updated panel entry

    Raises
    ------
    ValueError
        If panel not found
    """
    canvas_json = load_canvas_json(project_dir, canvas_name, verify_data_hashes=False)

    # Find panel
    panel_entry = None
    for panel in canvas_json["panels"]:
        if panel.get("name") == panel_name:
            panel_entry = panel
            break

    if panel_entry is None:
        raise ValueError(f"Panel not found: {panel_name}")

    # Apply updates
    _deep_merge(panel_entry, updates)

    save_canvas_json(project_dir, canvas_name, canvas_json)

    return panel_entry


def list_panels(
    project_dir: Union[str, Path],
    canvas_name: str,
) -> List[Dict[str, Any]]:
    """
    List all panels in a canvas.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Canvas name

    Returns
    -------
    List[Dict[str, Any]]
        List of panel entries
    """
    canvas_json = load_canvas_json(project_dir, canvas_name, verify_data_hashes=False)
    return canvas_json.get("panels", [])


def get_panel(
    project_dir: Union[str, Path],
    canvas_name: str,
    panel_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Get a specific panel by name.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Canvas name
    panel_name : str
        Name of panel

    Returns
    -------
    Optional[Dict[str, Any]]
        Panel entry or None if not found
    """
    panels = list_panels(project_dir, canvas_name)
    for panel in panels:
        if panel.get("name") == panel_name:
            return panel
    return None


def reorder_panels(
    project_dir: Union[str, Path],
    canvas_name: str,
    panel_order: List[str],
) -> None:
    """
    Reorder panels by z_index based on provided order.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Canvas name
    panel_order : List[str]
        List of panel names in desired z-order (first = bottom)
    """
    canvas_json = load_canvas_json(project_dir, canvas_name, verify_data_hashes=False)

    # Update z_index based on order
    for idx, panel_name in enumerate(panel_order):
        for panel in canvas_json["panels"]:
            if panel.get("name") == panel_name:
                panel["z_index"] = idx
                break

    save_canvas_json(project_dir, canvas_name, canvas_json)


def _deep_merge(base: Dict, updates: Dict) -> None:
    """Deep merge updates into base dictionary (in-place)."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# EOF
