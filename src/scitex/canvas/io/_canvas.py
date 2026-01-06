#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: ./src/scitex/vis/io/canvas.py
"""
Canvas JSON operations for scitex.canvas.

Handles saving, loading, and updating canvas.json files.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
from datetime import datetime
import json

from ._directory import get_canvas_directory_path, SCHEMA_VERSION


def _get_empty_canvas_template(canvas_name: str) -> Dict[str, Any]:
    """
    Get empty canvas.json template.

    Parameters
    ----------
    canvas_name : str
        Name of the canvas

    Returns
    -------
    Dict[str, Any]
        Empty canvas template with schema version
    """
    now = datetime.utcnow().isoformat() + "Z"

    return {
        "schema_version": SCHEMA_VERSION,
        "canvas_name": canvas_name,
        "size": {
            "width_mm": 180,
            "height_mm": 240,
        },
        "background": {
            "color": "#ffffff",
            "grid": False,
        },
        "panels": [],
        "annotations": [],
        "title": {
            "text": "",
            "position": {"x_mm": 90, "y_mm": 5},
            "fontsize": 14,
        },
        "data_files": [],
        "metadata": {
            "created_at": now,
            "updated_at": now,
            "author": "",
            "description": "",
        },
        "manual_overrides": {},
    }


def save_canvas_json(
    project_dir: Union[str, Path],
    canvas_name: str,
    canvas_json: Dict[str, Any],
) -> Path:
    """
    Save canvas specification to canvas.json.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Descriptive canvas name
    canvas_json : Dict[str, Any]
        Canvas specification dictionary

    Returns
    -------
    Path
        Path to saved canvas.json file
    """
    canvas_dir = get_canvas_directory_path(project_dir, canvas_name)
    json_path = canvas_dir / "canvas.json"

    # Ensure directory exists
    canvas_dir.mkdir(parents=True, exist_ok=True)

    # Update metadata timestamp
    if "metadata" not in canvas_json:
        canvas_json["metadata"] = {}
    canvas_json["metadata"]["updated_at"] = datetime.utcnow().isoformat() + "Z"

    # Ensure schema version is set
    if "schema_version" not in canvas_json:
        canvas_json["schema_version"] = SCHEMA_VERSION

    with open(json_path, "w") as f:
        json.dump(canvas_json, f, indent=2)

    return json_path


def load_canvas_json(
    project_dir: Union[str, Path],
    canvas_name: str,
    verify_data_hashes: bool = True,
) -> Dict[str, Any]:
    """
    Load canvas specification from canvas.json.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Descriptive canvas name
    verify_data_hashes : bool, optional
        If True, verify all data file hashes (default: True)

    Returns
    -------
    Dict[str, Any]
        Canvas specification dictionary

    Raises
    ------
    FileNotFoundError
        If canvas.json does not exist
    HashMismatchError
        If verify_data_hashes=True and any data file hash doesn't match
    """
    canvas_dir = get_canvas_directory_path(project_dir, canvas_name)
    json_path = canvas_dir / "canvas.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Canvas not found: {json_path}")

    with open(json_path, "r") as f:
        canvas_json = json.load(f)

    # Verify data hashes if requested
    if verify_data_hashes and canvas_json.get("data_files"):
        from ._data import verify_all_data_hashes

        hash_results = verify_all_data_hashes(project_dir, canvas_name)
        invalid_files = [f for f, valid in hash_results.items() if not valid]
        if invalid_files:
            from ._data import HashMismatchError

            raise HashMismatchError(
                f"Data file hash mismatch for: {', '.join(invalid_files)}"
            )

    return canvas_json


def update_canvas_json(
    project_dir: Union[str, Path],
    canvas_name: str,
    updates: Dict[str, Any],
) -> Path:
    """
    Partial update of canvas.json (merge with existing).

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Descriptive canvas name
    updates : Dict[str, Any]
        Dictionary of updates to merge

    Returns
    -------
    Path
        Path to updated canvas.json file
    """
    # Load existing
    canvas_json = load_canvas_json(project_dir, canvas_name, verify_data_hashes=False)

    # Deep merge updates
    _deep_merge(canvas_json, updates)

    # Save back
    return save_canvas_json(project_dir, canvas_name, canvas_json)


def _deep_merge(base: Dict, updates: Dict) -> None:
    """Deep merge updates into base dictionary (in-place)."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_canvas_schema_version(
    project_dir: Union[str, Path],
    canvas_name: str,
) -> Optional[str]:
    """
    Get schema version of a canvas.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Descriptive canvas name

    Returns
    -------
    Optional[str]
        Schema version string or None if not found
    """
    try:
        canvas_json = load_canvas_json(
            project_dir, canvas_name, verify_data_hashes=False
        )
        return canvas_json.get("schema_version")
    except FileNotFoundError:
        return None


# EOF
