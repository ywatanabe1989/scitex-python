#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: ./src/scitex/vis/io/data.py
"""
Data operations for scitex.fig.

Handles SHA256 hash computation and verification for data integrity.
"""

from pathlib import Path
from typing import Dict, Union
import hashlib


class HashMismatchError(Exception):
    """Raised when a data file hash doesn't match expected value."""

    pass


def compute_file_hash(filepath: Union[str, Path]) -> str:
    """
    Compute SHA256 hash of a file.

    Parameters
    ----------
    filepath : str or Path
        Path to file

    Returns
    -------
    str
        Hash string in format "sha256:{hex_digest}"
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    sha256_hash = hashlib.sha256()

    with open(filepath, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)

    return f"sha256:{sha256_hash.hexdigest()}"


def verify_data_hash(
    filepath: Union[str, Path],
    expected_hash: str,
) -> bool:
    """
    Verify file hash against expected value.

    Parameters
    ----------
    filepath : str or Path
        Path to file
    expected_hash : str
        Expected hash string (format: "sha256:{hex_digest}")

    Returns
    -------
    bool
        True if hash matches, False otherwise
    """
    try:
        actual_hash = compute_file_hash(filepath)
        return actual_hash == expected_hash
    except FileNotFoundError:
        return False


def verify_all_data_hashes(
    project_dir: Union[str, Path],
    canvas_name: str,
) -> Dict[str, bool]:
    """
    Verify hashes of all data files in a canvas.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Canvas name

    Returns
    -------
    Dict[str, bool]
        Dictionary mapping file paths to validation results
    """
    from ._directory import get_canvas_directory_path
    from ._canvas import load_canvas_json

    canvas_dir = get_canvas_directory_path(project_dir, canvas_name)

    # Load canvas.json without hash verification to avoid recursion
    json_path = canvas_dir / "canvas.json"
    if not json_path.exists():
        return {}

    import json

    with open(json_path, "r") as f:
        canvas_json = json.load(f)

    results = {}
    for data_file in canvas_json.get("data_files", []):
        rel_path = data_file.get("path", "")
        expected_hash = data_file.get("hash", "")

        if rel_path and expected_hash:
            filepath = canvas_dir / rel_path
            results[rel_path] = verify_data_hash(filepath, expected_hash)

    return results


def update_data_hash(
    project_dir: Union[str, Path],
    canvas_name: str,
    rel_path: str,
) -> str:
    """
    Update hash for a data file in canvas.json.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Canvas name
    rel_path : str
        Relative path to data file within canvas directory

    Returns
    -------
    str
        New hash value
    """
    from ._directory import get_canvas_directory_path
    from ._canvas import load_canvas_json, save_canvas_json

    canvas_dir = get_canvas_directory_path(project_dir, canvas_name)
    filepath = canvas_dir / rel_path

    new_hash = compute_file_hash(filepath)

    canvas_json = load_canvas_json(project_dir, canvas_name, verify_data_hashes=False)

    # Update or add data file entry
    found = False
    for data_file in canvas_json.get("data_files", []):
        if data_file.get("path") == rel_path:
            data_file["hash"] = new_hash
            found = True
            break

    if not found:
        if "data_files" not in canvas_json:
            canvas_json["data_files"] = []
        canvas_json["data_files"].append(
            {
                "path": rel_path,
                "hash": new_hash,
            }
        )

    save_canvas_json(project_dir, canvas_name, canvas_json)

    return new_hash


def list_data_files(
    project_dir: Union[str, Path],
    canvas_name: str,
) -> Dict[str, str]:
    """
    List all data files and their hashes.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Canvas name

    Returns
    -------
    Dict[str, str]
        Dictionary mapping file paths to hash values
    """
    from ._canvas import load_canvas_json

    canvas_json = load_canvas_json(project_dir, canvas_name, verify_data_hashes=False)

    return {
        d.get("path", ""): d.get("hash", "")
        for d in canvas_json.get("data_files", [])
        if d.get("path")
    }


# EOF
