#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-14 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/editor/edit/path_resolver.py

"""Basic path resolution for figure files."""

from pathlib import Path
from typing import Optional, Tuple

__all__ = ["resolve_figure_paths"]


def resolve_figure_paths(path: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """
    Resolve JSON, CSV, and PNG paths from any input file path.

    Handles two patterns:
    1. Flat (sibling): path/to/figure.{json,csv,png}
    2. Organized (subdirs): path/to/{json,csv,png}/figure.{ext}

    Parameters
    ----------
    path : Path
        Input path (can be JSON, CSV, or PNG)

    Returns
    -------
    tuple
        (json_path, csv_path, png_path) - csv_path/png_path may be None if not found
    """
    path = Path(path)
    stem = path.stem
    parent = path.parent

    # Check if this is organized pattern (parent is json/, csv/, png/)
    if parent.name in ("json", "csv", "png"):
        base_dir = parent.parent
        json_path = base_dir / "json" / f"{stem}.json"
        csv_path = base_dir / "csv" / f"{stem}.csv"
        png_path = base_dir / "png" / f"{stem}.png"
    else:
        # Flat pattern - sibling files
        json_path = parent / f"{stem}.json"
        csv_path = parent / f"{stem}.csv"
        png_path = parent / f"{stem}.png"

    # If input was .manual.json, get base json
    if stem.endswith(".manual"):
        base_stem = stem[:-7]  # Remove '.manual'
        if parent.name == "json":
            json_path = parent / f"{base_stem}.json"
            csv_path = parent.parent / "csv" / f"{base_stem}.csv"
            png_path = parent.parent / "png" / f"{base_stem}.png"
        else:
            json_path = parent / f"{base_stem}.json"
            csv_path = parent / f"{base_stem}.csv"
            png_path = parent / f"{base_stem}.png"

    return (
        json_path,
        csv_path if csv_path.exists() else None,
        png_path if png_path.exists() else None,
    )


# EOF
