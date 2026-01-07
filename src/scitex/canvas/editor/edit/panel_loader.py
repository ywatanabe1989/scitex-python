#!/usr/bin/env python3
# Timestamp: "2025-12-14 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/editor/edit/panel_loader.py

"""Panel data loading for figure editor."""

import io
import json as json_module
from pathlib import Path
from typing import Any, Dict, Optional, Union

__all__ = ["load_panel_data"]


def load_panel_data(
    panel_path: Union[Path, str], is_zip: bool = None
) -> Optional[Dict[str, Any]]:
    """
    Load panel data from either a .plot directory or a .plot.zip file.

    Handles both formats transparently using in-memory reading for zips.

    Parameters
    ----------
    panel_path : Path or str
        Path to .plot directory or .plot.zip file
    is_zip : bool, optional
        If True, treat as zip file. If False, treat as directory.
        If None, auto-detect based on path suffix and existence.

    Returns
    -------
    dict or None
        Dictionary with keys: metadata, csv_data, png_bytes, hitmap_bytes, img_size
        For directories, also includes: json_path, png_path, hitmap_path
        Returns None if panel cannot be loaded
    """

    panel_path = Path(panel_path)

    # Auto-detect if not specified
    if is_zip is None:
        spath = str(panel_path)
        if spath.endswith(".plot.zip"):
            is_zip = panel_path.is_file()
        else:
            is_zip = False

    if is_zip:
        return _load_from_zip(panel_path)
    else:
        return _load_from_directory(panel_path)


def _load_from_zip(panel_path: Path) -> Optional[Dict[str, Any]]:
    """Load panel data from a .plot.zip file."""
    from PIL import Image

    from scitex.io.bundle import ZipBundle

    if not panel_path.exists():
        return None

    try:
        with ZipBundle(panel_path, mode="r") as zb:
            # Load spec.json for metadata
            try:
                metadata = zb.read_json("spec.json")
            except FileNotFoundError:
                metadata = {}

            # Load style.json if exists
            try:
                style = zb.read_json("style.json")
                metadata["style"] = style
            except FileNotFoundError:
                pass

            # Find and read PNG
            png_bytes = None
            for name in zb.namelist():
                if (
                    name.endswith(".png")
                    and "_hitmap" not in name
                    and "_overview" not in name
                ):
                    if "exports/" in name:
                        png_bytes = zb.read_bytes(name)
                        break

            # If no PNG in exports/, try root level
            if not png_bytes:
                for name in zb.namelist():
                    if (
                        name.endswith(".png")
                        and "_hitmap" not in name
                        and "_overview" not in name
                    ):
                        png_bytes = zb.read_bytes(name)
                        break

            # Get image size
            img_size = None
            if png_bytes:
                img = Image.open(io.BytesIO(png_bytes))
                img_size = {"width": img.size[0], "height": img.size[1]}
                img.close()

            # Find and read hitmap
            hitmap_bytes = None
            for name in zb.namelist():
                if "_hitmap.png" in name:
                    hitmap_bytes = zb.read_bytes(name)
                    break

            # Load geometry_px.json if available
            geometry_data = None
            try:
                geometry_data = zb.read_json("cache/geometry_px.json")
            except FileNotFoundError:
                pass

            return {
                "metadata": metadata,
                "png_bytes": png_bytes,
                "hitmap_bytes": hitmap_bytes,
                "img_size": img_size,
                "geometry_data": geometry_data,
                "is_zip": True,
            }
    except Exception as e:
        print(f"Error loading panel zip {panel_path}: {e}")
        return None


def _load_from_directory(panel_path: Path) -> Optional[Dict[str, Any]]:
    """Load panel data from a .plot directory."""

    panel_dir = panel_path
    if not panel_dir.exists():
        return None

    # Check for layered vs legacy format
    spec_path = panel_dir / "spec.json"
    if spec_path.exists():
        return _load_layered_directory(panel_dir)
    else:
        return _load_legacy_directory(panel_dir)


def _load_layered_directory(panel_dir: Path) -> Dict[str, Any]:
    """Load panel data from layered format directory."""
    import scitex as stx
    from scitex.plt.io import load_layered_plot_bundle

    bundle_data = load_layered_plot_bundle(panel_dir)
    metadata = bundle_data.get("merged", {})

    # Find CSV
    csv_data = None
    for f in panel_dir.glob("*.csv"):
        csv_data = stx.io.load(f)
        break

    # Find exports - prefer PNG over SVG (PIL can't open SVG)
    png_path = None
    svg_path = None
    hitmap_path = None
    exports_dir = panel_dir / "exports"
    if exports_dir.exists():
        for f in exports_dir.iterdir():
            name = f.name
            if name.endswith("_hitmap.png"):
                hitmap_path = f
            elif (
                name.endswith(".png")
                and "_hitmap" not in name
                and "_overview" not in name
            ):
                png_path = f
            elif name.endswith(".svg") and "_hitmap" not in name and svg_path is None:
                svg_path = f

    if png_path is None:
        png_path = svg_path

    # Load geometry_px.json if available
    geometry_data = None
    geometry_path = panel_dir / "cache" / "geometry_px.json"
    if geometry_path.exists():
        with open(geometry_path) as f:
            geometry_data = json_module.load(f)

    return {
        "json_path": panel_dir / "spec.json",
        "metadata": metadata,
        "csv_data": csv_data,
        "png_path": png_path,
        "hitmap_path": hitmap_path,
        "geometry_data": geometry_data,
        "is_zip": False,
    }


def _load_legacy_directory(panel_dir: Path) -> Optional[Dict[str, Any]]:
    """Load panel data from legacy format directory."""
    import scitex as stx

    json_path = None
    csv_data = None
    png_path = None
    hitmap_path = None

    for f in panel_dir.iterdir():
        name = f.name
        if name.endswith(".json") and not name.endswith(".manual.json"):
            json_path = f
        elif name.endswith(".csv"):
            csv_data = stx.io.load(f)
        elif name.endswith("_hitmap.png"):
            hitmap_path = f
        elif name.endswith(".svg") and "_hitmap" not in name:
            png_path = f
        elif (
            name.endswith(".png") and "_hitmap" not in name and "_overview" not in name
        ):
            if png_path is None:
                png_path = f

    if json_path is None:
        return None

    with open(json_path) as f:
        metadata = json_module.load(f)

    return {
        "json_path": json_path,
        "metadata": metadata,
        "csv_data": csv_data,
        "png_path": png_path,
        "hitmap_path": hitmap_path,
        "is_zip": False,
    }


# EOF
