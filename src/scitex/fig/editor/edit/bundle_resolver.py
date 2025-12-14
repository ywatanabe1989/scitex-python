#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-14 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/editor/edit/bundle_resolver.py

"""Bundle path resolution for .pltz and .figz formats."""

import json as json_module
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

__all__ = ["resolve_pltz_bundle", "resolve_figz_bundle", "resolve_layered_pltz_bundle"]


def resolve_figz_bundle(path: Path, panel_index: int = 0) -> Tuple:
    """
    Resolve paths from a .figz bundle (multi-panel figure).

    Uses in-memory zip reading for .pltz panels - no disk extraction.

    Parameters
    ----------
    path : Path
        Path to .figz bundle (.figz or .figz.d)
    panel_index : int, optional
        Index of panel to open (default: 0 for first panel)

    Returns
    -------
    tuple
        (json_path, csv_path, png_path, hitmap_path, bundle_spec, panel_info)
    """
    spath = str(path)
    figz_is_zip = False

    # Handle ZIP vs directory for figz
    if spath.endswith('.figz') and not spath.endswith('.figz.d'):
        figz_is_zip = True
        if not path.exists():
            raise FileNotFoundError(f"Figz bundle not found: {path}")
        # For figz zip, extract to access nested pltz
        temp_dir = tempfile.mkdtemp(prefix='scitex_edit_figz_')
        with zipfile.ZipFile(path, 'r') as zf:
            zf.extractall(temp_dir)
        bundle_dir = Path(temp_dir)
        for item in bundle_dir.iterdir():
            if item.is_dir() and str(item).endswith('.figz.d'):
                bundle_dir = item
                break
    else:
        bundle_dir = Path(path)
        if not bundle_dir.exists():
            raise FileNotFoundError(f"Figz bundle directory not found: {bundle_dir}")

    # Find nested pltz bundles
    panel_paths = []
    panel_is_zip = []

    for item in sorted(bundle_dir.iterdir(), key=lambda x: x.name):
        if item.is_dir() and str(item).endswith('.pltz.d'):
            panel_paths.append(str(item))
            panel_is_zip.append(False)
        elif item.is_file() and str(item).endswith('.pltz'):
            panel_paths.append(str(item))
            panel_is_zip.append(True)

    if not panel_paths:
        raise FileNotFoundError(f"No .pltz panels found in figz bundle: {bundle_dir}")

    # Validate panel index
    if panel_index < 0 or panel_index >= len(panel_paths):
        panel_index = 0

    selected_panel_path = panel_paths[panel_index]
    panel_name = Path(selected_panel_path).name
    print(f"Opening panel: {panel_name}")
    if len(panel_paths) > 1:
        print(f"  (Figz contains {len(panel_paths)} panels)")

    # Build panel info
    panel_names = [Path(p).name for p in panel_paths]
    panel_info = {
        "panels": panel_names,
        "panel_paths": panel_paths,
        "panel_is_zip": panel_is_zip,
        "current_index": panel_index,
        "figz_dir": str(bundle_dir),
        "figz_is_zip": figz_is_zip,
    }

    # Resolve the selected panel
    result = resolve_pltz_bundle(Path(selected_panel_path))
    return result + (panel_info,)


def resolve_pltz_bundle(path: Path) -> Tuple:
    """
    Resolve paths from a .pltz bundle (directory or ZIP).

    Supports both:
    - Legacy format (single {basename}.json)
    - Layered format v2.0 (spec.json + style.json + cache/)

    Parameters
    ----------
    path : Path
        Path to .pltz bundle (.pltz or .pltz.d)

    Returns
    -------
    tuple
        (json_path, csv_path, png_path, hitmap_path, bundle_spec)
    """
    spath = str(path)

    # Handle ZIP vs directory
    if spath.endswith('.pltz') and not spath.endswith('.pltz.d'):
        if not path.exists():
            raise FileNotFoundError(f"Bundle not found: {path}")
        temp_dir = tempfile.mkdtemp(prefix='scitex_edit_')
        with zipfile.ZipFile(path, 'r') as zf:
            zf.extractall(temp_dir)
        bundle_dir = Path(temp_dir)
        for item in bundle_dir.iterdir():
            if item.is_dir() and str(item).endswith('.pltz.d'):
                bundle_dir = item
                break
    else:
        bundle_dir = Path(path)
        if not bundle_dir.exists():
            raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    # Check if this is a layered bundle (v2.0)
    spec_path = bundle_dir / "spec.json"
    if spec_path.exists():
        return resolve_layered_pltz_bundle(bundle_dir)

    # === Legacy format ===
    json_path = None
    csv_path = None
    png_path = None
    svg_path = None
    hitmap_path = None
    bundle_spec = None

    for f in bundle_dir.iterdir():
        name = f.name
        if name.endswith('.json') and not name.endswith('.manual.json'):
            json_path = f
        elif name.endswith('.csv'):
            csv_path = f
        elif name.endswith('_hitmap.png'):
            hitmap_path = f
        elif name.endswith('.svg') and '_hitmap' not in name:
            svg_path = f
        elif name.endswith('.png') and '_hitmap' not in name and '_overview' not in name:
            png_path = f

    # Prefer SVG for display
    if svg_path:
        png_path = svg_path

    if json_path and json_path.exists():
        with open(json_path, 'r') as f:
            bundle_spec = json_module.load(f)

    return (
        json_path,
        csv_path if csv_path and csv_path.exists() else None,
        png_path if png_path and png_path.exists() else None,
        hitmap_path if hitmap_path and hitmap_path.exists() else None,
        bundle_spec,
    )


def resolve_layered_pltz_bundle(bundle_dir: Path) -> Tuple:
    """
    Resolve paths from a layered .pltz bundle (v2.0 format).

    Layered format structure:
        plot.pltz.d/
            spec.json           # Semantic
            style.json          # Appearance
            {basename}.csv      # Data
            exports/            # PNG, SVG, hitmap
            cache/              # geometry_px.json

    Parameters
    ----------
    bundle_dir : Path
        Path to .pltz.d bundle directory.

    Returns
    -------
    tuple
        (json_path, csv_path, png_path, hitmap_path, bundle_spec)
    """
    from scitex.plt.io import load_layered_pltz_bundle

    bundle_data = load_layered_pltz_bundle(bundle_dir)
    spec_path = bundle_dir / "spec.json"
    csv_path = None
    png_path = None
    hitmap_path = None

    # Find CSV
    for f in bundle_dir.glob("*.csv"):
        csv_path = f
        break

    # Find exports
    exports_dir = bundle_dir / "exports"
    if exports_dir.exists():
        for f in exports_dir.iterdir():
            name = f.name
            if name.endswith('_hitmap.png'):
                hitmap_path = f
            elif name.endswith('.svg') and '_hitmap' not in name:
                png_path = f
            elif name.endswith('.png') and '_hitmap' not in name and png_path is None:
                png_path = f

    bundle_spec = bundle_data.get("merged", {})

    if hitmap_path and "hit_regions" in bundle_spec:
        bundle_spec["hit_regions"]["hit_map"] = str(hitmap_path.name)

    return (
        spec_path,
        csv_path if csv_path and csv_path.exists() else None,
        png_path if png_path and png_path.exists() else None,
        hitmap_path if hitmap_path and hitmap_path.exists() else None,
        bundle_spec,
    )


# EOF
