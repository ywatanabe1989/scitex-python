#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/_edit.py
"""Main edit function for launching visual editor."""

from pathlib import Path
from typing import Union, Optional, Literal
import hashlib
import json
import warnings


def _print_available_backends():
    """Print available backends status."""
    backends = {
        "flask": ["flask"],
        "dearpygui": ["dearpygui"],
        "qt": ["PyQt6", "PyQt5", "PySide6", "PySide2"],
        "tkinter": ["tkinter"],
        "mpl": ["matplotlib"],
    }

    print("\n" + "=" * 50)
    print("SciTeX Visual Editor - Available Backends")
    print("=" * 50)

    for backend, packages in backends.items():
        available = False
        available_pkg = None
        for pkg in packages:
            try:
                __import__(pkg)
                available = True
                available_pkg = pkg
                break
            except ImportError:
                pass

        status = f"[OK] {available_pkg}" if available else "[NOT INSTALLED]"
        print(f"  {backend:12s}: {status}")

    print("=" * 50)
    print("Install: pip install scitex[gui]")
    print("=" * 50 + "\n")


def edit(
    path: Union[str, Path],
    backend: Literal["auto", "flask", "dearpygui", "qt", "tkinter", "mpl"] = "auto",
    apply_manual: bool = True,
    port: int = 5050,
) -> None:
    """
    Launch interactive editor for figure style/annotation editing.

    Parameters
    ----------
    path : str or Path
        Path to figure file. Can be:
        - .pltz.d directory bundle (recommended for hitmap selection)
        - .pltz ZIP bundle
        - JSON file (figure.json or figure.manual.json)
        - CSV file (figure.csv) - for data-only start
        - PNG file (figure.png)
        Will auto-detect sibling files in same directory or organized subdirectories.
    backend : str, optional
        GUI backend to use (default: "auto"):
        - "auto": Pick best available with graceful degradation
          (flask -> dearpygui -> qt -> tkinter -> mpl)
        - "flask": Browser-based editor (Flask, modern UI)
        - "dearpygui": GPU-accelerated modern GUI (fast, requires dearpygui)
        - "qt": Rich desktop editor (requires PyQt5/6 or PySide2/6)
        - "tkinter": Built-in Python GUI (works everywhere)
        - "mpl": Minimal matplotlib interactive mode (always works)
    apply_manual : bool, optional
        If True, load .manual.json overrides if exists (default: True)
    port : int, optional
        Port number for web-based editors (flask). Default: 5050.
        If port is in use, will attempt to free it or find an alternative.

    Returns
    -------
    None
        Editor runs in GUI event loop. Changes saved to .manual.json.

    Examples
    --------
    >>> import scitex as stx
    >>> stx.fig.edit("output/myplot.pltz.d")  # Edit pltz bundle with hitmap selection
    >>> stx.fig.edit("output/myplot.pltz")    # Also works with ZIP bundles
    >>> stx.fig.edit("output/figure.json")    # Auto-select best backend
    >>> stx.fig.edit("output/figure.png", backend="flask")  # Force flask editor

    Notes
    -----
    - Changes are saved to `{basename}.manual.json` alongside the original
    - Manual JSON includes hash of base JSON for staleness detection
    - Original JSON/CSV files are never modified
    - Backend auto-detection order: flask > dearpygui > qt > tkinter > mpl
    - For .pltz bundles, hitmap-based element selection is available
    """
    path = Path(path)
    spath = str(path)
    parent_str = str(path.parent) if path.is_file() else ""

    # Panel info for multi-panel figures
    panel_info = None

    # Check if this is a .figz bundle (multi-panel figure)
    if spath.endswith('.figz.d') or spath.endswith('.figz'):
        json_path, csv_path, png_path, hitmap_path, bundle_spec, panel_info = _resolve_figz_bundle(path)
    # Check if this is a .pltz bundle (single panel) - either the directory or a file inside it
    elif spath.endswith('.pltz.d') or spath.endswith('.pltz') or parent_str.endswith('.pltz.d'):
        # If it's a file inside .pltz.d, use the parent directory
        bundle_path = path.parent if parent_str.endswith('.pltz.d') else path
        json_path, csv_path, png_path, hitmap_path, bundle_spec = _resolve_pltz_bundle(bundle_path)
    # Check if a JSON/CSV/PNG file is inside a .figz.d (passed individual panel file)
    elif parent_str.endswith('.figz.d') or (path.parent.parent and str(path.parent.parent).endswith('.figz.d')):
        # File is inside figz bundle, resolve from figz root
        figz_path = path.parent if parent_str.endswith('.figz.d') else path.parent.parent
        json_path, csv_path, png_path, hitmap_path, bundle_spec, panel_info = _resolve_figz_bundle(figz_path)
    else:
        # Resolve paths (JSON, CSV, PNG)
        json_path, csv_path, png_path = _resolve_figure_paths(path)
        hitmap_path = None
        bundle_spec = None

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Load data
    import scitex as stx

    metadata = bundle_spec if bundle_spec else stx.io.load(json_path)
    csv_data = None
    if csv_path and csv_path.exists():
        csv_data = stx.io.load(csv_path)

    # Load manual overrides if exists
    manual_path = json_path.with_suffix(".manual.json")
    manual_overrides = None
    if apply_manual and manual_path.exists():
        manual_data = stx.io.load(manual_path)
        manual_overrides = manual_data.get("overrides", {})

    # Resolve backend if "auto"
    if backend == "auto":
        backend = _detect_best_backend()

    # Print status
    _print_available_backends()
    print(f"Launching {backend} editor for: {json_path}")

    # Launch appropriate backend
    if backend == "flask":
        try:
            from ._flask_editor import WebEditor

            editor = WebEditor(
                json_path=json_path,
                metadata=metadata,
                csv_data=csv_data,
                png_path=png_path,
                hitmap_path=hitmap_path,
                manual_overrides=manual_overrides,
                port=port,
                panel_info=panel_info,
            )
            editor.run()
        except ImportError as e:
            raise ImportError(
                "Flask backend requires Flask. Install with: pip install flask"
            ) from e
    elif backend == "dearpygui":
        try:
            from ._dearpygui_editor import DearPyGuiEditor

            editor = DearPyGuiEditor(
                json_path=json_path,
                metadata=metadata,
                csv_data=csv_data,
                png_path=png_path,
                manual_overrides=manual_overrides,
            )
            editor.run()
        except ImportError as e:
            raise ImportError(
                "DearPyGui backend requires dearpygui. "
                "Install with: pip install dearpygui"
            ) from e
    elif backend == "qt":
        try:
            from ._qt_editor import QtEditor

            editor = QtEditor(
                json_path=json_path,
                metadata=metadata,
                csv_data=csv_data,
                png_path=png_path,
                manual_overrides=manual_overrides,
                hitmap_path=hitmap_path,
                bundle_spec=bundle_spec,
            )
            editor.run()
        except ImportError as e:
            raise ImportError(
                "Qt backend requires PyQt5/PyQt6 or PySide2/PySide6. "
                "Install with: pip install PyQt6"
            ) from e
    elif backend == "tkinter":
        from ._tkinter_editor import TkinterEditor

        editor = TkinterEditor(
            json_path=json_path,
            metadata=metadata,
            csv_data=csv_data,
            manual_overrides=manual_overrides,
        )
        editor.run()
    elif backend == "mpl":
        from ._mpl_editor import MplEditor

        editor = MplEditor(
            json_path=json_path,
            metadata=metadata,
            csv_data=csv_data,
            manual_overrides=manual_overrides,
        )
        editor.run()
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            "Use 'auto', 'flask', 'dearpygui', 'qt', 'tkinter', or 'mpl'."
        )


def _detect_best_backend() -> str:
    """
    Detect the best available GUI backend with graceful degradation.

    Order: flask > dearpygui > qt > tkinter > mpl
    Shows warnings when falling back to less capable backends.
    """
    import warnings

    # Try Flask - best for modern UI
    try:
        import flask

        return "flask"
    except ImportError:
        pass

    # Try DearPyGui - GPU-accelerated, modern
    try:
        import dearpygui

        return "dearpygui"
    except ImportError:
        warnings.warn(
            "Flask not available. Consider: pip install flask\nTrying DearPyGui..."
        )

    # Try DearPyGui
    try:
        import dearpygui

        return "dearpygui"
    except ImportError:
        pass

    # Try Qt (richest desktop features)
    qt_available = False
    try:
        import PyQt6

        qt_available = True
    except ImportError:
        pass
    if not qt_available:
        try:
            import PyQt5

            qt_available = True
        except ImportError:
            pass
    if not qt_available:
        try:
            import PySide6

            qt_available = True
        except ImportError:
            pass
    if not qt_available:
        try:
            import PySide2

            qt_available = True
        except ImportError:
            pass

    if qt_available:
        warnings.warn(
            "DearPyGui not available. Consider: pip install dearpygui\n"
            "Using Qt backend instead."
        )
        return "qt"

    # Try Tkinter (built-in, good features)
    try:
        import tkinter

        warnings.warn(
            "Qt not available. Consider: pip install PyQt6\n"
            "Using Tkinter backend (basic features)."
        )
        return "tkinter"
    except ImportError:
        pass

    # Fall back to matplotlib interactive (always works)
    warnings.warn(
        "No GUI toolkit found. Using minimal matplotlib editor.\n"
        "For better experience, install: pip install flask (web) or pip install PyQt6 (desktop)"
    )
    return "mpl"


def _resolve_figure_paths(path: Path) -> tuple:
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


def _resolve_figz_bundle(path: Path, panel_index: int = 0) -> tuple:
    """
    Resolve paths from a .figz bundle (multi-panel figure).

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
        panel_info is a dict with keys: panels, current_index, figz_dir
    """
    import json as json_module
    import tempfile
    import zipfile

    spath = str(path)

    # Handle ZIP vs directory
    if spath.endswith('.figz') and not spath.endswith('.figz.d'):
        # It's a ZIP - extract to temp directory
        if not path.exists():
            raise FileNotFoundError(f"Figz bundle not found: {path}")
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
    pltz_dirs = sorted([d for d in bundle_dir.iterdir()
                       if d.is_dir() and str(d).endswith('.pltz.d')])

    if not pltz_dirs:
        raise FileNotFoundError(f"No .pltz.d panels found in figz bundle: {bundle_dir}")

    # Validate panel index
    if panel_index < 0 or panel_index >= len(pltz_dirs):
        panel_index = 0

    selected_panel = pltz_dirs[panel_index]
    print(f"Opening panel: {selected_panel.name}")
    if len(pltz_dirs) > 1:
        print(f"  (Figz contains {len(pltz_dirs)} panels: {[d.name for d in pltz_dirs]})")

    # Build panel info for editor
    panel_info = {
        "panels": [d.name for d in pltz_dirs],
        "current_index": panel_index,
        "figz_dir": str(bundle_dir),
    }

    # Delegate to pltz resolver
    result = _resolve_pltz_bundle(selected_panel)
    # Append panel_info to the result
    return result + (panel_info,)


def _resolve_pltz_bundle(path: Path) -> tuple:
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
    import json as json_module
    import tempfile
    import zipfile

    spath = str(path)

    # Handle ZIP vs directory
    if spath.endswith('.pltz') and not spath.endswith('.pltz.d'):
        # It's a ZIP - extract to temp directory
        if not path.exists():
            raise FileNotFoundError(f"Bundle not found: {path}")
        temp_dir = tempfile.mkdtemp(prefix='scitex_edit_')
        with zipfile.ZipFile(path, 'r') as zf:
            zf.extractall(temp_dir)
        # Find the .pltz.d directory inside
        bundle_dir = Path(temp_dir)
        for item in bundle_dir.iterdir():
            if item.is_dir() and str(item).endswith('.pltz.d'):
                bundle_dir = item
                break
    else:
        # It's already a directory
        bundle_dir = Path(path)
        if not bundle_dir.exists():
            raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    # Check if this is a layered bundle (v2.0)
    spec_path = bundle_dir / "spec.json"
    if spec_path.exists():
        return _resolve_layered_pltz_bundle(bundle_dir)

    # === Legacy format below ===
    # Find files by pattern (supports basename-based naming)
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

    # Prefer SVG for display (contains complete figure with all data)
    # PNG (panel_A.png) is transparent overlay without line data
    # Coordinate mapping is handled by _extract_bboxes_from_metadata with actual display dimensions
    if svg_path:
        png_path = svg_path  # Use SVG for display

    # Load the spec
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


def _resolve_layered_pltz_bundle(bundle_dir: Path) -> tuple:
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
    import json as json_module
    from scitex.plt.io import load_layered_pltz_bundle, merge_layered_bundle

    # Load layered bundle
    bundle_data = load_layered_pltz_bundle(bundle_dir)
    basename = bundle_data.get("basename", "plot")

    # Paths
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
                png_path = f  # Prefer SVG
            elif name.endswith('.png') and '_hitmap' not in name and png_path is None:
                png_path = f

    # Merged spec for backward compatibility with editor
    bundle_spec = bundle_data.get("merged", {})

    # Add hit_regions path reference for editor
    if hitmap_path and "hit_regions" in bundle_spec:
        bundle_spec["hit_regions"]["hit_map"] = str(hitmap_path.name)

    return (
        spec_path,  # Return spec.json as the main JSON path
        csv_path if csv_path and csv_path.exists() else None,
        png_path if png_path and png_path.exists() else None,
        hitmap_path if hitmap_path and hitmap_path.exists() else None,
        bundle_spec,
    )


def _load_panel_data(panel_dir: Path) -> Optional[dict]:
    """
    Load panel data from a .pltz.d directory.

    Used by switch_panel endpoint to load a different panel's data.

    Parameters
    ----------
    panel_dir : Path
        Path to .pltz.d panel directory

    Returns
    -------
    dict or None
        Dictionary with keys: json_path, metadata, csv_data, png_path, hitmap_path
        Returns None if panel cannot be loaded
    """
    import json as json_module
    import scitex as stx

    if not panel_dir.exists():
        return None

    # Check for layered vs legacy format
    spec_path = panel_dir / "spec.json"
    if spec_path.exists():
        # Layered format
        from scitex.plt.io import load_layered_pltz_bundle
        bundle_data = load_layered_pltz_bundle(panel_dir)
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
                if name.endswith('_hitmap.png'):
                    hitmap_path = f
                elif name.endswith('.png') and '_hitmap' not in name and '_overview' not in name:
                    png_path = f
                elif name.endswith('.svg') and '_hitmap' not in name and svg_path is None:
                    svg_path = f
        # Fall back to SVG only if no PNG found (though PIL can't open it)
        if png_path is None:
            png_path = svg_path

        return {
            "json_path": spec_path,
            "metadata": metadata,
            "csv_data": csv_data,
            "png_path": png_path,
            "hitmap_path": hitmap_path,
        }
    else:
        # Legacy format
        json_path = None
        csv_data = None
        png_path = None
        hitmap_path = None

        for f in panel_dir.iterdir():
            name = f.name
            if name.endswith('.json') and not name.endswith('.manual.json'):
                json_path = f
            elif name.endswith('.csv'):
                csv_data = stx.io.load(f)
            elif name.endswith('_hitmap.png'):
                hitmap_path = f
            elif name.endswith('.svg') and '_hitmap' not in name:
                png_path = f
            elif name.endswith('.png') and '_hitmap' not in name and '_overview' not in name:
                if png_path is None:
                    png_path = f

        if json_path is None:
            return None

        with open(json_path, 'r') as f:
            metadata = json_module.load(f)

        return {
            "json_path": json_path,
            "metadata": metadata,
            "csv_data": csv_data,
            "png_path": png_path,
            "hitmap_path": hitmap_path,
        }


def _compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def save_manual_overrides(
    json_path: Path,
    overrides: dict,
) -> Path:
    """
    Save manual overrides to .manual.json file.

    Parameters
    ----------
    json_path : Path
        Path to base JSON file
    overrides : dict
        Override settings (styles, annotations, etc.)

    Returns
    -------
    Path
        Path to saved manual.json file
    """
    import scitex as stx

    manual_path = json_path.with_suffix(".manual.json")

    # Compute hash of base JSON for staleness detection
    base_hash = _compute_file_hash(json_path)

    manual_data = {
        "base_file": json_path.name,
        "base_hash": base_hash,
        "overrides": overrides,
    }

    stx.io.save(manual_data, manual_path)
    return manual_path


# EOF
