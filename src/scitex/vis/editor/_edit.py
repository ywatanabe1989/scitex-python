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
    >>> stx.vis.edit("output/figure.json")  # Auto-select best backend
    >>> stx.vis.edit("output/figure.png", backend="flask")  # Force flask editor
    >>> stx.vis.edit("output/figure.json", backend="tkinter")  # Force tkinter

    Notes
    -----
    - Changes are saved to `{basename}.manual.json` alongside the original
    - Manual JSON includes hash of base JSON for staleness detection
    - Original JSON/CSV files are never modified
    - Backend auto-detection order: flask > dearpygui > qt > tkinter > mpl
    """
    path = Path(path)

    # Resolve paths (JSON, CSV, PNG)
    json_path, csv_path, png_path = _resolve_figure_paths(path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Load data
    import scitex as stx

    metadata = stx.io.load(json_path)
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
                manual_overrides=manual_overrides,
                port=port,
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
