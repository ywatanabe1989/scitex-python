#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-14 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/editor/edit/editor_launcher.py

"""Main edit function for launching visual editor."""

from pathlib import Path
from typing import Literal, Union

from .backend_detector import detect_best_backend, print_available_backends
from .bundle_resolver import resolve_figure_bundle, resolve_plot_bundle
from .path_resolver import resolve_figure_paths

__all__ = ["edit"]


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
        - .plot directory bundle (recommended for hitmap selection)
        - .plot ZIP bundle
        - .figure multi-panel bundle
        - JSON file (figure.json or figure.manual.json)
        - CSV file (figure.csv) - for data-only start
        - PNG file (figure.png)
    backend : str, optional
        GUI backend to use (default: "auto"):
        - "auto": Pick best available (flask -> dearpygui -> qt -> tkinter -> mpl)
        - "flask": Browser-based editor (Flask, modern UI)
        - "dearpygui": GPU-accelerated modern GUI
        - "qt": Rich desktop editor (requires PyQt5/6 or PySide2/6)
        - "tkinter": Built-in Python GUI
        - "mpl": Minimal matplotlib interactive mode
    apply_manual : bool, optional
        If True, load .manual.json overrides if exists (default: True)
    port : int, optional
        Port number for web-based editors. Default: 5050.

    Returns
    -------
    None
        Editor runs in GUI event loop. Changes saved to .manual.json.
    """
    path = Path(path)
    spath = str(path)
    parent_str = str(path.parent) if path.is_file() else ""

    # Panel info for multi-panel figures
    panel_info = None

    # Resolve paths based on input type
    json_path, csv_path, png_path, hitmap_path, bundle_spec, panel_info = (
        _resolve_paths(path, spath, parent_str)
    )

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
        backend = detect_best_backend()

    # Print status
    print_available_backends()
    print(f"Launching {backend} editor for: {json_path}")

    # Launch appropriate backend
    _launch_backend(
        backend=backend,
        json_path=json_path,
        metadata=metadata,
        csv_data=csv_data,
        png_path=png_path,
        hitmap_path=hitmap_path,
        manual_overrides=manual_overrides,
        bundle_spec=bundle_spec,
        panel_info=panel_info,
        port=port,
    )


def _resolve_paths(path: Path, spath: str, parent_str: str) -> tuple:
    """Resolve paths based on input type."""
    panel_info = None
    hitmap_path = None
    bundle_spec = None

    # Check if this is a .figure bundle (multi-panel figure)
    if spath.endswith(".figure") or spath.endswith(".figure"):
        json_path, csv_path, png_path, hitmap_path, bundle_spec, panel_info = (
            resolve_figure_bundle(path)
        )
    # Check if this is a .plot bundle
    elif (
        spath.endswith(".plot")
        or spath.endswith(".plot")
        or parent_str.endswith(".plot")
    ):
        bundle_path = path.parent if parent_str.endswith(".plot") else path
        json_path, csv_path, png_path, hitmap_path, bundle_spec = resolve_plot_bundle(
            bundle_path
        )
    # Check if file is inside a .figure
    elif parent_str.endswith(".figure") or (
        path.parent.parent and str(path.parent.parent).endswith(".figure")
    ):
        figure_path = (
            path.parent if parent_str.endswith(".figure") else path.parent.parent
        )
        json_path, csv_path, png_path, hitmap_path, bundle_spec, panel_info = (
            resolve_figure_bundle(figure_path)
        )
    else:
        # Standard file paths
        json_path, csv_path, png_path = resolve_figure_paths(path)

    return json_path, csv_path, png_path, hitmap_path, bundle_spec, panel_info


def _launch_backend(
    backend: str,
    json_path: Path,
    metadata: dict,
    csv_data,
    png_path,
    hitmap_path,
    manual_overrides,
    bundle_spec,
    panel_info,
    port: int,
) -> None:
    """Launch the specified editor backend."""
    if backend == "flask":
        _launch_flask(
            json_path,
            metadata,
            csv_data,
            png_path,
            hitmap_path,
            manual_overrides,
            port,
            panel_info,
        )
    elif backend == "dearpygui":
        _launch_dearpygui(json_path, metadata, csv_data, png_path, manual_overrides)
    elif backend == "qt":
        _launch_qt(
            json_path,
            metadata,
            csv_data,
            png_path,
            manual_overrides,
            hitmap_path,
            bundle_spec,
        )
    elif backend == "tkinter":
        _launch_tkinter(json_path, metadata, csv_data, manual_overrides)
    elif backend == "mpl":
        _launch_mpl(json_path, metadata, csv_data, manual_overrides)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            "Use 'auto', 'flask', 'dearpygui', 'qt', 'tkinter', or 'mpl'."
        )


def _launch_flask(
    json_path,
    metadata,
    csv_data,
    png_path,
    hitmap_path,
    manual_overrides,
    port,
    panel_info,
):
    """Launch Flask web editor."""
    try:
        from .._flask_editor import WebEditor

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


def _launch_dearpygui(json_path, metadata, csv_data, png_path, manual_overrides):
    """Launch DearPyGui editor."""
    try:
        from .._dearpygui_editor import DearPyGuiEditor

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
            "DearPyGui backend requires dearpygui. Install with: pip install dearpygui"
        ) from e


def _launch_qt(
    json_path, metadata, csv_data, png_path, manual_overrides, hitmap_path, bundle_spec
):
    """Launch Qt editor."""
    try:
        from .._qt_editor import QtEditor

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
            "Qt backend requires PyQt5/PyQt6 or PySide2/PySide6. Install with: pip install PyQt6"
        ) from e


def _launch_tkinter(json_path, metadata, csv_data, manual_overrides):
    """Launch Tkinter editor."""
    from .._tkinter_editor import TkinterEditor

    editor = TkinterEditor(
        json_path=json_path,
        metadata=metadata,
        csv_data=csv_data,
        manual_overrides=manual_overrides,
    )
    editor.run()


def _launch_mpl(json_path, metadata, csv_data, manual_overrides):
    """Launch matplotlib editor."""
    from .._mpl_editor import MplEditor

    editor = MplEditor(
        json_path=json_path,
        metadata=metadata,
        csv_data=csv_data,
        manual_overrides=manual_overrides,
    )
    editor.run()


# EOF
