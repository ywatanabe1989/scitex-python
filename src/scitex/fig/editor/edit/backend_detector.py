#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-14 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/editor/edit/backend_detector.py

"""Backend detection and selection for figure editor."""

import warnings

__all__ = ["print_available_backends", "detect_best_backend"]

# Backend packages mapping
BACKENDS = {
    "flask": ["flask"],
    "dearpygui": ["dearpygui"],
    "qt": ["PyQt6", "PyQt5", "PySide6", "PySide2"],
    "tkinter": ["tkinter"],
    "mpl": ["matplotlib"],
}


def print_available_backends() -> None:
    """Print available backends status."""
    print("\n" + "=" * 50)
    print("SciTeX Visual Editor - Available Backends")
    print("=" * 50)

    for backend, packages in BACKENDS.items():
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


def detect_best_backend() -> str:
    """
    Detect the best available GUI backend with graceful degradation.

    Order: flask > dearpygui > qt > tkinter > mpl
    Shows warnings when falling back to less capable backends.
    """
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

    # Try DearPyGui again
    try:
        import dearpygui
        return "dearpygui"
    except ImportError:
        pass

    # Try Qt (richest desktop features)
    for qt_pkg in ["PyQt6", "PyQt5", "PySide6", "PySide2"]:
        try:
            __import__(qt_pkg)
            warnings.warn(
                "DearPyGui not available. Consider: pip install dearpygui\n"
                "Using Qt backend instead."
            )
            return "qt"
        except ImportError:
            pass

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


# EOF
