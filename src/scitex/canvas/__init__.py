#!/usr/bin/env python3
# Timestamp: 2026-01-29
# File: src/scitex/canvas/__init__.py
"""
SciTeX Canvas Module (DEPRECATED)

.. deprecated:: 2.16.0
    This module is deprecated. Use figrecipe instead:

    - Interactive editor: ``figrecipe.edit()`` (browser GUI at port 5050)
    - Multi-panel composition: ``figrecipe.compose()``
    - Save/reproduce: ``figrecipe.save()`` / ``figrecipe.reproduce()``

    Install: ``pip install figrecipe``

    Migration examples::

        # Old (scitex.canvas)
        from scitex.canvas import edit
        edit(fig)

        # New (figrecipe)
        import figrecipe as fr
        fr.edit(fig)

        # Old (scitex.canvas multi-panel)
        stx.canvas.create_canvas(...)
        stx.canvas.add_panel(...)

        # New (figrecipe.compose)
        fig, axes = fr.compose(
            sources={
                "panel_a.png": {"xy_mm": (10, 10), "size_mm": (80, 60)},
                "panel_b.png": {"xy_mm": (100, 10), "size_mm": (80, 60)},
            },
            canvas_size_mm=(190, 80),
            panel_labels=True,
        )
"""

from __future__ import annotations

import warnings

# Issue deprecation warning on import
warnings.warn(
    "scitex.canvas is deprecated. Use figrecipe instead:\n"
    "  - fr.edit() for interactive GUI editor (port 5050)\n"
    "  - fr.compose() for multi-panel composition\n"
    "  - pip install figrecipe",
    DeprecationWarning,
    stacklevel=2,
)


# =============================================================================
# Delegate to figrecipe (preferred)
# =============================================================================
def edit(*args, **kwargs):
    """Launch interactive GUI editor.

    .. deprecated:: 2.16.0
        Use ``figrecipe.edit()`` instead.
    """
    warnings.warn(
        "scitex.canvas.edit() is deprecated. Use figrecipe.edit() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        import figrecipe as fr

        return fr.edit(*args, **kwargs)
    except ImportError as e:
        raise ImportError(
            "figrecipe is required for the editor. Install with: pip install figrecipe"
        ) from e


def compose(*args, **kwargs):
    """Compose multi-panel figures.

    .. deprecated:: 2.16.0
        Use ``figrecipe.compose()`` instead.
    """
    warnings.warn(
        "scitex.canvas.compose() is deprecated. Use figrecipe.compose() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        import figrecipe as fr

        return fr.compose(*args, **kwargs)
    except ImportError as e:
        raise ImportError(
            "figrecipe is required for composition. Install with: pip install figrecipe"
        ) from e


# =============================================================================
# Legacy imports (deprecated, for backward compatibility)
# =============================================================================
def __getattr__(name):
    """Lazy import legacy submodules with deprecation warnings."""
    _legacy_submodules = {"backend", "editor", "io", "model", "utils"}
    _legacy_functions = {
        "Canvas",
        "create_canvas",
        "get_canvas_path",
        "canvas_exists",
        "list_canvases",
        "delete_canvas",
        "add_panel",
        "update_panel",
        "remove_panel",
        "list_panels",
        "export_canvas",
        "verify_data",
        "save_figure",
        "load_figure",
        "add_panel_from_image",
        "add_panel_from_scitex",
    }

    if name in _legacy_submodules:
        warnings.warn(
            f"scitex.canvas.{name} is deprecated. Use figrecipe instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name == "backend":
            from . import backend as mod
        elif name == "editor":
            from . import editor as mod
        elif name == "io":
            from . import io as mod
        elif name == "model":
            from . import model as mod
        elif name == "utils":
            from . import utils as mod
        return mod

    if name in _legacy_functions:
        warnings.warn(
            f"scitex.canvas.{name} is deprecated. Use figrecipe.compose() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Import from legacy locations
        if name == "Canvas":
            from .canvas import Canvas

            return Canvas
        elif name in {
            "create_canvas",
            "get_canvas_path",
            "canvas_exists",
            "list_canvases",
            "delete_canvas",
            "add_panel_from_image",
            "add_panel_from_scitex",
            "list_panels",
            "remove_panel",
            "update_panel",
            "export_canvas",
            "verify_data",
        }:
            from . import io as _io

            _name_map = {
                "create_canvas": "ensure_canvas_directory",
                "get_canvas_path": "get_canvas_directory_path",
                "canvas_exists": "canvas_directory_exists",
                "list_canvases": "list_canvas_directories",
                "delete_canvas": "delete_canvas_directory",
                "export_canvas": "export_canvas_to_file",
                "verify_data": "verify_all_data_hashes",
            }
            actual_name = _name_map.get(name, name)
            return getattr(_io, actual_name)
        elif name == "add_panel":
            from ._legacy import add_panel

            return add_panel
        elif name in {"save_figure", "load_figure"}:
            from ._legacy import load_figure, save_figure

            if name == "save_figure":
                return save_figure
            return load_figure

    raise AttributeError(f"module 'scitex.canvas' has no attribute '{name}'")


__all__ = [
    # Recommended (delegates to figrecipe)
    "edit",
    "compose",
    # Legacy (deprecated)
    "Canvas",
    "io",
    "model",
    "backend",
    "utils",
    "editor",
    "create_canvas",
    "get_canvas_path",
    "canvas_exists",
    "list_canvases",
    "delete_canvas",
    "add_panel",
    "update_panel",
    "remove_panel",
    "list_panels",
    "export_canvas",
    "verify_data",
    "save_figure",
    "load_figure",
]

# EOF
