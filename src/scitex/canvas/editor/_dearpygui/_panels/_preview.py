#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/_dearpygui/_panels/_preview.py

"""
Preview panel creation for DearPyGui editor.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._state import EditorState


def create_preview_panel(state: "EditorState", dpg) -> None:
    """Create the preview panel with figure image, click handler, and hover detection."""
    from .._handlers import on_preview_click, on_preview_hover

    with dpg.child_window(width=900, height=-1, tag="preview_panel"):
        dpg.add_text(
            "Figure Preview (click to select, hover to highlight)",
            color=(100, 200, 100),
        )
        dpg.add_separator()

        # Image display with click and move handlers
        with dpg.handler_registry(tag="preview_handler"):
            dpg.add_mouse_click_handler(
                callback=lambda s, a: on_preview_click(state, dpg, a)
            )
            dpg.add_mouse_move_handler(
                callback=lambda s, a: on_preview_hover(state, dpg, a)
            )

        dpg.add_image("preview_texture", tag="preview_image")

        dpg.add_separator()
        dpg.add_text("", tag="hover_text", color=(150, 200, 150))
        dpg.add_text("", tag="status_text", color=(150, 150, 150))
        dpg.add_text("", tag="selection_text", color=(200, 200, 100))


# EOF
