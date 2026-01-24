#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/_dearpygui/_editor.py

"""
DearPyGui-based figure editor with GPU-accelerated rendering.

Thin orchestrator class that delegates to modular components.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from ._state import EditorState


class DearPyGuiEditor:
    """
    GPU-accelerated figure editor using DearPyGui.

    Features:
    - Modern immediate-mode GUI with GPU acceleration
    - Real-time figure preview
    - Property editors with sliders, color pickers, and input fields
    - Click-to-select traces on preview
    - Save to .manual.json
    - SciTeX style defaults pre-filled
    - Dark/light theme support
    """

    def __init__(
        self,
        json_path: Path,
        metadata: Dict[str, Any],
        csv_data: Optional[Any] = None,
        png_path: Optional[Path] = None,
        manual_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the DearPyGui editor.

        Parameters
        ----------
        json_path : Path
            Path to the JSON metadata file
        metadata : dict
            Figure metadata dictionary
        csv_data : pd.DataFrame, optional
            CSV data for plotting
        png_path : Path, optional
            Path to the PNG file
        manual_overrides : dict, optional
            Manual override settings
        """
        self.state = EditorState.create(
            json_path=json_path,
            metadata=metadata,
            csv_data=csv_data,
            png_path=png_path,
            manual_overrides=manual_overrides,
        )

        # Backward compatibility properties
        self._texture_id = None

    @property
    def json_path(self) -> Path:
        """Get JSON path."""
        return self.state.json_path

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata."""
        return self.state.metadata

    @property
    def csv_data(self) -> Optional[Any]:
        """Get CSV data."""
        return self.state.csv_data

    @property
    def current_overrides(self) -> Dict[str, Any]:
        """Get current overrides."""
        return self.state.current_overrides

    @current_overrides.setter
    def current_overrides(self, value: Dict[str, Any]) -> None:
        """Set current overrides."""
        self.state.current_overrides = value

    def run(self):
        """Launch the DearPyGui editor."""
        try:
            import dearpygui.dearpygui as dpg
        except ImportError:
            raise ImportError(
                "DearPyGui is required for this editor. "
                "Install with: pip install dearpygui"
            )

        from ._panels import create_control_panel, create_preview_panel
        from ._rendering import update_preview

        dpg.create_context()

        # Configure viewport
        dpg.create_viewport(
            title=f"SciTeX Editor ({self.state.backend_name}) - {self.state.json_path.name}",
            width=1400,
            height=900,
        )

        # Create texture registry for image preview
        with dpg.texture_registry(show=False):
            # Create initial texture with placeholder
            width, height = 800, 600
            texture_data = [0.2, 0.2, 0.2, 1.0] * (width * height)
            self._texture_id = dpg.add_dynamic_texture(
                width=width,
                height=height,
                default_value=texture_data,
                tag="preview_texture",
            )
            self.state.texture_id = self._texture_id

        # Create main window
        with dpg.window(label="SciTeX Figure Editor", tag="main_window"):
            with dpg.group(horizontal=True):
                # Left panel: Preview
                create_preview_panel(self.state, dpg)

                # Right panel: Controls
                create_control_panel(self.state, dpg)

        # Set main window as primary
        dpg.set_primary_window("main_window", True)

        # Initial render
        update_preview(self.state, dpg)

        # Setup and show
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()


# EOF
