#!/usr/bin/env python3
# Timestamp: 2025-12-08
# File: ./src/scitex/vis/canvas.py
"""
Canvas class for scitex.canvas.

Provides object-oriented interface to canvas operations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union


class Canvas:
    """
    Canvas for composing publication-quality figures.

    A Canvas represents a paper figure workspace containing multiple panels.
    It can be saved to a .canvas directory bundle for portability.

    Parameters
    ----------
    name : str
        Canvas name (e.g., "fig1_neural_results")
    width_mm : float
        Canvas width in millimeters (default: 180)
    height_mm : float
        Canvas height in millimeters (default: 240)

    Examples
    --------
    >>> import scitex as stx
    >>> # Create canvas
    >>> canvas = stx.vis.Canvas("fig1", width_mm=180, height_mm=120)
    >>> # Add panels
    >>> canvas.add_panel("panel_a", "plot.png", xy_mm=(10, 10), size_mm=(80, 50), label="A")
    >>> canvas.add_panel("panel_b", "chart.png", xy_mm=(100, 10), size_mm=(80, 50), label="B")
    >>> # Save (auto-exports PNG/PDF/SVG)
    >>> stx.io.save(canvas, "/output/fig1.canvas")
    """

    def __init__(
        self,
        name: str,
        width_mm: float = 180,
        height_mm: float = 240,
    ):
        self._name = name
        self._width_mm = width_mm
        self._height_mm = height_mm
        self._panels: List[Dict[str, Any]] = []
        self._annotations: List[Dict[str, Any]] = []
        self._title: Dict[str, Any] = {"text": "", "position": {}, "fontsize": 14}
        self._background: Dict[str, Any] = {"color": "#ffffff", "grid": False}
        self._metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "author": "",
            "description": "",
        }
        self._caption: Dict[str, Any] = {
            "text": "",
            "render": False,
            "fontsize": 10,
            "width_mm": None,
        }
        self._source_files: Dict[str, Path] = {}  # panel_name -> source_path

    @property
    def name(self) -> str:
        """Canvas name."""
        return self._name

    @property
    def panels(self) -> List[Dict[str, Any]]:
        """List of panel configurations."""
        return self._panels

    def add_panel(
        self,
        panel_name: str,
        source: Union[str, Path],
        xy_mm: tuple = (0, 0),
        size_mm: tuple = (50, 50),
        label: str = "",
        **kwargs,
    ) -> "Canvas":
        """
        Add a panel to the canvas.

        Parameters
        ----------
        panel_name : str
            Name for the panel
        source : str or Path
            Path to source file (PNG, JPG, SVG)
        xy_mm : tuple
            (x_mm, y_mm) position on canvas in millimeters
        size_mm : tuple
            (width_mm, height_mm) panel size in millimeters
        label : str
            Panel label (A, B, C...)
        **kwargs
            Additional panel properties (rotation_deg, opacity, flip_h, etc.)

        Returns
        -------
        Canvas
            Self for method chaining
        """
        source = Path(source)

        # Determine panel type
        json_sibling = source.parent / f"{source.stem}.json"
        panel_type = "scitex" if json_sibling.exists() else "image"

        # Build panel entry
        panel_entry = {
            "name": panel_name,
            "type": panel_type,
            "position": {"x_mm": xy_mm[0], "y_mm": xy_mm[1]},
            "size": {"width_mm": size_mm[0], "height_mm": size_mm[1]},
            "z_index": len(self._panels),
            "rotation_deg": kwargs.get("rotation_deg", 0),
            "opacity": kwargs.get("opacity", 1.0),
            "flip_h": kwargs.get("flip_h", False),
            "flip_v": kwargs.get("flip_v", False),
            "visible": kwargs.get("visible", True),
            "clip": {
                "enabled": False,
                "x_mm": 0,
                "y_mm": 0,
                "width_mm": None,
                "height_mm": None,
            },
            "label": {
                "text": label,
                "position": "top-left",
                "fontsize": 12,
                "fontweight": "bold",
            },
            "border": {
                "visible": False,
                "color": "#000000",
                "width_mm": 0.2,
            },
        }

        if panel_type == "image":
            panel_entry["source"] = f"panel{source.suffix}"

        # Apply any additional kwargs to nested dicts
        for key, value in kwargs.items():
            if (
                key in panel_entry
                and isinstance(panel_entry[key], dict)
                and isinstance(value, dict)
            ):
                panel_entry[key].update(value)
            elif key not in ["rotation_deg", "opacity", "flip_h", "flip_v", "visible"]:
                panel_entry[key] = value

        # Remove existing panel with same name
        self._panels = [p for p in self._panels if p.get("name") != panel_name]
        self._panels.append(panel_entry)

        # Store source file path for later
        self._source_files[panel_name] = source.resolve()

        return self

    def update_panel(self, panel_name: str, updates: Dict[str, Any]) -> "Canvas":
        """
        Update panel properties.

        Parameters
        ----------
        panel_name : str
            Name of panel to update
        updates : Dict[str, Any]
            Properties to update

        Returns
        -------
        Canvas
            Self for method chaining
        """
        for panel in self._panels:
            if panel.get("name") == panel_name:
                _deep_merge(panel, updates)
                break
        return self

    def remove_panel(self, panel_name: str) -> "Canvas":
        """
        Remove a panel from the canvas.

        Parameters
        ----------
        panel_name : str
            Name of panel to remove

        Returns
        -------
        Canvas
            Self for method chaining
        """
        self._panels = [p for p in self._panels if p.get("name") != panel_name]
        self._source_files.pop(panel_name, None)
        return self

    def add_annotation(
        self,
        ann_type: str,
        **kwargs,
    ) -> "Canvas":
        """
        Add an annotation to the canvas.

        Parameters
        ----------
        ann_type : str
            Annotation type: "text", "arrow", "bracket", "line", "rectangle", "legend"
        **kwargs
            Type-specific properties

        Returns
        -------
        Canvas
            Self for method chaining
        """
        annotation = {"type": ann_type, **kwargs}
        self._annotations.append(annotation)
        return self

    def set_caption(
        self,
        text: str,
        render: bool = False,
        position: tuple = None,
        fontsize: int = 10,
        width_mm: float = None,
    ) -> "Canvas":
        """
        Set figure caption (legend in scientific sense).

        Caption is stored as metadata by default. Use render=True to
        include it in the exported image. "Figure X." numbering should
        be handled by LaTeX/document side, not included in caption text.

        Parameters
        ----------
        text : str
            Caption text describing the figure, e.g.,
            "Neural activity across conditions. (A) Control. (B) Treatment."
            Do NOT include "Figure X." - that's handled by LaTeX.
        render : bool
            If True, render caption in exported image.
            If False (default), store as metadata only.
        position : tuple, optional
            (x_mm, y_mm) position when render=True. Default: below canvas
        fontsize : int
            Font size when rendered (default: 10)
        width_mm : float, optional
            Text wrap width in mm. Default: canvas width - 20mm margins

        Returns
        -------
        Canvas
            Self for method chaining

        Examples
        --------
        >>> # Caption as metadata only (for LaTeX)
        >>> canvas.set_caption(
        ...     "Neural responses to visual stimuli. "
        ...     "(A) Raw signals. (B) Filtered signals."
        ... )
        >>>
        >>> # Caption rendered in image
        >>> canvas.set_caption(
        ...     "Neural responses to visual stimuli.",
        ...     render=True
        ... )
        """
        self._caption = {
            "text": text,
            "render": render,
            "fontsize": fontsize,
            "width_mm": width_mm,
        }
        if position:
            self._caption["position"] = {"x_mm": position[0], "y_mm": position[1]}

        return self

    def set_title(
        self, text: str, position: tuple = None, fontsize: int = 14
    ) -> "Canvas":
        """
        Set canvas title.

        Parameters
        ----------
        text : str
            Title text
        position : tuple, optional
            (x_mm, y_mm) position
        fontsize : int
            Font size

        Returns
        -------
        Canvas
            Self for method chaining
        """
        self._title["text"] = text
        if position:
            self._title["position"] = {"x_mm": position[0], "y_mm": position[1]}
        self._title["fontsize"] = fontsize
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert canvas to dictionary representation.

        Returns
        -------
        Dict[str, Any]
            Canvas as dictionary (canvas.json structure)
        """
        return {
            "schema_version": "2.0.0",
            "canvas_name": self._name,
            "size": {
                "width_mm": self._width_mm,
                "height_mm": self._height_mm,
            },
            "background": self._background,
            "panels": self._panels,
            "annotations": self._annotations,
            "title": self._title,
            "caption": self._caption,
            "data_files": [],
            "metadata": self._metadata,
            "manual_overrides": {},
            "_source_files": {k: str(v) for k, v in self._source_files.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Canvas":
        """
        Create canvas from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Canvas dictionary (from canvas.json or to_dict())

        Returns
        -------
        Canvas
            New Canvas instance
        """
        size = data.get("size", {})
        canvas = cls(
            name=data.get("canvas_name", "untitled"),
            width_mm=size.get("width_mm", 180),
            height_mm=size.get("height_mm", 240),
        )
        canvas._panels = data.get("panels", [])
        canvas._annotations = data.get("annotations", [])
        canvas._title = data.get("title", canvas._title)
        canvas._caption = data.get("caption", canvas._caption)
        canvas._background = data.get("background", canvas._background)
        canvas._metadata = data.get("metadata", canvas._metadata)

        # Restore source files if present
        source_files = data.get("_source_files", {})
        for panel_name, path_str in source_files.items():
            canvas._source_files[panel_name] = Path(path_str)

        return canvas

    def save(
        self,
        path: Union[str, Path],
        bundle: bool = False,
        **kwargs,
    ) -> Path:
        """
        Save canvas to a .canvas directory.

        Parameters
        ----------
        path : str or Path
            Path where the .canvas directory should be created.
            Must end with .canvas extension.
        bundle : bool
            If True, copy source files. If False (default), use symlinks.
        **kwargs
            Additional arguments passed to stx.io.save

        Returns
        -------
        Path
            Path to the created .canvas directory.

        Examples
        --------
        >>> canvas = stx.vis.Canvas("fig1")
        >>> canvas.add_panel("panel_a", "plot.png", ...)
        >>> canvas.save("/output/fig1.canvas")  # Uses symlinks
        >>> canvas.save("/output/fig1.canvas", bundle=True)  # Copies files
        """
        import scitex as stx

        return stx.io.save(self, path, bundle=bundle, **kwargs)

    def __repr__(self) -> str:
        return f"Canvas(name='{self._name}', panels={len(self._panels)})"


def _deep_merge(base: Dict, updates: Dict) -> None:
    """Deep merge updates into base dictionary (in-place)."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# EOF
