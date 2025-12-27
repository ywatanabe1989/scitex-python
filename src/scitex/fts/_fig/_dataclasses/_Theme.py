#!/usr/bin/env python3
# Timestamp: 2025-12-21
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_dataclasses/_Theme.py

"""Theme - Visual aesthetics for FTS bundles."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

THEME_VERSION = "1.0.0"


@dataclass
class Colors:
    """Color palette configuration."""

    palette: List[str] = field(
        default_factory=lambda: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    )
    primary: Optional[str] = None
    secondary: Optional[str] = None
    background: str = "#ffffff"
    text: str = "#000000"
    axis: str = "#333333"
    grid: str = "#e0e0e0"

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "palette": self.palette,
            "background": self.background,
            "text": self.text,
            "axis": self.axis,
            "grid": self.grid,
        }
        if self.primary:
            result["primary"] = self.primary
        if self.secondary:
            result["secondary"] = self.secondary
        return result

    @classmethod
    def from_dict(cls, data: Any) -> "Colors":
        # Return if already a Colors instance
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return cls()
        return cls(
            palette=data.get("palette", ["#1f77b4", "#ff7f0e", "#2ca02c"]),
            primary=data.get("primary"),
            secondary=data.get("secondary"),
            background=data.get("background", "#ffffff"),
            text=data.get("text", "#000000"),
            axis=data.get("axis", "#333333"),
            grid=data.get("grid", "#e0e0e0"),
        )


@dataclass
class Typography:
    """Typography configuration."""

    family: str = "sans-serif"
    size_pt: float = 8.0
    title_size_pt: float = 10.0
    label_size_pt: float = 8.0
    tick_size_pt: float = 7.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "size_pt": self.size_pt,
            "title_size_pt": self.title_size_pt,
            "label_size_pt": self.label_size_pt,
            "tick_size_pt": self.tick_size_pt,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Typography":
        # Return if already a Typography instance
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return cls()
        return cls(
            family=data.get("family", "sans-serif"),
            size_pt=data.get("size_pt", 8.0),
            title_size_pt=data.get("title_size_pt", data.get("title_size", 10.0)),
            label_size_pt=data.get("label_size_pt", data.get("label_size", 8.0)),
            tick_size_pt=data.get("tick_size_pt", data.get("tick_size", 7.0)),
        )


@dataclass
class Lines:
    """Line style configuration."""

    width_pt: float = 1.0
    style: str = "solid"  # solid, dashed, dotted, dashdot

    def to_dict(self) -> Dict[str, Any]:
        return {"width_pt": self.width_pt, "style": self.style}

    @classmethod
    def from_dict(cls, data: Any) -> "Lines":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return cls()
        return cls(
            width_pt=data.get("width_pt", 1.0),
            style=data.get("style", "solid"),
        )


@dataclass
class Markers:
    """Marker style configuration."""

    size_pt: float = 4.0
    symbol: str = "circle"  # circle, square, triangle, cross, diamond

    def to_dict(self) -> Dict[str, Any]:
        return {"size_pt": self.size_pt, "symbol": self.symbol}

    @classmethod
    def from_dict(cls, data: Any) -> "Markers":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return cls()
        return cls(
            size_pt=data.get("size_pt", 4.0),
            symbol=data.get("symbol", "circle"),
        )


@dataclass
class Grid:
    """Grid line configuration."""

    show: bool = True
    major_width_pt: float = 0.5
    minor_width_pt: float = 0.25
    major_alpha: float = 0.3
    minor_alpha: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "show": self.show,
            "major_width_pt": self.major_width_pt,
            "minor_width_pt": self.minor_width_pt,
            "major_alpha": self.major_alpha,
            "minor_alpha": self.minor_alpha,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Grid":
        if isinstance(data, cls):
            return data
        if isinstance(data, bool):
            return cls(show=data)
        if not isinstance(data, dict):
            return cls()
        return cls(
            show=data.get("show", True),
            major_width_pt=data.get("major_width_pt", 0.5),
            minor_width_pt=data.get("minor_width_pt", 0.25),
            major_alpha=data.get("major_alpha", 0.3),
            minor_alpha=data.get("minor_alpha", 0.1),
        )


@dataclass
class TraceTheme:
    """Per-trace theme overrides."""

    trace_id: str
    color: Optional[str] = None
    line_width_pt: Optional[float] = None
    marker_size_pt: Optional[float] = None
    alpha: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"trace_id": self.trace_id}
        if self.color:
            result["color"] = self.color
        if self.line_width_pt is not None:
            result["line_width_pt"] = self.line_width_pt
        if self.marker_size_pt is not None:
            result["marker_size_pt"] = self.marker_size_pt
        if self.alpha is not None:
            result["alpha"] = self.alpha
        return result

    @classmethod
    def from_dict(cls, data: Any) -> "TraceTheme":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return cls(trace_id="trace_0")
        return cls(
            trace_id=data.get("trace_id", "trace_0"),
            color=data.get("color"),
            line_width_pt=data.get("line_width_pt"),
            marker_size_pt=data.get("marker_size_pt"),
            alpha=data.get("alpha"),
        )


@dataclass
class FigureTitle:
    """Figure title configuration for publications."""

    text: str = ""
    prefix: str = "Figure"
    number: Optional[int] = None
    fontsize: Optional[float] = None
    fontweight: str = "bold"

    def to_dict(self) -> Dict[str, Any]:
        result = {"text": self.text, "prefix": self.prefix, "fontweight": self.fontweight}
        if self.number is not None:
            result["number"] = self.number
        if self.fontsize is not None:
            result["fontsize"] = self.fontsize
        return result

    @classmethod
    def from_dict(cls, data: Any) -> "FigureTitle":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return cls()
        return cls(
            text=data.get("text", ""),
            prefix=data.get("prefix", "Figure"),
            number=data.get("number"),
            fontsize=data.get("fontsize"),
            fontweight=data.get("fontweight", "bold"),
        )


@dataclass
class PanelDescription:
    """Single panel description for captions."""

    label: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "description": self.description}

    @classmethod
    def from_dict(cls, data: Any) -> "PanelDescription":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return cls(label="", description="")
        return cls(
            label=data.get("label", ""),
            description=data.get("description", ""),
        )


@dataclass
class Caption:
    """Figure caption for publications."""

    text: str = ""
    panels: List[PanelDescription] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {"text": self.text}
        if self.panels:
            result["panels"] = [p.to_dict() for p in self.panels]
        return result

    @classmethod
    def from_dict(cls, data: Any) -> "Caption":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return cls()
        panels = [PanelDescription.from_dict(p) for p in data.get("panels", [])]
        return cls(text=data.get("text", ""), panels=panels)


@dataclass
class PanelLabels:
    """Panel label styling (A, B, C, etc.)."""

    style: str = "uppercase"  # uppercase, lowercase, numeric
    fontsize: float = 12.0
    fontweight: str = "bold"
    position: str = "top-left"  # top-left, top-right, bottom-left, bottom-right
    offset_x: float = 0.02
    offset_y: float = 0.98

    def to_dict(self) -> Dict[str, Any]:
        return {
            "style": self.style,
            "fontsize": self.fontsize,
            "fontweight": self.fontweight,
            "position": self.position,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "PanelLabels":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return cls()
        return cls(
            style=data.get("style", "uppercase"),
            fontsize=data.get("fontsize", 12.0),
            fontweight=data.get("fontweight", "bold"),
            position=data.get("position", "top-left"),
            offset_x=data.get("offset_x", 0.02),
            offset_y=data.get("offset_y", 0.98),
        )


@dataclass
class Theme:
    """Complete theme specification for an FTS bundle.

    Stored in canonical/theme.json.
    """

    # Core mode
    mode: str = "light"  # "light" | "dark"

    # Styling components
    colors: Colors = field(default_factory=Colors)
    typography: Typography = field(default_factory=Typography)
    lines: Lines = field(default_factory=Lines)
    markers: Markers = field(default_factory=Markers)
    grid: Grid = field(default_factory=Grid)

    # Per-trace overrides
    traces: List[TraceTheme] = field(default_factory=list)

    # Publication metadata
    preset: Optional[str] = None  # nature, science, cell, ieee, acs, minimal, presentation
    figure_title: Optional[FigureTitle] = None
    caption: Optional[Caption] = None
    panel_labels: Optional[PanelLabels] = None

    # Alias for typography (for convenience)
    fonts: Optional[Typography] = None

    # Schema metadata
    schema_name: str = "fts.theme"
    schema_version: str = THEME_VERSION

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "mode": self.mode,
            "colors": self.colors.to_dict(),
            "typography": self.typography.to_dict(),
            "lines": self.lines.to_dict(),
            "markers": self.markers.to_dict(),
            "grid": self.grid.to_dict(),
        }
        if self.traces:
            result["traces"] = [t.to_dict() for t in self.traces]
        if self.preset:
            result["preset"] = self.preset
        if self.figure_title:
            result["figure_title"] = self.figure_title.to_dict()
        if self.caption:
            result["caption"] = self.caption.to_dict()
        if self.panel_labels:
            result["panel_labels"] = self.panel_labels.to_dict()
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Theme":
        # Handle fonts as alias for typography
        typography_data = data.get("typography", {})
        fonts_data = data.get("fonts")
        if fonts_data and not typography_data:
            typography_data = fonts_data

        # Parse figure_title
        figure_title = None
        if "figure_title" in data and data["figure_title"]:
            figure_title = FigureTitle.from_dict(data["figure_title"])

        # Parse caption
        caption = None
        if "caption" in data and data["caption"]:
            caption = Caption.from_dict(data["caption"])

        # Parse panel_labels
        panel_labels = None
        if "panel_labels" in data and data["panel_labels"]:
            panel_labels = PanelLabels.from_dict(data["panel_labels"])

        return cls(
            mode=data.get("mode", "light"),
            colors=Colors.from_dict(data.get("colors", {})),
            typography=Typography.from_dict(typography_data),
            lines=Lines.from_dict(data.get("lines", {})),
            markers=Markers.from_dict(data.get("markers", {})),
            grid=Grid.from_dict(data.get("grid", {})),
            traces=[TraceTheme.from_dict(t) for t in data.get("traces", [])],
            preset=data.get("preset"),
            figure_title=figure_title,
            caption=caption,
            panel_labels=panel_labels,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Theme":
        return cls.from_dict(json.loads(json_str))


__all__ = [
    "THEME_VERSION",
    "Colors",
    "Typography",
    "Lines",
    "Markers",
    "Grid",
    "TraceTheme",
    "FigureTitle",
    "PanelDescription",
    "Caption",
    "PanelLabels",
    "Theme",
]

# EOF
