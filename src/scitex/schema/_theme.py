#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/schema/_theme.py

"""
Theme Schema - Pure Aesthetics for Visual Presentation.

This module defines the theme layer that controls visual appearance
without affecting scientific meaning. Theme can be changed without
affecting reproducibility of the data representation.

Theme (theme.json) - Aesthetics Only:
  - Colors and color palettes
  - Fonts and typography
  - Line styles and widths
  - Marker styles and sizes
  - Layout and spacing
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

THEME_VERSION = "1.0.0"


@dataclass
class ColorScheme:
    """
    Color scheme specification.

    Parameters
    ----------
    mode : str
        Color mode (light, dark, auto)
    background : str
        Figure background color
    axes_bg : str
        Axes background color
    text : str
        Text color
    spine : str
        Axes spine color
    tick : str
        Tick mark color
    grid : str
        Grid line color
    palette : str, optional
        Named color palette for traces
    """

    mode: str = "light"
    background: str = "transparent"
    axes_bg: str = "white"
    text: str = "black"
    spine: str = "black"
    tick: str = "black"
    grid: str = "#cccccc"
    palette: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "mode": self.mode,
            "background": self.background,
            "axes_bg": self.axes_bg,
            "text": self.text,
            "spine": self.spine,
            "tick": self.tick,
            "grid": self.grid,
        }
        if self.palette:
            result["palette"] = self.palette
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColorScheme":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Typography:
    """
    Typography specification.

    Parameters
    ----------
    family : str
        Font family
    size_pt : float
        Base font size in points
    title_size_pt : float
        Title font size in points
    label_size_pt : float
        Axis label font size in points
    tick_size_pt : float
        Tick label font size in points
    legend_size_pt : float
        Legend font size in points
    weight : str
        Font weight (normal, bold, etc.)
    """

    family: str = "sans-serif"
    size_pt: float = 7.0
    title_size_pt: float = 8.0
    label_size_pt: float = 7.0
    tick_size_pt: float = 6.0
    legend_size_pt: float = 6.0
    weight: str = "normal"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Typography":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LineDefaults:
    """Default line style settings."""

    width: float = 1.0
    style: str = "-"  # solid
    alpha: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineDefaults":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MarkerDefaults:
    """Default marker style settings."""

    size: float = 6.0
    style: str = "o"
    edge_width: float = 0.5
    edge_color: str = "auto"  # auto means same as fill or black

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarkerDefaults":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TraceStyle:
    """
    Per-trace style overrides.

    Parameters
    ----------
    trace_id : str
        ID of the trace this style applies to
    color : str, optional
        Trace color
    linewidth : float, optional
        Line width
    linestyle : str, optional
        Line style
    marker : str, optional
        Marker style
    markersize : float, optional
        Marker size
    alpha : float, optional
        Opacity
    """

    trace_id: str
    color: Optional[str] = None
    linewidth: Optional[float] = None
    linestyle: Optional[str] = None
    marker: Optional[str] = None
    markersize: Optional[float] = None
    alpha: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"trace_id": self.trace_id}
        for name in [
            "color",
            "linewidth",
            "linestyle",
            "marker",
            "markersize",
            "alpha",
        ]:
            val = getattr(self, name)
            if val is not None:
                result[name] = val
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceStyle":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LegendStyle:
    """Legend styling options."""

    visible: bool = True
    location: str = "best"
    frameon: bool = True
    ncols: int = 1
    title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "visible": self.visible,
            "location": self.location,
            "frameon": self.frameon,
            "ncols": self.ncols,
        }
        if self.title:
            result["title"] = self.title
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegendStyle":
        if isinstance(data, bool):
            return cls(visible=data)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PlotTheme:
    """
    Complete theme specification for a plot.

    Stored in theme.json. Contains pure aesthetics that can be
    changed without affecting scientific interpretation.

    Parameters
    ----------
    colors : ColorScheme
        Color scheme settings
    typography : Typography
        Font settings
    line : LineDefaults
        Default line style
    marker : MarkerDefaults
        Default marker style
    traces : list of TraceStyle
        Per-trace style overrides
    legend : LegendStyle
        Legend styling
    grid : bool
        Whether to show grid
    """

    colors: ColorScheme = field(default_factory=ColorScheme)
    typography: Typography = field(default_factory=Typography)
    line: LineDefaults = field(default_factory=LineDefaults)
    marker: MarkerDefaults = field(default_factory=MarkerDefaults)
    traces: List[TraceStyle] = field(default_factory=list)
    legend: LegendStyle = field(default_factory=LegendStyle)
    grid: bool = False

    # Schema metadata
    scitex_schema: str = "scitex.plt.theme"
    scitex_schema_version: str = THEME_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": {
                "name": self.scitex_schema,
                "version": self.scitex_schema_version,
            },
            "colors": self.colors.to_dict(),
            "typography": self.typography.to_dict(),
            "line": self.line.to_dict(),
            "marker": self.marker.to_dict(),
            "traces": [t.to_dict() for t in self.traces],
            "legend": self.legend.to_dict(),
            "grid": self.grid,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlotTheme":
        return cls(
            colors=ColorScheme.from_dict(data.get("colors", {})),
            typography=Typography.from_dict(data.get("typography", {})),
            line=LineDefaults.from_dict(data.get("line", {})),
            marker=MarkerDefaults.from_dict(data.get("marker", {})),
            traces=[TraceStyle.from_dict(t) for t in data.get("traces", [])],
            legend=LegendStyle.from_dict(data.get("legend", {})),
            grid=data.get("grid", False),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PlotTheme":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_style(cls, style: dict) -> "PlotTheme":
        """
        Create theme from PlotStyle dict for backward compatibility.
        """
        theme_data = style.get("theme", {})
        font_data = style.get("font", {})
        size_data = style.get("size", {})

        return cls(
            colors=ColorScheme(
                mode=theme_data.get("mode", "light"),
                background=theme_data.get("colors", {}).get(
                    "background", "transparent"
                ),
                axes_bg=theme_data.get("colors", {}).get("axes_bg", "white"),
                text=theme_data.get("colors", {}).get("text", "black"),
                spine=theme_data.get("colors", {}).get("spine", "black"),
                tick=theme_data.get("colors", {}).get("tick", "black"),
                palette=theme_data.get("palette"),
            ),
            typography=Typography(
                family=font_data.get("family", "sans-serif"),
                size_pt=font_data.get("size_pt", 7.0),
                title_size_pt=font_data.get("title_size_pt", 8.0),
                label_size_pt=font_data.get("label_size_pt", 7.0),
                tick_size_pt=font_data.get("tick_size_pt", 6.0),
            ),
            traces=[
                TraceStyle(
                    trace_id=t.get("trace_id", ""),
                    color=t.get("color"),
                    linewidth=t.get("linewidth"),
                    linestyle=t.get("linestyle"),
                    marker=t.get("marker"),
                    markersize=t.get("markersize"),
                    alpha=t.get("alpha"),
                )
                for t in style.get("traces", [])
            ],
            legend=LegendStyle.from_dict(style.get("legend", {})),
            grid=style.get("grid", False),
        )


__all__ = [
    "THEME_VERSION",
    "ColorScheme",
    "Typography",
    "LineDefaults",
    "MarkerDefaults",
    "TraceStyle",
    "LegendStyle",
    "PlotTheme",
]


# EOF
