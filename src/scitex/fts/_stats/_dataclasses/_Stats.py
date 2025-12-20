#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_stats/_dataclasses/_Stats.py

"""
Stats - Statistical analysis results with provenance and GUI support.

This module provides the single source of truth for statistical result schemas,
supporting both:
1. Bundle storage (data provenance, test results)
2. GUI annotation (positioning, styling for interactive editing)
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

STATS_VERSION = "1.1.0"

# Type aliases for GUI support
PositionMode = Literal["absolute", "relative_to_plot", "above_whisker", "auto"]
UnitType = Literal["mm", "px", "inch", "data"]
SymbolStyle = Literal["asterisk", "text", "bracket", "compact", "detailed", "publication"]


@dataclass
class DataRef:
    """Reference to data used in analysis."""

    path: str
    columns: Optional[List[str]] = None
    filter: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"path": self.path}
        if self.columns:
            result["columns"] = self.columns
        if self.filter:
            result["filter"] = self.filter
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataRef":
        return cls(
            path=data.get("path", ""),
            columns=data.get("columns"),
            filter=data.get("filter"),
        )


# =============================================================================
# GUI Position and Styling Classes (merged from scitex.schema._stats)
# =============================================================================


@dataclass
class Position:
    """
    Position specification with unit support for GUI integration.

    Supports multiple coordinate systems for flexibility across
    matplotlib (mm), Fabric.js (px), and data coordinates.
    """

    x: float
    y: float
    unit: UnitType = "mm"
    relative_to: Optional[str] = None  # Plot ID or "axes"
    offset: Optional[Dict[str, float]] = None  # {"dx": 0, "dy": 0}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "unit": self.unit,
            "relative_to": self.relative_to,
            "offset": self.offset,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        return cls(**data)

    def to_mm(self, dpi: float = 300.0) -> "Position":
        """Convert position to mm (for matplotlib)."""
        if self.unit == "mm":
            return self
        elif self.unit == "px":
            mm_per_px = 25.4 / dpi
            return Position(
                x=self.x * mm_per_px,
                y=self.y * mm_per_px,
                unit="mm",
                relative_to=self.relative_to,
                offset=self.offset,
            )
        elif self.unit == "inch":
            return Position(
                x=self.x * 25.4,
                y=self.y * 25.4,
                unit="mm",
                relative_to=self.relative_to,
                offset=self.offset,
            )
        return self

    def to_px(self, dpi: float = 300.0) -> "Position":
        """Convert position to px (for Fabric.js canvas)."""
        mm_pos = self.to_mm(dpi)
        px_per_mm = dpi / 25.4
        return Position(
            x=mm_pos.x * px_per_mm,
            y=mm_pos.y * px_per_mm,
            unit="px",
            relative_to=self.relative_to,
            offset=self.offset,
        )


@dataclass
class StatStyling:
    """Styling configuration for statistical annotation display."""

    font_size_pt: float = 7.0
    font_family: str = "Arial"
    color: str = "#000000"
    symbol_style: SymbolStyle = "asterisk"
    line_width_mm: Optional[float] = None
    bracket_height_mm: Optional[float] = None
    theme: Literal["light", "dark", "auto"] = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatStyling":
        return cls(**data)

    def get_theme_color(self, is_dark: bool = False) -> str:
        """Get appropriate color for theme."""
        if self.theme == "auto":
            return "#ffffff" if is_dark else "#000000"
        elif self.theme == "dark":
            return "#ffffff"
        return "#000000"


@dataclass
class StatPositioning:
    """Position configuration for GUI-ready annotation placement."""

    mode: PositionMode = "auto"
    position: Optional[Position] = None
    preferred_corner: Optional[str] = None  # "top-right", "bottom-left", etc.
    avoid_overlap: bool = True
    min_distance_mm: float = 2.0
    anchor_to: Optional[str] = None  # "plot_center", "whisker_top", etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "position": self.position.to_dict() if self.position else None,
            "preferred_corner": self.preferred_corner,
            "avoid_overlap": self.avoid_overlap,
            "min_distance_mm": self.min_distance_mm,
            "anchor_to": self.anchor_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatPositioning":
        data_copy = data.copy()
        if data_copy.get("position"):
            data_copy["position"] = Position.from_dict(data_copy["position"])
        return cls(**data_copy)


@dataclass
class EffectSize:
    """Effect size with confidence interval."""

    name: str  # cohens_d, hedges_g, eta_squared, r_squared, etc.
    value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    ci_level: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        result = {"name": self.name, "value": self.value}
        if self.ci_lower is not None:
            result["ci_lower"] = self.ci_lower
        if self.ci_upper is not None:
            result["ci_upper"] = self.ci_upper
        if self.ci_level != 0.95:
            result["ci_level"] = self.ci_level
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EffectSize":
        return cls(
            name=data.get("name", ""),
            value=data.get("value", 0.0),
            ci_lower=data.get("ci_lower"),
            ci_upper=data.get("ci_upper"),
            ci_level=data.get("ci_level", 0.95),
        )


@dataclass
class StatMethod:
    """Statistical method specification."""

    name: str  # t-test, anova, chi-squared, correlation, etc.
    variant: Optional[str] = None  # independent, paired, one-way, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {"name": self.name}
        if self.variant:
            result["variant"] = self.variant
        if self.parameters:
            result["parameters"] = self.parameters
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatMethod":
        return cls(
            name=data.get("name", ""),
            variant=data.get("variant"),
            parameters=data.get("parameters", {}),
        )


@dataclass
class StatResult:
    """Statistical test result."""

    statistic: float
    statistic_name: str  # t, F, chi2, r, etc.
    p_value: float
    df: Optional[float] = None
    effect_size: Optional[EffectSize] = None
    significant: Optional[bool] = None
    alpha: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "statistic": self.statistic,
            "statistic_name": self.statistic_name,
            "p_value": self.p_value,
        }
        if self.df is not None:
            result["df"] = self.df
        if self.effect_size:
            result["effect_size"] = self.effect_size.to_dict()
        if self.significant is not None:
            result["significant"] = self.significant
        if self.alpha != 0.05:
            result["alpha"] = self.alpha
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatResult":
        effect_size = None
        if "effect_size" in data:
            effect_size = EffectSize.from_dict(data["effect_size"])
        return cls(
            statistic=data.get("statistic", 0.0),
            statistic_name=data.get("statistic_name", ""),
            p_value=data.get("p_value", 1.0),
            df=data.get("df"),
            effect_size=effect_size,
            significant=data.get("significant"),
            alpha=data.get("alpha", 0.05),
        )


@dataclass
class StatDisplay:
    """How to display the statistical result (with GUI support)."""

    show_stars: bool = True
    show_p_value: bool = True
    show_effect_size: bool = False
    bracket_groups: Optional[List[str]] = None
    # Enhanced GUI support
    positioning: Optional[StatPositioning] = None
    styling: Optional[StatStyling] = None
    plot_id: Optional[str] = None  # Associated plot in TrackingMixin

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "show_stars": self.show_stars,
            "show_p_value": self.show_p_value,
            "show_effect_size": self.show_effect_size,
        }
        if self.bracket_groups:
            result["bracket_groups"] = self.bracket_groups
        if self.positioning:
            result["positioning"] = self.positioning.to_dict()
        if self.styling:
            result["styling"] = self.styling.to_dict()
        if self.plot_id:
            result["plot_id"] = self.plot_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatDisplay":
        positioning = None
        if "positioning" in data and data["positioning"]:
            positioning = StatPositioning.from_dict(data["positioning"])
        styling = None
        if "styling" in data and data["styling"]:
            styling = StatStyling.from_dict(data["styling"])
        return cls(
            show_stars=data.get("show_stars", True),
            show_p_value=data.get("show_p_value", True),
            show_effect_size=data.get("show_effect_size", False),
            bracket_groups=data.get("bracket_groups"),
            positioning=positioning,
            styling=styling,
            plot_id=data.get("plot_id"),
        )


@dataclass
class Analysis:
    """Complete analysis record with provenance."""

    result_id: str
    method: StatMethod
    results: StatResult
    data_refs: List[DataRef] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    display: Optional[StatDisplay] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "result_id": self.result_id,
            "method": self.method.to_dict(),
            "results": self.results.to_dict(),
        }
        if self.data_refs:
            result["data_refs"] = [d.to_dict() for d in self.data_refs]
        if self.inputs:
            result["inputs"] = self.inputs
        if self.display:
            result["display"] = self.display.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Analysis":
        display = None
        if "display" in data:
            display = StatDisplay.from_dict(data["display"])
        return cls(
            result_id=data.get("result_id", ""),
            method=StatMethod.from_dict(data.get("method", {})),
            results=StatResult.from_dict(data.get("results", {})),
            data_refs=[DataRef.from_dict(d) for d in data.get("data_refs", [])],
            inputs=data.get("inputs", {}),
            display=display,
        )


@dataclass
class Stats:
    """Complete statistics specification for a bundle.

    Stored in stats/stats.json.
    """

    analyses: List[Analysis] = field(default_factory=list)
    software: Dict[str, str] = field(default_factory=dict)

    # Schema metadata
    schema_name: str = "fsb.stats"
    schema_version: str = STATS_VERSION

    def to_dict(self) -> Dict[str, Any]:
        result = {"analyses": [a.to_dict() for a in self.analyses]}
        if self.software:
            result["software"] = self.software
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Stats":
        return cls(
            analyses=[Analysis.from_dict(a) for a in data.get("analyses", [])],
            software=data.get("software", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Stats":
        return cls.from_dict(json.loads(json_str))


__all__ = [
    # Version
    "STATS_VERSION",
    # Type aliases
    "PositionMode",
    "UnitType",
    "SymbolStyle",
    # Data reference
    "DataRef",
    # GUI classes (merged from scitex.schema._stats)
    "Position",
    "StatStyling",
    "StatPositioning",
    # Core stats classes
    "EffectSize",
    "StatMethod",
    "StatResult",
    "StatDisplay",
    "Analysis",
    "Stats",
]

# EOF
