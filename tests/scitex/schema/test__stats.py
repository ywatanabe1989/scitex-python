#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./tests/scitex/schema/test__stats.py
# Time-stamp: "2024-12-09 09:30:00 (ywatanabe)"
"""Tests for scitex.schema._stats module - the central stats schema."""

import json

import pytest

# Some tests import from scitex.stats which requires matplotlib
pytest.importorskip("matplotlib")


class TestSchemaOwnership:
    """Test that schema._stats is the source of truth."""

    def test_schema_owns_stat_result(self):
        """Test StatResult is defined in schema._stats."""
        from scitex.schema._stats import StatResult

        # Should have the module path in schema
        assert "scitex.schema._stats" in StatResult.__module__

    def test_schema_owns_position(self):
        """Test Position is defined in schema._stats."""
        from scitex.schema._stats import Position

        assert "scitex.schema._stats" in Position.__module__

    def test_schema_owns_stat_styling(self):
        """Test StatStyling is defined in schema._stats."""
        from scitex.schema._stats import StatStyling

        assert "scitex.schema._stats" in StatStyling.__module__

    def test_schema_owns_stat_positioning(self):
        """Test StatPositioning is defined in schema._stats."""
        from scitex.schema._stats import StatPositioning

        assert "scitex.schema._stats" in StatPositioning.__module__


class TestBackwardCompatibility:
    """Test backward compatibility with stats._schema imports."""

    def test_stats_schema_reexports_stat_result(self):
        """Test stats._schema re-exports StatResult from schema."""
        from scitex.schema._stats import StatResult as SchemaStatResult
        from scitex.stats._schema import StatResult as StatsStatResult

        # They should be the exact same class
        assert SchemaStatResult is StatsStatResult

    def test_stats_schema_reexports_position(self):
        """Test stats._schema re-exports Position from schema."""
        from scitex.schema._stats import Position as SchemaPosition
        from scitex.stats._schema import Position as StatsPosition

        assert SchemaPosition is StatsPosition

    def test_stats_schema_reexports_styling(self):
        """Test stats._schema re-exports StatStyling from schema."""
        from scitex.schema._stats import StatStyling as SchemaStyling
        from scitex.stats._schema import StatStyling as StatsStyling

        assert SchemaStyling is StatsStyling

    def test_stats_schema_reexports_positioning(self):
        """Test stats._schema re-exports StatPositioning from schema."""
        from scitex.schema._stats import StatPositioning as SchemaPositioning
        from scitex.stats._schema import StatPositioning as StatsPositioning

        assert SchemaPositioning is StatsPositioning

    def test_stats_schema_reexports_create_function(self):
        """Test stats._schema re-exports create_stat_result from schema."""
        from scitex.schema._stats import create_stat_result as schema_create
        from scitex.stats._schema import create_stat_result as stats_create

        assert schema_create is stats_create

    def test_type_aliases_reexported(self):
        """Test type aliases are re-exported."""
        from scitex.schema._stats import PositionMode, UnitType, SymbolStyle
        from scitex.stats._schema import (
            PositionMode as PM2,
            UnitType as UT2,
            SymbolStyle as SS2,
        )

        # Type aliases may be recreated, but should have same value
        assert PositionMode == PM2
        assert UnitType == UT2
        assert SymbolStyle == SS2


class TestPosition:
    """Tests for Position class."""

    def test_default_unit(self):
        """Test default unit is mm."""
        from scitex.schema import Position

        pos = Position(x=10, y=20)
        assert pos.unit == "mm"

    def test_to_mm_already_mm(self):
        """Test to_mm returns self when already mm."""
        from scitex.schema import Position

        pos = Position(x=10, y=20, unit="mm")
        pos_mm = pos.to_mm()
        assert pos_mm is pos

    def test_to_mm_from_px(self):
        """Test conversion from px to mm."""
        from scitex.schema import Position

        pos = Position(x=300, y=300, unit="px")
        pos_mm = pos.to_mm(dpi=300)

        # 300 px at 300 dpi = 1 inch = 25.4 mm
        assert pos_mm.unit == "mm"
        assert abs(pos_mm.x - 25.4) < 0.01
        assert abs(pos_mm.y - 25.4) < 0.01

    def test_to_mm_from_inch(self):
        """Test conversion from inch to mm."""
        from scitex.schema import Position

        pos = Position(x=1, y=2, unit="inch")
        pos_mm = pos.to_mm()

        assert pos_mm.unit == "mm"
        assert abs(pos_mm.x - 25.4) < 0.01
        assert abs(pos_mm.y - 50.8) < 0.01

    def test_to_px(self):
        """Test conversion to px."""
        from scitex.schema import Position

        pos = Position(x=25.4, y=25.4, unit="mm")
        pos_px = pos.to_px(dpi=300)

        # 25.4 mm = 1 inch = 300 px at 300 dpi
        assert pos_px.unit == "px"
        assert abs(pos_px.x - 300) < 0.01
        assert abs(pos_px.y - 300) < 0.01

    def test_to_dict(self):
        """Test serialization to dict."""
        from scitex.schema import Position

        pos = Position(x=10, y=20, unit="mm", relative_to="plot_0")
        d = pos.to_dict()

        assert d["x"] == 10
        assert d["y"] == 20
        assert d["unit"] == "mm"
        assert d["relative_to"] == "plot_0"

    def test_from_dict(self):
        """Test deserialization from dict."""
        from scitex.schema import Position

        data = {"x": 15, "y": 25, "unit": "px", "relative_to": "axes"}
        pos = Position.from_dict(data)

        assert pos.x == 15
        assert pos.y == 25
        assert pos.unit == "px"
        assert pos.relative_to == "axes"


class TestStatStyling:
    """Tests for StatStyling class."""

    def test_defaults(self):
        """Test default styling values."""
        from scitex.schema import StatStyling

        style = StatStyling()
        assert style.font_size_pt == 7.0
        assert style.font_family == "Arial"
        assert style.color == "#000000"
        assert style.symbol_style == "asterisk"
        assert style.theme == "auto"

    def test_get_theme_color_auto_light(self):
        """Test auto theme returns black for light mode."""
        from scitex.schema import StatStyling

        style = StatStyling(theme="auto")
        assert style.get_theme_color(is_dark=False) == "#000000"

    def test_get_theme_color_auto_dark(self):
        """Test auto theme returns white for dark mode."""
        from scitex.schema import StatStyling

        style = StatStyling(theme="auto")
        assert style.get_theme_color(is_dark=True) == "#ffffff"

    def test_get_theme_color_explicit(self):
        """Test explicit theme colors."""
        from scitex.schema import StatStyling

        light = StatStyling(theme="light")
        dark = StatStyling(theme="dark")

        assert light.get_theme_color() == "#000000"
        assert dark.get_theme_color() == "#ffffff"


class TestStatPositioning:
    """Tests for StatPositioning class."""

    def test_defaults(self):
        """Test default positioning values."""
        from scitex.schema import StatPositioning

        pos = StatPositioning()
        assert pos.mode == "auto"
        assert pos.position is None
        assert pos.avoid_overlap is True
        assert pos.min_distance_mm == 2.0

    def test_with_position(self):
        """Test positioning with explicit position."""
        from scitex.schema import StatPositioning, Position

        pos = StatPositioning(
            mode="absolute",
            position=Position(x=50, y=100, unit="mm"),
        )
        assert pos.mode == "absolute"
        assert pos.position.x == 50

    def test_roundtrip(self):
        """Test dict roundtrip."""
        from scitex.schema import StatPositioning, Position

        original = StatPositioning(
            mode="relative_to_plot",
            position=Position(x=10, y=20, unit="mm"),
            preferred_corner="top-right",
        )
        restored = StatPositioning.from_dict(original.to_dict())

        assert original.mode == restored.mode
        assert original.position.x == restored.position.x
        assert original.preferred_corner == restored.preferred_corner


class TestStatResult:
    """Tests for StatResult class."""

    def test_minimal_creation(self):
        """Test creating StatResult with minimal fields."""
        from scitex.schema import StatResult

        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 2.5},
            p_value=0.015,
            stars="*",
        )
        assert result.test_type == "t-test"
        assert result.p_value == 0.015
        assert result.stars == "*"

    def test_auto_creates_styling(self):
        """Test StatResult auto-creates default styling."""
        from scitex.schema import StatResult, StatStyling

        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 2.5},
            p_value=0.015,
            stars="*",
        )
        assert result.styling is not None
        assert isinstance(result.styling, StatStyling)

    def test_auto_creates_positioning(self):
        """Test StatResult auto-creates default positioning."""
        from scitex.schema import StatResult, StatPositioning

        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 2.5},
            p_value=0.015,
            stars="*",
        )
        assert result.positioning is not None
        assert isinstance(result.positioning, StatPositioning)

    def test_auto_creates_timestamp(self):
        """Test StatResult auto-creates timestamp."""
        from scitex.schema import StatResult

        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 2.5},
            p_value=0.015,
            stars="*",
        )
        assert result.created_at is not None

    def test_format_text_compact(self):
        """Test compact formatting."""
        from scitex.schema import StatResult

        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.001,
            stars="***",
        )
        assert result.format_text("compact") == "r = 0.850***"

    def test_format_text_asterisk(self):
        """Test asterisk formatting."""
        from scitex.schema import StatResult

        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 2.5},
            p_value=0.001,
            stars="***",
        )
        assert result.format_text("asterisk") == "***"

    def test_format_text_publication(self):
        """Test publication formatting."""
        from scitex.schema import StatResult

        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.0001,  # Use very small p for p < 0.001
            stars="***",
        )
        assert result.format_text("publication") == "(r = 0.85, p < 0.001)"

    def test_format_text_ns(self):
        """Test formatting for non-significant results."""
        from scitex.schema import StatResult

        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 0.5},
            p_value=0.6,
            stars="ns",
        )
        assert result.format_text("asterisk") == "ns"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from scitex.schema import StatResult

        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 2.5},
            p_value=0.015,
            stars="*",
        )
        d = result.to_dict()

        assert d["test_type"] == "t-test"
        assert d["p_value"] == 0.015
        assert d["statistic"]["name"] == "t"

    def test_to_json(self):
        """Test JSON serialization."""
        from scitex.schema import StatResult

        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 2.5},
            p_value=0.015,
            stars="*",
        )
        json_str = result.to_json()
        data = json.loads(json_str)

        assert data["test_type"] == "t-test"
        assert data["p_value"] == 0.015

    def test_from_json(self):
        """Test JSON deserialization."""
        from scitex.schema import StatResult

        original = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.001,
            stars="***",
        )
        json_str = original.to_json()
        restored = StatResult.from_json(json_str)

        assert restored.test_type == original.test_type
        assert restored.p_value == original.p_value
        assert restored.stars == original.stars

    def test_to_annotation_dict(self):
        """Test conversion to annotation dict for GUI."""
        from scitex.schema import StatResult

        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 2.5},
            p_value=0.015,
            stars="*",
            plot_id="stat_001",
        )
        ann = result.to_annotation_dict()

        assert ann["type"] == "stat"
        assert ann["id"] == "stat_001"
        assert ann["statResult"]["p_value"] == 0.015

    def test_get_interpretation_correlation(self):
        """Test interpretation for correlation."""
        from scitex.schema import StatResult

        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.001,
            stars="***",
        )
        interp = result.get_interpretation()

        assert "Strong" in interp
        assert "positive" in interp
        assert "correlation" in interp

    def test_effect_size(self):
        """Test StatResult with effect size."""
        from scitex.schema import StatResult

        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.5},
            p_value=0.001,
            stars="***",
            effect_size={"name": "cohens_d", "value": 0.8, "interpretation": "large"},
        )
        detailed = result.format_text("detailed")

        assert "cohens_d" in detailed
        assert "0.80" in detailed


class TestCreateStatResult:
    """Tests for create_stat_result convenience function."""

    def test_basic_creation(self):
        """Test basic StatResult creation."""
        from scitex.schema import create_stat_result

        result = create_stat_result(
            test_type="pearson",
            statistic_name="r",
            statistic_value=0.85,
            p_value=0.001,
        )
        assert result.test_type == "pearson"
        assert result.test_category == "correlation"
        assert result.statistic["name"] == "r"
        assert result.stars == "***"

    def test_auto_category_detection(self):
        """Test automatic category detection."""
        from scitex.schema import create_stat_result

        # Correlation
        result = create_stat_result("spearman", "rho", 0.7, 0.001)
        assert result.test_category == "correlation"

        # Parametric
        result = create_stat_result("t-test", "t", 2.5, 0.01)
        assert result.test_category == "parametric"

    def test_stars_generation(self):
        """Test automatic stars generation."""
        from scitex.schema import create_stat_result

        # p < 0.001 -> ***
        r1 = create_stat_result("t-test", "t", 3.5, 0.0001)
        assert r1.stars == "***"

        # p < 0.01 -> **
        r2 = create_stat_result("t-test", "t", 2.5, 0.005)
        assert r2.stars == "**"

        # p < 0.05 -> *
        r3 = create_stat_result("t-test", "t", 2.0, 0.04)
        assert r3.stars == "*"


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/schema/_stats.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-09 20:42:10 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/schema/_stats.py
#
# # Time-stamp: "2024-12-09 09:15:00 (ywatanabe)"
# """
# Statistical Result Schema - Central Source of Truth.
#
# This module defines the canonical schema for statistical test results
# that integrates with:
# - scitex.plt (TrackingMixin, metadata embedding)
# - scitex.canvas (JSON serialization, FigureModel)
# - scitex-cloud GUI (Fabric.js canvas, properties panel, positioning)
# - scitex.bridge (cross-module adapters)
#
# This schema supports multiple coordinate systems, units, and position modes
# to enable GUI-based adjustment while maintaining publication-ready output.
#
# Note: This is the SOURCE OF TRUTH. Other modules (stats, plt, vis) should
# import from here, not define their own versions.
# """
#
# from dataclasses import dataclass
# from dataclasses import field, asdict
# from typing import Optional
# from typing import Dict, Any, List, Literal, Union
# from datetime import datetime
# import json
# import numpy as np
#
#
# # Schema version for statistical result schemas
# STATS_SCHEMA_VERSION = "0.1.0"
#
#
# # =============================================================================
# # Type Aliases
# # =============================================================================
#
# PositionMode = Literal["absolute", "relative_to_plot", "above_whisker", "auto"]
# UnitType = Literal["mm", "px", "inch", "data"]
# SymbolStyle = Literal[
#     "asterisk", "text", "bracket", "compact", "detailed", "publication"
# ]
#
#
# # =============================================================================
# # Position Schema
# # =============================================================================
#
#
# @dataclass
# class Position:
#     """
#     Position specification with unit support for GUI integration.
#
#     Supports multiple coordinate systems for flexibility across
#     matplotlib (mm), Fabric.js (px), and data coordinates.
#
#     Parameters
#     ----------
#     x : float
#         X coordinate
#     y : float
#         Y coordinate
#     unit : UnitType
#         Coordinate unit ("mm", "px", "inch", "data")
#     relative_to : str, optional
#         Plot ID or "axes" for relative positioning
#     offset : dict, optional
#         Offset values {"dx": 0, "dy": 0}
#
#     Examples
#     --------
#     >>> pos = Position(x=10, y=20, unit="mm")
#     >>> pos_px = pos.to_px(dpi=300)
#     >>> pos_px.unit
#     'px'
#     """
#
#     x: float
#     y: float
#     unit: UnitType = "mm"
#
#     # For relative positioning (GUI anchoring)
#     relative_to: Optional[str] = None  # Plot ID or "axes"
#     offset: Optional[Dict[str, float]] = None  # {"dx": 0, "dy": 0}
#
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary."""
#         return {
#             "x": self.x,
#             "y": self.y,
#             "unit": self.unit,
#             "relative_to": self.relative_to,
#             "offset": self.offset,
#         }
#
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "Position":
#         """Create from dictionary."""
#         return cls(**data)
#
#     def to_mm(self, dpi: float = 300.0) -> "Position":
#         """Convert position to mm (for matplotlib)."""
#         if self.unit == "mm":
#             return self
#         elif self.unit == "px":
#             # Convert px to mm using DPI
#             mm_per_px = 25.4 / dpi
#             return Position(
#                 x=self.x * mm_per_px,
#                 y=self.y * mm_per_px,
#                 unit="mm",
#                 relative_to=self.relative_to,
#                 offset=self.offset,
#             )
#         elif self.unit == "inch":
#             # Convert inch to mm
#             return Position(
#                 x=self.x * 25.4,
#                 y=self.y * 25.4,
#                 unit="mm",
#                 relative_to=self.relative_to,
#                 offset=self.offset,
#             )
#         return self
#
#     def to_px(self, dpi: float = 300.0) -> "Position":
#         """Convert position to px (for Fabric.js canvas)."""
#         mm_pos = self.to_mm(dpi)
#         px_per_mm = dpi / 25.4
#         return Position(
#             x=mm_pos.x * px_per_mm,
#             y=mm_pos.y * px_per_mm,
#             unit="px",
#             relative_to=self.relative_to,
#             offset=self.offset,
#         )
#
#
# # =============================================================================
# # Styling Schema
# # =============================================================================
#
#
# @dataclass
# class StatStyling:
#     """
#     Styling configuration for statistical annotation display.
#
#     Parameters
#     ----------
#     font_size_pt : float
#         Font size in points (default: 7.0)
#     font_family : str
#         Font family name (default: "Arial")
#     color : str
#         Text color, supports hex codes (default: "#000000")
#     symbol_style : SymbolStyle
#         How to display significance (default: "asterisk")
#     line_width_mm : float, optional
#         Line width for brackets in mm
#     bracket_height_mm : float, optional
#         Bracket height in mm
#     theme : str
#         Color theme ("light", "dark", "auto")
#     """
#
#     # Typography
#     font_size_pt: float = 7.0
#     font_family: str = "Arial"
#     color: str = "#000000"  # Supports theme colors
#
#     # Symbol representation
#     symbol_style: SymbolStyle = "asterisk"
#
#     # For bracket-style comparisons
#     line_width_mm: Optional[float] = None
#     bracket_height_mm: Optional[float] = None
#
#     # Theme support
#     theme: Literal["light", "dark", "auto"] = "auto"
#
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary."""
#         return asdict(self)
#
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "StatStyling":
#         """Create from dictionary."""
#         return cls(**data)
#
#     def get_theme_color(self, is_dark: bool = False) -> str:
#         """Get appropriate color for theme."""
#         if self.theme == "auto":
#             return "#ffffff" if is_dark else "#000000"
#         elif self.theme == "dark":
#             return "#ffffff"
#         elif self.theme == "light":
#             return "#000000"
#         return self.color
#
#
# # =============================================================================
# # Positioning Schema
# # =============================================================================
#
#
# @dataclass
# class StatPositioning:
#     """
#     Position configuration for GUI-ready annotation placement.
#
#     Parameters
#     ----------
#     mode : PositionMode
#         Positioning mode ("absolute", "relative_to_plot", "above_whisker", "auto")
#     position : Position, optional
#         Explicit position coordinates
#     preferred_corner : str, optional
#         Preferred placement ("top-right", "bottom-left", etc.)
#     avoid_overlap : bool
#         Whether to auto-adjust to avoid overlaps (default: True)
#     min_distance_mm : float
#         Minimum distance from plot elements in mm (default: 2.0)
#     anchor_to : str, optional
#         Anchor point ("plot_center", "whisker_top", etc.)
#     """
#
#     mode: PositionMode = "auto"
#     position: Optional[Position] = None
#
#     # GUI hints for smart positioning
#     preferred_corner: Optional[str] = None  # "top-right", "bottom-left", etc.
#     avoid_overlap: bool = True
#     min_distance_mm: float = 2.0  # Minimum distance from plot elements
#
#     # Anchoring (for relative positioning)
#     anchor_to: Optional[str] = None  # "plot_center", "whisker_top", etc.
#
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary."""
#         return {
#             "mode": self.mode,
#             "position": self.position.to_dict() if self.position else None,
#             "preferred_corner": self.preferred_corner,
#             "avoid_overlap": self.avoid_overlap,
#             "min_distance_mm": self.min_distance_mm,
#             "anchor_to": self.anchor_to,
#         }
#
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "StatPositioning":
#         """Create from dictionary."""
#         data_copy = data.copy()
#         if data_copy.get("position"):
#             data_copy["position"] = Position.from_dict(data_copy["position"])
#         return cls(**data_copy)
#
#
# # =============================================================================
# # StatResult Schema
# # =============================================================================
#
#
# @dataclass
# class StatResult:
#     """
#     Standardized statistical test result with GUI-ready metadata.
#
#     This is the central schema for statistical results that integrates with:
#     - scitex.plt: Automatic annotation via TrackingMixin
#     - scitex.canvas: JSON serialization for FigureModel
#     - scitex-cloud GUI: Fabric.js canvas positioning and properties panel
#     - scitex.bridge: Cross-module adapters
#
#     Parameters
#     ----------
#     test_type : str
#         Type of statistical test ("t-test", "pearson", "anova", etc.)
#     test_category : str
#         Category ("parametric", "non-parametric", "correlation", "other")
#     statistic : dict
#         Test statistic {"name": "t", "value": 3.45}
#     p_value : float
#         P-value from the test
#     stars : str
#         Significance stars ("***", "**", "*", "ns", "")
#     effect_size : dict, optional
#         Effect size information
#     correction : dict, optional
#         Multiple comparison correction info
#     samples : dict, optional
#         Sample information (n, mean, std per group)
#     assumptions : dict, optional
#         Assumption test results
#     ci_95 : list, optional
#         95% confidence interval
#     positioning : StatPositioning, optional
#         GUI positioning configuration
#     styling : StatStyling, optional
#         Display styling configuration
#     extra : dict, optional
#         Additional test-specific data
#     created_at : str, optional
#         ISO timestamp of creation
#     software_version : str, optional
#         Version of software that created this
#     plot_id : str, optional
#         Associated plot ID in TrackingMixin
#
#     Examples
#     --------
#     >>> result = StatResult(
#     ...     test_type="pearson",
#     ...     test_category="correlation",
#     ...     statistic={"name": "r", "value": 0.85},
#     ...     p_value=0.001,
#     ...     stars="***"
#     ... )
#     >>> result.format_text("compact")
#     'r = 0.850***'
#     >>> result.format_text("publication")
#     '(r = 0.85, p < 0.001)'
#     """
#
#     # Core test information
#     test_type: str  # "t-test", "pearson", "anova", etc.
#     test_category: str  # "parametric", "non-parametric", "correlation"
#
#     # Primary results
#     statistic: Dict[str, Union[float, str]]  # {"name": "t", "value": 3.45}
#     p_value: float
#     stars: str  # "***", "**", "*", "ns", ""
#
#     # Effect size (optional)
#     effect_size: Optional[Dict[str, Any]] = None
#     # Example: {"name": "cohens_d", "value": 0.85,
#     #           "interpretation": "large", "ci_95": [0.42, 1.28]}
#
#     # Multiple comparison correction (optional)
#     correction: Optional[Dict[str, Any]] = None
#     # Example: {"method": "bonferroni", "n_comparisons": 10,
#     #           "corrected_p": 0.010, "alpha": 0.05}
#
#     # Sample information
#     samples: Optional[Dict[str, Any]] = None
#     # Example: {"group1": {"n": 30, "mean": 5.2, "std": 1.1},
#     #           "group2": {"n": 32, "mean": 6.8, "std": 1.3}}
#
#     # Statistical assumptions testing
#     assumptions: Optional[Dict[str, Dict]] = None
#     # Example: {"normality": {"test": "shapiro", "passed": True, "p": 0.23},
#     #           "homogeneity": {"test": "levene", "passed": True, "p": 0.45}}
#
#     # Confidence intervals
#     ci_95: Optional[List[float]] = None
#     # Example: [0.42, 1.28] for effect size or correlation
#
#     # GUI-ready positioning
#     positioning: Optional[StatPositioning] = None
#
#     # Display styling
#     styling: Optional[StatStyling] = None
#
#     # Additional test-specific data
#     extra: Optional[Dict[str, Any]] = None
#     # For test-specific outputs (degrees of freedom, test alternatives, etc.)
#
#     # Metadata
#     created_at: Optional[str] = None
#     software_version: Optional[str] = None
#     plot_id: Optional[str] = None  # Associated plot in TrackingMixin
#
#     # Schema identification for forward/backward compatibility
#     scitex_schema: str = "scitex.schema.stats"
#     scitex_schema_version: str = STATS_SCHEMA_VERSION
#
#     def __post_init__(self):
#         """Initialize default values."""
#         if self.created_at is None:
#             self.created_at = datetime.now().isoformat()
#
#         if self.styling is None:
#             self.styling = StatStyling()
#
#         if self.positioning is None:
#             self.positioning = StatPositioning()
#
#     def to_dict(self) -> Dict[str, Any]:
#         """
#         Convert to dictionary for JSON serialization.
#
#         Returns
#         -------
#         dict
#             Dictionary representation with nested objects converted
#         """
#         data = asdict(self)
#
#         # Convert numpy types to native Python types
#         def convert_numpy(obj):
#             if isinstance(obj, np.integer):
#                 return int(obj)
#             elif isinstance(obj, np.floating):
#                 return float(obj)
#             elif isinstance(obj, np.ndarray):
#                 return obj.tolist()
#             elif isinstance(obj, dict):
#                 return {k: convert_numpy(v) for k, v in obj.items()}
#             elif isinstance(obj, list):
#                 return [convert_numpy(item) for item in obj]
#             return obj
#
#         return convert_numpy(data)
#
#     def to_json(self, indent: int = 2) -> str:
#         """
#         Convert to JSON string.
#
#         Parameters
#         ----------
#         indent : int
#             Number of spaces for indentation (default: 2)
#
#         Returns
#         -------
#         str
#             JSON string representation
#         """
#         return json.dumps(self.to_dict(), indent=indent)
#
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "StatResult":
#         """
#         Create from dictionary.
#
#         Parameters
#         ----------
#         data : dict
#             Dictionary representation
#
#         Returns
#         -------
#         StatResult
#             StatResult instance
#         """
#         data_copy = data.copy()
#
#         # Convert nested objects
#         if "positioning" in data_copy and data_copy["positioning"]:
#             data_copy["positioning"] = StatPositioning.from_dict(
#                 data_copy["positioning"]
#             )
#
#         if "styling" in data_copy and data_copy["styling"]:
#             data_copy["styling"] = StatStyling.from_dict(data_copy["styling"])
#
#         return cls(**data_copy)
#
#     @classmethod
#     def from_json(cls, json_str: str) -> "StatResult":
#         """
#         Create from JSON string.
#
#         Parameters
#         ----------
#         json_str : str
#             JSON string representation
#
#         Returns
#         -------
#         StatResult
#             StatResult instance
#         """
#         return cls.from_dict(json.loads(json_str))
#
#     def format_text(self, style: str = "compact") -> str:
#         """
#         Format statistical result as text for display.
#
#         Parameters
#         ----------
#         style : str
#             Formatting style. Options:
#             - "compact": "r = 0.850***"
#             - "asterisk": "***" (stars only)
#             - "text": "p = 0.001"
#             - "detailed": "r = 0.850, p = 1.000e-03, d = 1.23"
#             - "publication": "(r = 0.85, p < 0.001)"
#             - "bracket": For bracket-style display (returns stars or empty)
#
#         Returns
#         -------
#         str
#             Formatted text
#
#         Examples
#         --------
#         >>> result.format_text("compact")
#         'r = 0.850***'
#
#         >>> result.format_text("publication")
#         '(r = 0.85, p < 0.001)'
#         """
#         stat_name = self.statistic.get("name", "stat")
#         stat_value = self.statistic.get("value", 0.0)
#
#         if style == "compact":
#             return f"{stat_name} = {stat_value:.3f}{self.stars}"
#
#         elif style == "asterisk":
#             return self.stars if self.stars != "ns" else "ns"
#
#         elif style == "text":
#             return f"p = {self.p_value:.3f}"
#
#         elif style == "detailed":
#             parts = [f"{stat_name} = {stat_value:.3f}"]
#             parts.append(f"p = {self.p_value:.3e}")
#
#             if self.effect_size:
#                 es_name = self.effect_size.get("name", "d")
#                 es_value = self.effect_size.get("value", 0.0)
#                 parts.append(f"{es_name} = {es_value:.2f}")
#
#             return ", ".join(parts)
#
#         elif style == "publication":
#             p_text = self._format_p_publication()
#             return f"({stat_name} = {stat_value:.2f}, {p_text})"
#
#         elif style == "bracket":
#             # For bracket display, return stars only
#             return self.stars if self.stars != "ns" else ""
#
#         return f"{stat_name} = {stat_value:.3f}{self.stars}"
#
#     def _format_p_publication(self) -> str:
#         """
#         Format p-value for publication style.
#
#         Returns
#         -------
#         str
#             Formatted p-value like "p < 0.001" or "p = 0.023"
#         """
#         if self.p_value < 0.001:
#             return "p < 0.001"
#         elif self.p_value < 0.01:
#             return "p < 0.01"
#         elif self.p_value < 0.05:
#             return "p < 0.05"
#         else:
#             return f"p = {self.p_value:.3f}"
#
#     def get_interpretation(self) -> str:
#         """
#         Get human-readable interpretation of results.
#
#         Returns
#         -------
#         str
#             Interpretation text
#
#         Examples
#         --------
#         >>> result.get_interpretation()
#         'Strong positive correlation (r=0.85, p<0.001)'
#         """
#         if self.test_category == "correlation":
#             r = self.statistic.get("value", 0)
#             direction = "positive" if r > 0 else "negative"
#
#             if abs(r) > 0.7:
#                 strength = "Strong"
#             elif abs(r) > 0.5:
#                 strength = "Moderate"
#             elif abs(r) > 0.3:
#                 strength = "Weak"
#             else:
#                 strength = "Very weak"
#
#             sig = (
#                 "significant"
#                 if self.stars and self.stars != "ns"
#                 else "non-significant"
#             )
#
#             return (
#                 f"{strength} {direction} correlation "
#                 f"({self.statistic['name']}={r:.2f}, {self._format_p_publication()}, {sig})"
#             )
#
#         elif "test" in self.test_type.lower():
#             sig = (
#                 "Significant"
#                 if self.stars and self.stars != "ns"
#                 else "Non-significant"
#             )
#             return (
#                 f"{sig} difference "
#                 f"({self.statistic['name']}={self.statistic['value']:.2f}, "
#                 f"{self._format_p_publication()})"
#             )
#
#         return f"Test result: {self.format_text('publication')}"
#
#     def to_annotation_dict(self) -> Dict[str, Any]:
#         """
#         Convert to annotation dictionary for GUI integration.
#
#         This format matches the scitex-cloud GUI Annotation interface.
#
#         Returns
#         -------
#         dict
#             Annotation dictionary compatible with TypeScript Annotation interface
#
#         Examples
#         --------
#         >>> ann = result.to_annotation_dict()
#         >>> ann["type"]
#         'stat'
#         >>> ann["statResult"]["p_value"]
#         0.001
#         """
#         return {
#             "id": self.plot_id or f"stat_{id(self)}",
#             "type": "stat",
#             "label": f"{self.test_type} result",
#             "content": self.format_text(
#                 style=self.styling.symbol_style if self.styling else "compact"
#             ),
#             "position": (
#                 self.positioning.position.to_dict()
#                 if self.positioning and self.positioning.position
#                 else None
#             ),
#             "statResult": {
#                 "id": self.plot_id or f"stat_{id(self)}",
#                 "test_name": self.test_type,
#                 "p_value": self.p_value,
#                 "effect_size": (
#                     self.effect_size.get("value") if self.effect_size else None
#                 ),
#                 "group1": (
#                     self.samples.get("group1", {}).get("name")
#                     if self.samples
#                     else None
#                 ),
#                 "group2": (
#                     self.samples.get("group2", {}).get("name")
#                     if self.samples
#                     else None
#                 ),
#                 "statistic": self.statistic.get("value"),
#                 "method": self.test_type,
#                 "formatted_output": self.format_text(style="compact"),
#             },
#             "positioning": (
#                 self.positioning.to_dict() if self.positioning else None
#             ),
#             "styling": self.styling.to_dict() if self.styling else None,
#         }
#
#
# # =============================================================================
# # Convenience Functions
# # =============================================================================
#
#
# def create_stat_result(
#     test_type: str,
#     statistic_name: str,
#     statistic_value: float,
#     p_value: float,
#     **kwargs,
# ) -> StatResult:
#     """
#     Create a StatResult with minimal required fields.
#
#     Parameters
#     ----------
#     test_type : str
#         Type of statistical test
#     statistic_name : str
#         Name of the test statistic (e.g., "t", "r", "F")
#     statistic_value : float
#         Value of the test statistic
#     p_value : float
#         P-value from the test
#     **kwargs
#         Additional fields for StatResult
#
#     Returns
#     -------
#     StatResult
#         Configured StatResult instance
#
#     Examples
#     --------
#     >>> result = create_stat_result(
#     ...     test_type="pearson",
#     ...     statistic_name="r",
#     ...     statistic_value=0.85,
#     ...     p_value=0.001
#     ... )
#     """
#     # Import here to avoid circular dependency
#     from scitex.stats.utils import p2stars
#
#     # Determine category from test type
#     category_map = {
#         "pearson": "correlation",
#         "spearman": "correlation",
#         "kendall": "correlation",
#         "t-test": "parametric",
#         "anova": "parametric",
#         "mannwhitney": "non-parametric",
#         "kruskal": "non-parametric",
#     }
#
#     test_category = category_map.get(test_type.lower(), "other")
#
#     # Get stars
#     stars = p2stars(p_value, ns_symbol=False)
#
#     return StatResult(
#         test_type=test_type,
#         test_category=test_category,
#         statistic={"name": statistic_name, "value": statistic_value},
#         p_value=p_value,
#         stars=stars,
#         **kwargs,
#     )
#
#
# # =============================================================================
# # Public API
# # =============================================================================
#
# __all__ = [
#     # Schema version
#     "STATS_SCHEMA_VERSION",
#     # Type aliases
#     "PositionMode",
#     "UnitType",
#     "SymbolStyle",
#     # Position and styling
#     "Position",
#     "StatStyling",
#     "StatPositioning",
#     # Main result class
#     "StatResult",
#     # Convenience function
#     "create_stat_result",
# ]
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/schema/_stats.py
# --------------------------------------------------------------------------------
