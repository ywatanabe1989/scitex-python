#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-11-29"

"""Tests for scitex.stats._schema module."""

import json
import pytest
import numpy as np
from scitex.stats._schema import (
    StatResult,
    Position,
    StatStyling,
    StatPositioning,
    create_stat_result,
)


class TestPosition:
    """Test Position class for coordinate handling."""

    def test_basic_creation(self):
        """Test basic Position creation."""
        pos = Position(x=10.0, y=20.0, unit="mm")
        assert pos.x == 10.0
        assert pos.y == 20.0
        assert pos.unit == "mm"

    def test_with_relative_positioning(self):
        """Test Position with relative coordinates."""
        pos = Position(
            x=5.0,
            y=10.0,
            unit="mm",
            relative_to="plot_0",
            offset={"dx": 2.0, "dy": -1.0}
        )
        assert pos.relative_to == "plot_0"
        assert pos.offset["dx"] == 2.0

    def test_to_dict(self):
        """Test Position to_dict conversion."""
        pos = Position(x=10.0, y=20.0, unit="mm")
        data = pos.to_dict()

        assert data["x"] == 10.0
        assert data["y"] == 20.0
        assert data["unit"] == "mm"

    def test_from_dict(self):
        """Test Position from_dict creation."""
        data = {"x": 15.0, "y": 25.0, "unit": "px"}
        pos = Position.from_dict(data)

        assert pos.x == 15.0
        assert pos.y == 25.0
        assert pos.unit == "px"

    def test_mm_to_px_conversion(self):
        """Test converting mm to pixels at 300 DPI."""
        pos = Position(x=10.0, y=20.0, unit="mm")
        pos_px = pos.to_px(dpi=300.0)

        # At 300 DPI: 1mm = 300/25.4 ≈ 11.811 px
        expected_px = 10.0 * (300.0 / 25.4)
        assert pos_px.unit == "px"
        assert abs(pos_px.x - expected_px) < 0.01

    def test_px_to_mm_conversion(self):
        """Test converting pixels to mm at 300 DPI."""
        pos = Position(x=118.11, y=236.22, unit="px")
        pos_mm = pos.to_mm(dpi=300.0)

        # At 300 DPI: 118.11 px ≈ 10mm
        assert pos_mm.unit == "mm"
        assert abs(pos_mm.x - 10.0) < 0.01

    def test_inch_to_mm_conversion(self):
        """Test converting inches to mm."""
        pos = Position(x=1.0, y=2.0, unit="inch")
        pos_mm = pos.to_mm()

        # 1 inch = 25.4 mm
        assert pos_mm.unit == "mm"
        assert abs(pos_mm.x - 25.4) < 0.01
        assert abs(pos_mm.y - 50.8) < 0.01


class TestStatStyling:
    """Test StatStyling class for display styling."""

    def test_default_styling(self):
        """Test default StatStyling values."""
        style = StatStyling()

        assert style.font_size_pt == 7.0
        assert style.font_family == "Arial"
        assert style.color == "#000000"
        assert style.symbol_style == "asterisk"

    def test_custom_styling(self):
        """Test custom StatStyling values."""
        style = StatStyling(
            font_size_pt=8.0,
            font_family="Times",
            color="#FF0000",
            symbol_style="bracket",
            line_width_mm=0.3
        )

        assert style.font_size_pt == 8.0
        assert style.font_family == "Times"
        assert style.symbol_style == "bracket"
        assert style.line_width_mm == 0.3

    def test_theme_color_auto_light(self):
        """Test automatic theme color for light mode."""
        style = StatStyling(theme="auto")
        color = style.get_theme_color(is_dark=False)

        assert color == "#000000"

    def test_theme_color_auto_dark(self):
        """Test automatic theme color for dark mode."""
        style = StatStyling(theme="auto")
        color = style.get_theme_color(is_dark=True)

        assert color == "#ffffff"

    def test_theme_color_explicit_dark(self):
        """Test explicit dark theme color."""
        style = StatStyling(theme="dark")
        color = style.get_theme_color(is_dark=False)  # Still returns dark color

        assert color == "#ffffff"

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        style = StatStyling(font_size_pt=8.0, color="#FF0000")
        data = style.to_dict()
        style2 = StatStyling.from_dict(data)

        assert style2.font_size_pt == 8.0
        assert style2.color == "#FF0000"


class TestStatPositioning:
    """Test StatPositioning class for GUI-ready positioning."""

    def test_default_positioning(self):
        """Test default StatPositioning values."""
        pos = StatPositioning()

        assert pos.mode == "auto"
        assert pos.avoid_overlap is True
        assert pos.min_distance_mm == 2.0

    def test_with_position(self):
        """Test StatPositioning with explicit position."""
        position = Position(x=10.0, y=20.0, unit="mm")
        pos = StatPositioning(
            mode="absolute",
            position=position,
            preferred_corner="top-right"
        )

        assert pos.mode == "absolute"
        assert pos.position.x == 10.0
        assert pos.preferred_corner == "top-right"

    def test_to_dict_with_position(self):
        """Test StatPositioning to_dict with nested Position."""
        position = Position(x=5.0, y=10.0, unit="mm")
        pos = StatPositioning(mode="relative_to_plot", position=position)
        data = pos.to_dict()

        assert data["mode"] == "relative_to_plot"
        assert data["position"]["x"] == 5.0
        assert data["position"]["unit"] == "mm"

    def test_from_dict_with_position(self):
        """Test StatPositioning from_dict with nested Position."""
        data = {
            "mode": "above_whisker",
            "position": {"x": 15.0, "y": 25.0, "unit": "px"},
            "avoid_overlap": True,
            "min_distance_mm": 2.0,
            "preferred_corner": None,
            "anchor_to": None
        }
        pos = StatPositioning.from_dict(data)

        assert pos.mode == "above_whisker"
        assert pos.position.x == 15.0
        assert pos.position.unit == "px"


class TestStatResult:
    """Test StatResult main class."""

    def test_basic_creation(self):
        """Test basic StatResult creation."""
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.001,
            stars="***"
        )

        assert result.test_type == "pearson"
        assert result.statistic["value"] == 0.85
        assert result.p_value == 0.001
        assert result.stars == "***"

    def test_post_init_defaults(self):
        """Test that __post_init__ sets defaults."""
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.45},
            p_value=0.002,
            stars="**"
        )

        # Should auto-create created_at
        assert result.created_at is not None

        # Should auto-create default styling
        assert result.styling is not None
        assert result.styling.font_size_pt == 7.0

        # Should auto-create default positioning
        assert result.positioning is not None
        assert result.positioning.mode == "auto"

    def test_with_effect_size(self):
        """Test StatResult with effect size."""
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.45},
            p_value=0.002,
            stars="**",
            effect_size={
                "name": "cohens_d",
                "value": 0.85,
                "interpretation": "large",
                "ci_95": [0.42, 1.28]
            }
        )

        assert result.effect_size["name"] == "cohens_d"
        assert result.effect_size["value"] == 0.85
        assert result.effect_size["interpretation"] == "large"

    def test_with_samples(self):
        """Test StatResult with sample information."""
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 2.34},
            p_value=0.023,
            stars="*",
            samples={
                "group1": {"name": "Control", "n": 30, "mean": 5.2, "std": 1.1},
                "group2": {"name": "Treatment", "n": 32, "mean": 6.8, "std": 1.3}
            }
        )

        assert result.samples["group1"]["n"] == 30
        assert result.samples["group2"]["mean"] == 6.8

    def test_to_dict_basic(self):
        """Test StatResult to_dict conversion."""
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.75},
            p_value=0.01,
            stars="**"
        )

        data = result.to_dict()

        assert data["test_type"] == "pearson"
        assert data["statistic"]["value"] == 0.75
        assert data["p_value"] == 0.01

    def test_to_dict_with_numpy(self):
        """Test that numpy types are converted correctly."""
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": np.float64(3.45)},
            p_value=np.float64(0.002),
            stars="**",
            ci_95=[np.float64(0.5), np.float64(1.2)]
        )

        data = result.to_dict()

        # Should be native Python types
        assert isinstance(data["statistic"]["value"], float)
        assert isinstance(data["p_value"], float)
        assert isinstance(data["ci_95"][0], float)

    def test_to_json(self):
        """Test StatResult to_json conversion."""
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.001,
            stars="***"
        )

        json_str = result.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["test_type"] == "pearson"
        assert data["stars"] == "***"

    def test_from_dict(self):
        """Test StatResult from_dict creation."""
        data = {
            "test_type": "spearman",
            "test_category": "correlation",
            "statistic": {"name": "rho", "value": 0.72},
            "p_value": 0.005,
            "stars": "**",
            "created_at": "2025-01-01T00:00:00",
            "software_version": "1.0.0",
            "plot_id": "plot_0",
            "effect_size": None,
            "correction": None,
            "samples": None,
            "assumptions": None,
            "ci_95": None,
            "positioning": None,
            "styling": None,
            "extra": None
        }

        result = StatResult.from_dict(data)

        assert result.test_type == "spearman"
        assert result.statistic["value"] == 0.72

    def test_from_json(self):
        """Test StatResult from_json creation."""
        json_str = '''
        {
            "test_type": "kendall",
            "test_category": "correlation",
            "statistic": {"name": "tau", "value": 0.65},
            "p_value": 0.01,
            "stars": "**",
            "created_at": "2025-01-01T00:00:00",
            "software_version": null,
            "plot_id": null,
            "effect_size": null,
            "correction": null,
            "samples": null,
            "assumptions": null,
            "ci_95": null,
            "positioning": null,
            "styling": null,
            "extra": null
        }
        '''

        result = StatResult.from_json(json_str)

        assert result.test_type == "kendall"
        assert result.p_value == 0.01

    def test_format_text_compact(self):
        """Test format_text with compact style."""
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.850},
            p_value=0.001,
            stars="***"
        )

        text = result.format_text(style="compact")

        assert text == "r = 0.850***"

    def test_format_text_asterisk(self):
        """Test format_text with asterisk style."""
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.850},
            p_value=0.001,
            stars="***"
        )

        text = result.format_text(style="asterisk")

        assert text == "***"

    def test_format_text_asterisk_ns(self):
        """Test format_text with asterisk style for non-significant."""
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.15},
            p_value=0.15,
            stars="ns"
        )

        text = result.format_text(style="asterisk")

        assert text == "ns"

    def test_format_text_text_style(self):
        """Test format_text with text style."""
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.850},
            p_value=0.023,
            stars="*"
        )

        text = result.format_text(style="text")

        assert text == "p = 0.023"

    def test_format_text_detailed(self):
        """Test format_text with detailed style."""
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.456},
            p_value=0.002,
            stars="**",
            effect_size={"name": "d", "value": 0.85}
        )

        text = result.format_text(style="detailed")

        assert "t = 3.456" in text
        assert "p = 2.000e-03" in text
        assert "d = 0.85" in text

    def test_format_text_publication(self):
        """Test format_text with publication style."""
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.0005,
            stars="***"
        )

        text = result.format_text(style="publication")

        assert text == "(r = 0.85, p < 0.001)"

    def test_format_p_publication_very_small(self):
        """Test p-value formatting for very small values."""
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 5.0},
            p_value=0.0001,
            stars="***"
        )

        p_text = result._format_p_publication()

        assert p_text == "p < 0.001"

    def test_format_p_publication_small(self):
        """Test p-value formatting for small values."""
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.0},
            p_value=0.005,
            stars="**"
        )

        p_text = result._format_p_publication()

        assert p_text == "p < 0.01"

    def test_format_p_publication_borderline(self):
        """Test p-value formatting for borderline values."""
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 2.0},
            p_value=0.045,
            stars="*"
        )

        p_text = result._format_p_publication()

        assert p_text == "p < 0.05"

    def test_format_p_publication_ns(self):
        """Test p-value formatting for non-significant values."""
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 1.0},
            p_value=0.15,
            stars="ns"
        )

        p_text = result._format_p_publication()

        assert p_text == "p = 0.150"

    def test_get_interpretation_correlation_strong_positive(self):
        """Test interpretation for strong positive correlation."""
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.001,
            stars="***"
        )

        interp = result.get_interpretation()

        assert "Strong positive correlation" in interp
        assert "significant" in interp
        assert "r=0.85" in interp

    def test_get_interpretation_correlation_weak_negative(self):
        """Test interpretation for weak negative correlation."""
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": -0.35},
            p_value=0.045,
            stars="*"
        )

        interp = result.get_interpretation()

        assert "Weak negative correlation" in interp

    def test_get_interpretation_correlation_ns(self):
        """Test interpretation for non-significant correlation."""
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.15},
            p_value=0.25,
            stars="ns"
        )

        interp = result.get_interpretation()

        assert "Very weak positive correlation" in interp
        assert "non-significant" in interp

    def test_get_interpretation_ttest(self):
        """Test interpretation for t-test."""
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.45},
            p_value=0.002,
            stars="**"
        )

        interp = result.get_interpretation()

        assert "Significant difference" in interp
        assert "t=3.45" in interp

    def test_to_annotation_dict(self):
        """Test conversion to annotation dictionary for GUI."""
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.001,
            stars="***",
            plot_id="plot_0",
            samples={
                "group1": {"name": "Control", "n": 30},
                "group2": {"name": "Treatment", "n": 32}
            }
        )

        ann = result.to_annotation_dict()

        assert ann["type"] == "stat"
        assert ann["id"] == "plot_0"
        assert ann["statResult"]["test_name"] == "pearson"
        assert ann["statResult"]["p_value"] == 0.001
        assert ann["statResult"]["formatted_output"] == "r = 0.850***"
        assert ann["statResult"]["group1"] == "Control"
        assert ann["statResult"]["group2"] == "Treatment"

    def test_to_annotation_dict_with_position(self):
        """Test annotation dict with explicit position."""
        position = Position(x=10.0, y=20.0, unit="mm")
        positioning = StatPositioning(mode="absolute", position=position)

        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.0},
            p_value=0.01,
            stars="**",
            positioning=positioning
        )

        ann = result.to_annotation_dict()

        assert ann["position"]["x"] == 10.0
        assert ann["position"]["unit"] == "mm"

    def test_serialization_round_trip(self):
        """Test full serialization round-trip."""
        original = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.001,
            stars="***",
            effect_size={"name": "r_squared", "value": 0.7225},
            ci_95=[0.65, 0.95]
        )

        # To JSON and back
        json_str = original.to_json()
        restored = StatResult.from_json(json_str)

        assert restored.test_type == original.test_type
        assert restored.statistic["value"] == original.statistic["value"]
        assert restored.p_value == original.p_value
        assert restored.effect_size["value"] == original.effect_size["value"]


class TestCreateStatResult:
    """Test create_stat_result convenience function."""

    def test_basic_creation(self):
        """Test basic creation with minimal fields."""
        result = create_stat_result(
            test_type="pearson",
            statistic_name="r",
            statistic_value=0.85,
            p_value=0.001
        )

        assert result.test_type == "pearson"
        assert result.test_category == "correlation"
        assert result.statistic["name"] == "r"
        assert result.statistic["value"] == 0.85
        assert result.stars == "***"

    def test_with_kwargs(self):
        """Test creation with additional kwargs."""
        result = create_stat_result(
            test_type="t-test",
            statistic_name="t",
            statistic_value=3.45,
            p_value=0.002,
            effect_size={"name": "cohens_d", "value": 0.85}
        )

        assert result.test_type == "t-test"
        assert result.test_category == "parametric"
        assert result.effect_size["value"] == 0.85

    def test_category_mapping_correlation(self):
        """Test automatic category mapping for correlation tests."""
        result = create_stat_result(
            test_type="spearman",
            statistic_name="rho",
            statistic_value=0.72,
            p_value=0.01
        )

        assert result.test_category == "correlation"

    def test_category_mapping_parametric(self):
        """Test automatic category mapping for parametric tests."""
        result = create_stat_result(
            test_type="anova",
            statistic_name="F",
            statistic_value=5.23,
            p_value=0.01
        )

        assert result.test_category == "parametric"

    def test_category_mapping_nonparametric(self):
        """Test automatic category mapping for non-parametric tests."""
        result = create_stat_result(
            test_type="mannwhitney",
            statistic_name="U",
            statistic_value=150.0,
            p_value=0.03
        )

        assert result.test_category == "non-parametric"

    def test_category_mapping_unknown(self):
        """Test category mapping for unknown test type."""
        result = create_stat_result(
            test_type="custom_test",
            statistic_name="X",
            statistic_value=1.0,
            p_value=0.05
        )

        assert result.test_category == "other"


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/_schema.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: ./src/scitex/stats/_schema.py
# # Timestamp: 2025-12-20
# """
# Statistical Result Schema - DEPRECATED
# 
# This module is deprecated. Import from scitex.io.bundle._stats instead:
#     from scitex.io.bundle._stats import Position, StatStyling, StatPositioning
# """
# 
# import warnings
# 
# warnings.warn(
#     "scitex.stats._schema is deprecated. Import from scitex.io.bundle._stats instead.",
#     DeprecationWarning,
#     stacklevel=2
# )
# 
# # Re-export from FTS (new single source of truth)
# from scitex.io.bundle._stats import (
#     # Type aliases
#     PositionMode,
#     UnitType,
#     SymbolStyle,
#     # Position and styling
#     Position,
#     StatStyling,
#     StatPositioning,
# )
# 
# # StatResult is no longer a dataclass - use dicts for test results
# StatResult = dict
# 
# def create_stat_result(
#     test_type: str,
#     statistic_name: str,
#     statistic_value: float,
#     p_value: float,
#     **kwargs,
# ) -> dict:
#     """Create a stat result dict (deprecated, use simple dicts instead)."""
#     from scitex.stats.utils import p2stars
# 
#     return {
#         "test_type": test_type,
#         "test_category": kwargs.get("test_category", "other"),
#         "statistic": {"name": statistic_name, "value": statistic_value},
#         "p_value": p_value,
#         "stars": p2stars(p_value, ns_symbol=False),
#         **{k: v for k, v in kwargs.items() if k != "test_category"},
#     }
# 
# __all__ = [
#     # Type aliases
#     "PositionMode",
#     "UnitType",
#     "SymbolStyle",
#     # Position and styling
#     "Position",
#     "StatStyling",
#     "StatPositioning",
#     # Deprecated
#     "StatResult",
#     "create_stat_result",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/_schema.py
# --------------------------------------------------------------------------------
