#!/usr/bin/env python3
# File: ./tests/scitex/schema/test__validation.py
# Time-stamp: "2024-12-09 08:40:00 (ywatanabe)"
"""Tests for scitex.schema._validation module."""

import pytest

from scitex.schema import (
    ValidationError,
    validate_axes,
    validate_color,
    validate_figure,
    validate_plot,
    validate_stat_result,
)

# validate_canvas was removed from the module
try:
    from scitex.schema import validate_canvas
except ImportError:
    validate_canvas = None


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_basic_error(self):
        """Test basic error message."""
        err = ValidationError("Test error")
        assert err.message == "Test error"
        assert str(err) == "Test error"

    def test_error_with_field(self):
        """Test error with field context."""
        err = ValidationError("Invalid value", field="width_mm")
        assert err.field == "width_mm"
        assert "width_mm" in str(err)

    def test_error_with_field_and_value(self):
        """Test error with field and value context."""
        err = ValidationError("Must be positive", field="width_mm", value=-10)
        assert err.field == "width_mm"
        assert err.value == -10
        assert "width_mm" in str(err)
        assert "-10" in str(err)


class TestValidateFigure:
    """Tests for validate_figure function."""

    def test_valid_minimal_figure(self):
        """Test validation of minimal valid figure."""
        fig_data = {"width_mm": 180.0, "height_mm": 120.0}
        assert validate_figure(fig_data) is True

    def test_missing_width(self):
        """Test error on missing width_mm."""
        fig_data = {"height_mm": 120.0}
        with pytest.raises(ValidationError) as exc_info:
            validate_figure(fig_data)
        assert "width_mm" in str(exc_info.value)

    def test_missing_height(self):
        """Test error on missing height_mm."""
        fig_data = {"width_mm": 180.0}
        with pytest.raises(ValidationError) as exc_info:
            validate_figure(fig_data)
        assert "height_mm" in str(exc_info.value)

    def test_invalid_width_type(self):
        """Test error on invalid width type."""
        fig_data = {"width_mm": "180", "height_mm": 120.0}
        with pytest.raises(ValidationError) as exc_info:
            validate_figure(fig_data)
        assert "width_mm" in str(exc_info.value)

    def test_negative_width(self):
        """Test error on negative width."""
        fig_data = {"width_mm": -180.0, "height_mm": 120.0}
        with pytest.raises(ValidationError) as exc_info:
            validate_figure(fig_data)
        assert "positive" in str(exc_info.value).lower()

    def test_zero_height(self):
        """Test error on zero height."""
        fig_data = {"width_mm": 180.0, "height_mm": 0}
        with pytest.raises(ValidationError) as exc_info:
            validate_figure(fig_data)
        assert "positive" in str(exc_info.value).lower()

    def test_valid_with_nrows_ncols(self):
        """Test valid figure with nrows and ncols."""
        fig_data = {
            "width_mm": 180.0,
            "height_mm": 240.0,
            "nrows": 2,
            "ncols": 3,
        }
        assert validate_figure(fig_data) is True

    def test_invalid_nrows(self):
        """Test error on invalid nrows."""
        fig_data = {"width_mm": 180.0, "height_mm": 120.0, "nrows": 0}
        with pytest.raises(ValidationError) as exc_info:
            validate_figure(fig_data)
        assert "nrows" in str(exc_info.value)

    def test_invalid_ncols_type(self):
        """Test error on invalid ncols type."""
        fig_data = {"width_mm": 180.0, "height_mm": 120.0, "ncols": 2.5}
        with pytest.raises(ValidationError) as exc_info:
            validate_figure(fig_data)
        assert "ncols" in str(exc_info.value)

    def test_valid_with_dpi(self):
        """Test valid figure with dpi."""
        fig_data = {"width_mm": 180.0, "height_mm": 120.0, "dpi": 300}
        assert validate_figure(fig_data) is True

    def test_invalid_dpi(self):
        """Test error on invalid dpi."""
        fig_data = {"width_mm": 180.0, "height_mm": 120.0, "dpi": -100}
        with pytest.raises(ValidationError) as exc_info:
            validate_figure(fig_data)
        assert "dpi" in str(exc_info.value)

    def test_too_many_axes(self):
        """Test error when too many axes for layout."""
        fig_data = {
            "width_mm": 180.0,
            "height_mm": 120.0,
            "nrows": 1,
            "ncols": 2,
            "axes": [{}, {}, {}],  # 3 axes for 1x2 layout
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_figure(fig_data)
        assert "axes" in str(exc_info.value).lower()


class TestValidateAxes:
    """Tests for validate_axes function."""

    def test_empty_axes_valid(self):
        """Test empty axes dict is valid."""
        assert validate_axes({}) is True

    def test_valid_with_row_col(self):
        """Test valid axes with row and col."""
        axes_data = {"row": 0, "col": 1}
        assert validate_axes(axes_data) is True

    def test_invalid_row_negative(self):
        """Test error on negative row."""
        axes_data = {"row": -1}
        with pytest.raises(ValidationError) as exc_info:
            validate_axes(axes_data)
        assert "row" in str(exc_info.value)

    def test_invalid_col_type(self):
        """Test error on invalid col type."""
        axes_data = {"col": 1.5}
        with pytest.raises(ValidationError) as exc_info:
            validate_axes(axes_data)
        assert "col" in str(exc_info.value)

    def test_valid_scales(self):
        """Test valid axis scales."""
        for scale in ["linear", "log", "symlog", "logit"]:
            axes_data = {"xscale": scale, "yscale": scale}
            assert validate_axes(axes_data) is True

    def test_invalid_scale(self):
        """Test error on invalid scale."""
        axes_data = {"xscale": "invalid_scale"}
        with pytest.raises(ValidationError) as exc_info:
            validate_axes(axes_data)
        assert "xscale" in str(exc_info.value)

    def test_valid_limits(self):
        """Test valid axis limits."""
        axes_data = {"xlim": [0, 10], "ylim": (-5, 5)}
        assert validate_axes(axes_data) is True

    def test_invalid_limits_length(self):
        """Test error on invalid limits length."""
        axes_data = {"xlim": [0, 10, 20]}
        with pytest.raises(ValidationError) as exc_info:
            validate_axes(axes_data)
        assert "xlim" in str(exc_info.value)

    def test_invalid_limits_type(self):
        """Test error on invalid limits type."""
        axes_data = {"ylim": "0-10"}
        with pytest.raises(ValidationError) as exc_info:
            validate_axes(axes_data)
        assert "ylim" in str(exc_info.value)


class TestValidatePlot:
    """Tests for validate_plot function."""

    def test_missing_plot_type(self):
        """Test error on missing plot_type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_plot({})
        assert "plot_type" in str(exc_info.value)

    def test_line_plot_valid(self):
        """Test valid line plot."""
        plot_data = {
            "plot_type": "line",
            "data": {"x": [1, 2, 3], "y": [4, 5, 6]},
        }
        assert validate_plot(plot_data) is True

    def test_line_plot_missing_y(self):
        """Test error on line plot missing y."""
        plot_data = {
            "plot_type": "line",
            "data": {"x": [1, 2, 3]},
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_plot(plot_data)
        assert "y" in str(exc_info.value)

    def test_line_plot_mismatched_lengths(self):
        """Test error on mismatched x,y lengths."""
        plot_data = {
            "plot_type": "scatter",
            "data": {"x": [1, 2, 3], "y": [4, 5]},
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_plot(plot_data)
        assert "length" in str(exc_info.value).lower()

    def test_bar_plot_valid(self):
        """Test valid bar plot."""
        plot_data = {
            "plot_type": "bar",
            "data": {"x": ["a", "b", "c"], "height": [1, 2, 3]},
        }
        assert validate_plot(plot_data) is True

    def test_bar_plot_with_y(self):
        """Test bar plot accepts y as height."""
        plot_data = {
            "plot_type": "bar",
            "data": {"x": ["a", "b"], "y": [1, 2]},
        }
        assert validate_plot(plot_data) is True

    def test_hist_plot_valid(self):
        """Test valid histogram."""
        plot_data = {
            "plot_type": "hist",
            "data": {"x": [1, 2, 2, 3, 3, 3]},
        }
        assert validate_plot(plot_data) is True

    def test_heatmap_valid(self):
        """Test valid heatmap."""
        plot_data = {
            "plot_type": "heatmap",
            "data": {"z": [[1, 2], [3, 4]]},
        }
        assert validate_plot(plot_data) is True

    def test_heatmap_with_img(self):
        """Test heatmap with img data."""
        plot_data = {
            "plot_type": "imshow",
            "data": {"img": [[1, 2], [3, 4]]},
        }
        assert validate_plot(plot_data) is True

    def test_contour_valid(self):
        """Test valid contour plot."""
        plot_data = {
            "plot_type": "contour",
            "data": {
                "x": [0, 1],
                "y": [0, 1],
                "z": [[1, 2], [3, 4]],
            },
        }
        assert validate_plot(plot_data) is True

    def test_contour_missing_z(self):
        """Test error on contour missing z."""
        plot_data = {
            "plot_type": "contourf",
            "data": {"x": [0, 1], "y": [0, 1]},
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_plot(plot_data)
        assert "z" in str(exc_info.value)


class TestValidateStatResult:
    """Tests for validate_stat_result function."""

    def test_valid_stat_result(self):
        """Test valid statistical result."""
        stat_data = {
            "test_type": "t-test",
            "test_category": "parametric",
            "statistic": {"name": "t", "value": 2.5},
            "p_value": 0.01,
            "stars": "**",
        }
        assert validate_stat_result(stat_data) is True

    def test_missing_test_type(self):
        """Test error on missing test_type."""
        stat_data = {
            "test_category": "parametric",
            "statistic": {"name": "t", "value": 2.5},
            "p_value": 0.01,
            "stars": "**",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_stat_result(stat_data)
        assert "test_type" in str(exc_info.value)

    def test_invalid_p_value_range(self):
        """Test error on p_value out of range."""
        stat_data = {
            "test_type": "t-test",
            "test_category": "parametric",
            "statistic": {"name": "t", "value": 2.5},
            "p_value": 1.5,  # Invalid: > 1
            "stars": "ns",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_stat_result(stat_data)
        assert "p_value" in str(exc_info.value)

    def test_invalid_p_value_negative(self):
        """Test error on negative p_value."""
        stat_data = {
            "test_type": "t-test",
            "test_category": "parametric",
            "statistic": {"name": "t", "value": 2.5},
            "p_value": -0.05,
            "stars": "*",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_stat_result(stat_data)
        assert "p_value" in str(exc_info.value)

    def test_valid_stars_values(self):
        """Test all valid stars values."""
        base = {
            "test_type": "t-test",
            "test_category": "parametric",
            "statistic": {"name": "t", "value": 2.5},
            "p_value": 0.05,
        }
        for stars in ["***", "**", "*", "ns", ""]:
            stat_data = {**base, "stars": stars}
            assert validate_stat_result(stat_data) is True

    def test_invalid_stars(self):
        """Test error on invalid stars."""
        stat_data = {
            "test_type": "t-test",
            "test_category": "parametric",
            "statistic": {"name": "t", "value": 2.5},
            "p_value": 0.05,
            "stars": "****",  # Invalid
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_stat_result(stat_data)
        assert "stars" in str(exc_info.value)

    def test_valid_test_categories(self):
        """Test all valid test categories."""
        base = {
            "test_type": "test",
            "statistic": {"name": "stat", "value": 1.0},
            "p_value": 0.05,
            "stars": "*",
        }
        for category in ["parametric", "non-parametric", "correlation", "other"]:
            stat_data = {**base, "test_category": category}
            assert validate_stat_result(stat_data) is True

    def test_invalid_test_category(self):
        """Test error on invalid test_category."""
        stat_data = {
            "test_type": "t-test",
            "test_category": "unknown",
            "statistic": {"name": "t", "value": 2.5},
            "p_value": 0.05,
            "stars": "*",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_stat_result(stat_data)
        assert "test_category" in str(exc_info.value)

    def test_invalid_statistic_structure(self):
        """Test error on invalid statistic structure."""
        stat_data = {
            "test_type": "t-test",
            "test_category": "parametric",
            "statistic": 2.5,  # Should be dict
            "p_value": 0.05,
            "stars": "*",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_stat_result(stat_data)
        assert "statistic" in str(exc_info.value)

    def test_statistic_missing_name(self):
        """Test error on statistic missing name."""
        stat_data = {
            "test_type": "t-test",
            "test_category": "parametric",
            "statistic": {"value": 2.5},
            "p_value": 0.05,
            "stars": "*",
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_stat_result(stat_data)
        assert "name" in str(exc_info.value).lower()

    def test_valid_with_effect_size(self):
        """Test valid stat result with effect size."""
        stat_data = {
            "test_type": "t-test",
            "test_category": "parametric",
            "statistic": {"name": "t", "value": 2.5},
            "p_value": 0.01,
            "stars": "**",
            "effect_size": {"name": "Cohen's d", "value": 0.8},
        }
        assert validate_stat_result(stat_data) is True

    def test_invalid_effect_size_structure(self):
        """Test error on invalid effect_size structure."""
        stat_data = {
            "test_type": "t-test",
            "test_category": "parametric",
            "statistic": {"name": "t", "value": 2.5},
            "p_value": 0.01,
            "stars": "**",
            "effect_size": {"name": "d"},  # Missing value
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_stat_result(stat_data)
        assert "effect_size" in str(exc_info.value)


@pytest.mark.skipif(validate_canvas is None, reason="validate_canvas not available")
class TestValidateCanvas:
    """Tests for validate_canvas function."""

    def test_valid_minimal_canvas(self):
        """Test valid minimal canvas."""
        canvas_data = {"canvas_name": "fig1"}
        assert validate_canvas(canvas_data) is True

    def test_missing_canvas_name(self):
        """Test error on missing canvas_name."""
        with pytest.raises(ValidationError) as exc_info:
            validate_canvas({})
        assert "canvas_name" in str(exc_info.value)

    def test_valid_with_size(self):
        """Test valid canvas with size."""
        canvas_data = {
            "canvas_name": "fig1",
            "size": {"width_mm": 180, "height_mm": 240},
        }
        assert validate_canvas(canvas_data) is True

    def test_invalid_size_width(self):
        """Test error on invalid size width."""
        canvas_data = {
            "canvas_name": "fig1",
            "size": {"width_mm": 0, "height_mm": 240},
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_canvas(canvas_data)
        assert "width_mm" in str(exc_info.value)

    def test_valid_with_panels(self):
        """Test valid canvas with panels."""
        canvas_data = {
            "canvas_name": "fig1",
            "panels": [
                {"name": "a", "type": "scitex"},
                {"name": "b", "type": "image"},
            ],
        }
        assert validate_canvas(canvas_data) is True

    def test_panel_missing_name(self):
        """Test error on panel missing name."""
        canvas_data = {
            "canvas_name": "fig1",
            "panels": [{"type": "scitex"}],
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_canvas(canvas_data)
        assert "name" in str(exc_info.value).lower()

    def test_duplicate_panel_names(self):
        """Test error on duplicate panel names."""
        canvas_data = {
            "canvas_name": "fig1",
            "panels": [
                {"name": "a", "type": "scitex"},
                {"name": "a", "type": "image"},  # Duplicate
            ],
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_canvas(canvas_data)
        assert "duplicate" in str(exc_info.value).lower()

    def test_invalid_panel_type(self):
        """Test error on invalid panel type."""
        canvas_data = {
            "canvas_name": "fig1",
            "panels": [{"name": "a", "type": "unknown"}],
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_canvas(canvas_data)
        assert "type" in str(exc_info.value).lower()


class TestValidateColor:
    """Tests for validate_color function."""

    def test_valid_color_name(self):
        """Test valid color name."""
        assert validate_color("red") is True
        assert validate_color("blue") is True

    def test_valid_hex_color(self):
        """Test valid hex color."""
        assert validate_color("#ff0000") is True
        assert validate_color("#FFFFFF") is True

    def test_none_color(self):
        """Test error on None color."""
        with pytest.raises(ValidationError):
            validate_color(None)

    def test_empty_color(self):
        """Test error on empty color."""
        with pytest.raises(ValidationError):
            validate_color("")

    def test_non_string_color(self):
        """Test error on non-string color."""
        with pytest.raises(ValidationError):
            validate_color([255, 0, 0])

        with pytest.raises(ValidationError):
            validate_color(0xFF0000)


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/schema/_validation.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: ./src/scitex/schema/_validation.py
# # Time-stamp: "2024-12-09 08:30:00 (ywatanabe)"
# """
# Unified Validation Layer for SciTeX Schemas.
#
# Provides validation functions for all cross-module data structures:
# - Figure specifications
# - Statistical results
# - Canvas specifications
#
# All validation functions follow the pattern:
# - Return True if valid
# - Raise ValidationError with detailed message if invalid
# """
#
# from typing import Dict, Any, List, Optional, Union
#
#
# class ValidationError(Exception):
#     """
#     Exception raised when schema validation fails.
#
#     Attributes
#     ----------
#     message : str
#         Explanation of the validation error
#     field : str, optional
#         The specific field that failed validation
#     value : Any, optional
#         The invalid value
#     """
#
#     def __init__(
#         self,
#         message: str,
#         field: Optional[str] = None,
#         value: Any = None,
#     ):
#         self.message = message
#         self.field = field
#         self.value = value
#         super().__init__(self._format_message())
#
#     def _format_message(self) -> str:
#         if self.field and self.value is not None:
#             return f"{self.message} (field='{self.field}', value={self.value!r})"
#         elif self.field:
#             return f"{self.message} (field='{self.field}')"
#         return self.message
#
#
# # =============================================================================
# # Figure Validation
# # =============================================================================
#
#
# def validate_figure(fig_data: Dict[str, Any]) -> bool:
#     """
#     Validate a figure specification.
#
#     Parameters
#     ----------
#     fig_data : Dict[str, Any]
#         Figure specification dictionary
#
#     Returns
#     -------
#     bool
#         True if valid
#
#     Raises
#     ------
#     ValidationError
#         If validation fails
#     """
#     # Required fields
#     required_fields = ["width_mm", "height_mm"]
#
#     for field in required_fields:
#         if field not in fig_data:
#             raise ValidationError(f"Missing required field: {field}", field=field)
#
#     # Type validation
#     if not isinstance(fig_data["width_mm"], (int, float)):
#         raise ValidationError(
#             "width_mm must be a number",
#             field="width_mm",
#             value=fig_data["width_mm"],
#         )
#
#     if not isinstance(fig_data["height_mm"], (int, float)):
#         raise ValidationError(
#             "height_mm must be a number",
#             field="height_mm",
#             value=fig_data["height_mm"],
#         )
#
#     # Value validation
#     if fig_data["width_mm"] <= 0:
#         raise ValidationError(
#             "width_mm must be positive",
#             field="width_mm",
#             value=fig_data["width_mm"],
#         )
#
#     if fig_data["height_mm"] <= 0:
#         raise ValidationError(
#             "height_mm must be positive",
#             field="height_mm",
#             value=fig_data["height_mm"],
#         )
#
#     # Optional fields validation
#     if "nrows" in fig_data:
#         if not isinstance(fig_data["nrows"], int) or fig_data["nrows"] <= 0:
#             raise ValidationError(
#                 "nrows must be a positive integer",
#                 field="nrows",
#                 value=fig_data["nrows"],
#             )
#
#     if "ncols" in fig_data:
#         if not isinstance(fig_data["ncols"], int) or fig_data["ncols"] <= 0:
#             raise ValidationError(
#                 "ncols must be a positive integer",
#                 field="ncols",
#                 value=fig_data["ncols"],
#             )
#
#     if "dpi" in fig_data:
#         if not isinstance(fig_data["dpi"], int) or fig_data["dpi"] <= 0:
#             raise ValidationError(
#                 "dpi must be a positive integer",
#                 field="dpi",
#                 value=fig_data["dpi"],
#             )
#
#     if "axes" in fig_data:
#         if not isinstance(fig_data["axes"], list):
#             raise ValidationError(
#                 "axes must be a list",
#                 field="axes",
#                 value=type(fig_data["axes"]).__name__,
#             )
#
#         # Validate axes layout
#         nrows = fig_data.get("nrows", 1)
#         ncols = fig_data.get("ncols", 1)
#         num_axes = len(fig_data["axes"])
#         max_axes = nrows * ncols
#
#         if num_axes > max_axes:
#             raise ValidationError(
#                 f"Too many axes: {num_axes} axes for {nrows}x{ncols} layout (max {max_axes})",
#                 field="axes",
#             )
#
#         # Validate each axes entry
#         for i, axes_data in enumerate(fig_data["axes"]):
#             try:
#                 validate_axes(axes_data)
#             except ValidationError as e:
#                 raise ValidationError(
#                     f"Invalid axes at index {i}: {e.message}",
#                     field=f"axes[{i}]",
#                 )
#
#     return True
#
#
# def validate_axes(axes_data: Dict[str, Any]) -> bool:
#     """
#     Validate an axes specification.
#
#     Parameters
#     ----------
#     axes_data : Dict[str, Any]
#         Axes specification dictionary
#
#     Returns
#     -------
#     bool
#         True if valid
#
#     Raises
#     ------
#     ValidationError
#         If validation fails
#     """
#     # Optional row/col must be non-negative integers
#     if "row" in axes_data:
#         if not isinstance(axes_data["row"], int) or axes_data["row"] < 0:
#             raise ValidationError(
#                 "row must be a non-negative integer",
#                 field="row",
#                 value=axes_data["row"],
#             )
#
#     if "col" in axes_data:
#         if not isinstance(axes_data["col"], int) or axes_data["col"] < 0:
#             raise ValidationError(
#                 "col must be a non-negative integer",
#                 field="col",
#                 value=axes_data["col"],
#             )
#
#     # Validate scale values
#     valid_scales = {"linear", "log", "symlog", "logit"}
#
#     if "xscale" in axes_data:
#         if axes_data["xscale"] not in valid_scales:
#             raise ValidationError(
#                 f"xscale must be one of {valid_scales}",
#                 field="xscale",
#                 value=axes_data["xscale"],
#             )
#
#     if "yscale" in axes_data:
#         if axes_data["yscale"] not in valid_scales:
#             raise ValidationError(
#                 f"yscale must be one of {valid_scales}",
#                 field="yscale",
#                 value=axes_data["yscale"],
#             )
#
#     # Validate limits
#     for lim_field in ["xlim", "ylim"]:
#         if lim_field in axes_data:
#             lim = axes_data[lim_field]
#             if not isinstance(lim, (list, tuple)) or len(lim) != 2:
#                 raise ValidationError(
#                     f"{lim_field} must be a list/tuple of [min, max]",
#                     field=lim_field,
#                     value=lim,
#                 )
#
#     # Validate plots
#     if "plots" in axes_data:
#         if not isinstance(axes_data["plots"], list):
#             raise ValidationError(
#                 "plots must be a list",
#                 field="plots",
#                 value=type(axes_data["plots"]).__name__,
#             )
#
#         for i, plot_data in enumerate(axes_data["plots"]):
#             try:
#                 validate_plot(plot_data)
#             except ValidationError as e:
#                 raise ValidationError(
#                     f"Invalid plot at index {i}: {e.message}",
#                     field=f"plots[{i}]",
#                 )
#
#     return True
#
#
# def validate_plot(plot_data: Dict[str, Any]) -> bool:
#     """
#     Validate a plot specification.
#
#     Parameters
#     ----------
#     plot_data : Dict[str, Any]
#         Plot specification dictionary
#
#     Returns
#     -------
#     bool
#         True if valid
#
#     Raises
#     ------
#     ValidationError
#         If validation fails
#     """
#     # plot_type is required
#     if "plot_type" not in plot_data:
#         raise ValidationError("Plot must specify plot_type", field="plot_type")
#
#     plot_type = plot_data["plot_type"]
#     data = plot_data.get("data", {})
#
#     # Type-specific requirements
#     if plot_type in ["line", "scatter", "errorbar"]:
#         if "x" not in data or "y" not in data:
#             raise ValidationError(
#                 f"{plot_type} requires 'x' and 'y' data",
#                 field="data",
#             )
#
#         # Validate arrays have same length
#         x_len = len(data["x"]) if hasattr(data["x"], "__len__") else 1
#         y_len = len(data["y"]) if hasattr(data["y"], "__len__") else 1
#
#         if x_len != y_len:
#             raise ValidationError(
#                 f"x and y data must have same length: {x_len} != {y_len}",
#                 field="data",
#             )
#
#     elif plot_type in ["bar", "barh"]:
#         if "x" not in data:
#             raise ValidationError(
#                 f"{plot_type} requires 'x' data",
#                 field="data.x",
#             )
#
#         if "height" not in data and "y" not in data:
#             raise ValidationError(
#                 f"{plot_type} requires 'height' or 'y' data",
#                 field="data",
#             )
#
#     elif plot_type == "hist":
#         if "x" not in data:
#             raise ValidationError(
#                 "hist requires 'x' data",
#                 field="data.x",
#             )
#
#     elif plot_type in ["heatmap", "imshow"]:
#         if "z" not in data and "img" not in data:
#             raise ValidationError(
#                 f"{plot_type} requires 'z' or 'img' data",
#                 field="data",
#             )
#
#     elif plot_type in ["contour", "contourf"]:
#         if "x" not in data or "y" not in data or "z" not in data:
#             raise ValidationError(
#                 f"{plot_type} requires 'x', 'y', and 'z' data",
#                 field="data",
#             )
#
#     return True
#
#
# # =============================================================================
# # Statistical Result Validation
# # =============================================================================
#
#
# def validate_stat_result(stat_data: Dict[str, Any]) -> bool:
#     """
#     Validate a statistical result specification.
#
#     Parameters
#     ----------
#     stat_data : Dict[str, Any]
#         Statistical result dictionary
#
#     Returns
#     -------
#     bool
#         True if valid
#
#     Raises
#     ------
#     ValidationError
#         If validation fails
#     """
#     # Required fields
#     required_fields = ["test_type", "test_category", "statistic", "p_value", "stars"]
#
#     for field in required_fields:
#         if field not in stat_data:
#             raise ValidationError(f"Missing required field: {field}", field=field)
#
#     # Validate statistic structure
#     statistic = stat_data["statistic"]
#     if not isinstance(statistic, dict):
#         raise ValidationError(
#             "statistic must be a dictionary",
#             field="statistic",
#             value=type(statistic).__name__,
#         )
#
#     if "name" not in statistic or "value" not in statistic:
#         raise ValidationError(
#             "statistic must have 'name' and 'value' keys",
#             field="statistic",
#         )
#
#     # Validate p_value
#     p_value = stat_data["p_value"]
#     if not isinstance(p_value, (int, float)):
#         raise ValidationError(
#             "p_value must be a number",
#             field="p_value",
#             value=p_value,
#         )
#
#     if p_value < 0 or p_value > 1:
#         raise ValidationError(
#             "p_value must be between 0 and 1",
#             field="p_value",
#             value=p_value,
#         )
#
#     # Validate stars
#     valid_stars = {"***", "**", "*", "ns", ""}
#     stars = stat_data["stars"]
#     if stars not in valid_stars:
#         raise ValidationError(
#             f"stars must be one of {valid_stars}",
#             field="stars",
#             value=stars,
#         )
#
#     # Validate test_category
#     valid_categories = {"parametric", "non-parametric", "correlation", "other"}
#     category = stat_data["test_category"]
#     if category not in valid_categories:
#         raise ValidationError(
#             f"test_category must be one of {valid_categories}",
#             field="test_category",
#             value=category,
#         )
#
#     # Validate effect_size structure if present
#     if "effect_size" in stat_data and stat_data["effect_size"]:
#         effect_size = stat_data["effect_size"]
#         if not isinstance(effect_size, dict):
#             raise ValidationError(
#                 "effect_size must be a dictionary",
#                 field="effect_size",
#                 value=type(effect_size).__name__,
#             )
#
#         if "name" not in effect_size or "value" not in effect_size:
#             raise ValidationError(
#                 "effect_size must have 'name' and 'value' keys",
#                 field="effect_size",
#             )
#
#     return True
#
#
# # =============================================================================
# # Color Validation
# # =============================================================================
#
#
# def validate_color(color: Any) -> bool:
#     """
#     Validate a color specification.
#
#     Parameters
#     ----------
#     color : Any
#         Color specification (name, hex, rgb, etc.)
#
#     Returns
#     -------
#     bool
#         True if valid
#
#     Raises
#     ------
#     ValidationError
#         If validation fails
#     """
#     if color is None:
#         raise ValidationError("Color cannot be None", field="color", value=color)
#
#     if not isinstance(color, str):
#         raise ValidationError(
#             f"Color must be a string, got {type(color).__name__}",
#             field="color",
#             value=color,
#         )
#
#     if not color:
#         raise ValidationError("Color cannot be empty string", field="color", value=color)
#
#     return True
#
#
# __all__ = [
#     "ValidationError",
#     "validate_figure",
#     "validate_axes",
#     "validate_plot",
#     "validate_stat_result",
#     "validate_color",
# ]
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/schema/_validation.py
# --------------------------------------------------------------------------------
