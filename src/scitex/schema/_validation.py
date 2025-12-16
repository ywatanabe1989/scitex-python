#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/schema/_validation.py
# Time-stamp: "2024-12-09 08:30:00 (ywatanabe)"
"""
Unified Validation Layer for SciTeX Schemas.

Provides validation functions for all cross-module data structures:
- Figure specifications
- Statistical results
- Canvas specifications

All validation functions follow the pattern:
- Return True if valid
- Raise ValidationError with detailed message if invalid
"""

from typing import Dict, Any, List, Optional, Union


class ValidationError(Exception):
    """
    Exception raised when schema validation fails.

    Attributes
    ----------
    message : str
        Explanation of the validation error
    field : str, optional
        The specific field that failed validation
    value : Any, optional
        The invalid value
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
    ):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.field and self.value is not None:
            return f"{self.message} (field='{self.field}', value={self.value!r})"
        elif self.field:
            return f"{self.message} (field='{self.field}')"
        return self.message


# =============================================================================
# Figure Validation
# =============================================================================


def validate_figure(fig_data: Dict[str, Any]) -> bool:
    """
    Validate a figure specification.

    Parameters
    ----------
    fig_data : Dict[str, Any]
        Figure specification dictionary

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ValidationError
        If validation fails
    """
    # Required fields
    required_fields = ["width_mm", "height_mm"]

    for field in required_fields:
        if field not in fig_data:
            raise ValidationError(f"Missing required field: {field}", field=field)

    # Type validation
    if not isinstance(fig_data["width_mm"], (int, float)):
        raise ValidationError(
            "width_mm must be a number",
            field="width_mm",
            value=fig_data["width_mm"],
        )

    if not isinstance(fig_data["height_mm"], (int, float)):
        raise ValidationError(
            "height_mm must be a number",
            field="height_mm",
            value=fig_data["height_mm"],
        )

    # Value validation
    if fig_data["width_mm"] <= 0:
        raise ValidationError(
            "width_mm must be positive",
            field="width_mm",
            value=fig_data["width_mm"],
        )

    if fig_data["height_mm"] <= 0:
        raise ValidationError(
            "height_mm must be positive",
            field="height_mm",
            value=fig_data["height_mm"],
        )

    # Optional fields validation
    if "nrows" in fig_data:
        if not isinstance(fig_data["nrows"], int) or fig_data["nrows"] <= 0:
            raise ValidationError(
                "nrows must be a positive integer",
                field="nrows",
                value=fig_data["nrows"],
            )

    if "ncols" in fig_data:
        if not isinstance(fig_data["ncols"], int) or fig_data["ncols"] <= 0:
            raise ValidationError(
                "ncols must be a positive integer",
                field="ncols",
                value=fig_data["ncols"],
            )

    if "dpi" in fig_data:
        if not isinstance(fig_data["dpi"], int) or fig_data["dpi"] <= 0:
            raise ValidationError(
                "dpi must be a positive integer",
                field="dpi",
                value=fig_data["dpi"],
            )

    if "axes" in fig_data:
        if not isinstance(fig_data["axes"], list):
            raise ValidationError(
                "axes must be a list",
                field="axes",
                value=type(fig_data["axes"]).__name__,
            )

        # Validate axes layout
        nrows = fig_data.get("nrows", 1)
        ncols = fig_data.get("ncols", 1)
        num_axes = len(fig_data["axes"])
        max_axes = nrows * ncols

        if num_axes > max_axes:
            raise ValidationError(
                f"Too many axes: {num_axes} axes for {nrows}x{ncols} layout (max {max_axes})",
                field="axes",
            )

        # Validate each axes entry
        for i, axes_data in enumerate(fig_data["axes"]):
            try:
                validate_axes(axes_data)
            except ValidationError as e:
                raise ValidationError(
                    f"Invalid axes at index {i}: {e.message}",
                    field=f"axes[{i}]",
                )

    return True


def validate_axes(axes_data: Dict[str, Any]) -> bool:
    """
    Validate an axes specification.

    Parameters
    ----------
    axes_data : Dict[str, Any]
        Axes specification dictionary

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ValidationError
        If validation fails
    """
    # Optional row/col must be non-negative integers
    if "row" in axes_data:
        if not isinstance(axes_data["row"], int) or axes_data["row"] < 0:
            raise ValidationError(
                "row must be a non-negative integer",
                field="row",
                value=axes_data["row"],
            )

    if "col" in axes_data:
        if not isinstance(axes_data["col"], int) or axes_data["col"] < 0:
            raise ValidationError(
                "col must be a non-negative integer",
                field="col",
                value=axes_data["col"],
            )

    # Validate scale values
    valid_scales = {"linear", "log", "symlog", "logit"}

    if "xscale" in axes_data:
        if axes_data["xscale"] not in valid_scales:
            raise ValidationError(
                f"xscale must be one of {valid_scales}",
                field="xscale",
                value=axes_data["xscale"],
            )

    if "yscale" in axes_data:
        if axes_data["yscale"] not in valid_scales:
            raise ValidationError(
                f"yscale must be one of {valid_scales}",
                field="yscale",
                value=axes_data["yscale"],
            )

    # Validate limits
    for lim_field in ["xlim", "ylim"]:
        if lim_field in axes_data:
            lim = axes_data[lim_field]
            if not isinstance(lim, (list, tuple)) or len(lim) != 2:
                raise ValidationError(
                    f"{lim_field} must be a list/tuple of [min, max]",
                    field=lim_field,
                    value=lim,
                )

    # Validate plots
    if "plots" in axes_data:
        if not isinstance(axes_data["plots"], list):
            raise ValidationError(
                "plots must be a list",
                field="plots",
                value=type(axes_data["plots"]).__name__,
            )

        for i, plot_data in enumerate(axes_data["plots"]):
            try:
                validate_plot(plot_data)
            except ValidationError as e:
                raise ValidationError(
                    f"Invalid plot at index {i}: {e.message}",
                    field=f"plots[{i}]",
                )

    return True


def validate_plot(plot_data: Dict[str, Any]) -> bool:
    """
    Validate a plot specification.

    Parameters
    ----------
    plot_data : Dict[str, Any]
        Plot specification dictionary

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ValidationError
        If validation fails
    """
    # plot_type is required
    if "plot_type" not in plot_data:
        raise ValidationError("Plot must specify plot_type", field="plot_type")

    plot_type = plot_data["plot_type"]
    data = plot_data.get("data", {})

    # Type-specific requirements
    if plot_type in ["line", "scatter", "errorbar"]:
        if "x" not in data or "y" not in data:
            raise ValidationError(
                f"{plot_type} requires 'x' and 'y' data",
                field="data",
            )

        # Validate arrays have same length
        x_len = len(data["x"]) if hasattr(data["x"], "__len__") else 1
        y_len = len(data["y"]) if hasattr(data["y"], "__len__") else 1

        if x_len != y_len:
            raise ValidationError(
                f"x and y data must have same length: {x_len} != {y_len}",
                field="data",
            )

    elif plot_type in ["bar", "barh"]:
        if "x" not in data:
            raise ValidationError(
                f"{plot_type} requires 'x' data",
                field="data.x",
            )

        if "height" not in data and "y" not in data:
            raise ValidationError(
                f"{plot_type} requires 'height' or 'y' data",
                field="data",
            )

    elif plot_type == "hist":
        if "x" not in data:
            raise ValidationError(
                "hist requires 'x' data",
                field="data.x",
            )

    elif plot_type in ["heatmap", "imshow"]:
        if "z" not in data and "img" not in data:
            raise ValidationError(
                f"{plot_type} requires 'z' or 'img' data",
                field="data",
            )

    elif plot_type in ["contour", "contourf"]:
        if "x" not in data or "y" not in data or "z" not in data:
            raise ValidationError(
                f"{plot_type} requires 'x', 'y', and 'z' data",
                field="data",
            )

    return True


# =============================================================================
# Statistical Result Validation
# =============================================================================


def validate_stat_result(stat_data: Dict[str, Any]) -> bool:
    """
    Validate a statistical result specification.

    Parameters
    ----------
    stat_data : Dict[str, Any]
        Statistical result dictionary

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ValidationError
        If validation fails
    """
    # Required fields
    required_fields = ["test_type", "test_category", "statistic", "p_value", "stars"]

    for field in required_fields:
        if field not in stat_data:
            raise ValidationError(f"Missing required field: {field}", field=field)

    # Validate statistic structure
    statistic = stat_data["statistic"]
    if not isinstance(statistic, dict):
        raise ValidationError(
            "statistic must be a dictionary",
            field="statistic",
            value=type(statistic).__name__,
        )

    if "name" not in statistic or "value" not in statistic:
        raise ValidationError(
            "statistic must have 'name' and 'value' keys",
            field="statistic",
        )

    # Validate p_value
    p_value = stat_data["p_value"]
    if not isinstance(p_value, (int, float)):
        raise ValidationError(
            "p_value must be a number",
            field="p_value",
            value=p_value,
        )

    if p_value < 0 or p_value > 1:
        raise ValidationError(
            "p_value must be between 0 and 1",
            field="p_value",
            value=p_value,
        )

    # Validate stars
    valid_stars = {"***", "**", "*", "ns", ""}
    stars = stat_data["stars"]
    if stars not in valid_stars:
        raise ValidationError(
            f"stars must be one of {valid_stars}",
            field="stars",
            value=stars,
        )

    # Validate test_category
    valid_categories = {"parametric", "non-parametric", "correlation", "other"}
    category = stat_data["test_category"]
    if category not in valid_categories:
        raise ValidationError(
            f"test_category must be one of {valid_categories}",
            field="test_category",
            value=category,
        )

    # Validate effect_size structure if present
    if "effect_size" in stat_data and stat_data["effect_size"]:
        effect_size = stat_data["effect_size"]
        if not isinstance(effect_size, dict):
            raise ValidationError(
                "effect_size must be a dictionary",
                field="effect_size",
                value=type(effect_size).__name__,
            )

        if "name" not in effect_size or "value" not in effect_size:
            raise ValidationError(
                "effect_size must have 'name' and 'value' keys",
                field="effect_size",
            )

    return True


# =============================================================================
# Color Validation
# =============================================================================


def validate_color(color: Any) -> bool:
    """
    Validate a color specification.

    Parameters
    ----------
    color : Any
        Color specification (name, hex, rgb, etc.)

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ValidationError
        If validation fails
    """
    if color is None:
        raise ValidationError("Color cannot be None", field="color", value=color)

    if not isinstance(color, str):
        raise ValidationError(
            f"Color must be a string, got {type(color).__name__}",
            field="color",
            value=color,
        )

    if not color:
        raise ValidationError("Color cannot be empty string", field="color", value=color)

    return True


__all__ = [
    "ValidationError",
    "validate_figure",
    "validate_axes",
    "validate_plot",
    "validate_stat_result",
    "validate_color",
]


# EOF
