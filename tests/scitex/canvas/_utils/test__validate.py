# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_utils/_validate.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # File: ./src/scitex/vis/utils/validate.py
# """Validation utilities for figure JSON specifications."""
# 
# from typing import Any, Dict
# 
# 
# def validate_json_structure(fig_json: Dict[str, Any]) -> bool:
#     """
#     Validate basic JSON structure requirements.
# 
#     Parameters
#     ----------
#     fig_json : Dict[str, Any]
#         Figure JSON to validate
# 
#     Returns
#     -------
#     bool
#         True if valid, raises ValueError otherwise
# 
#     Raises
#     ------
#     ValueError
#         If JSON structure is invalid
#     """
#     # Required fields
#     required_fields = ["width_mm", "height_mm"]
# 
#     for field in required_fields:
#         if field not in fig_json:
#             raise ValueError(f"Missing required field: {field}")
# 
#     # Type validation
#     if not isinstance(fig_json["width_mm"], (int, float)):
#         raise ValueError("width_mm must be a number")
# 
#     if not isinstance(fig_json["height_mm"], (int, float)):
#         raise ValueError("height_mm must be a number")
# 
#     # Value validation
#     if fig_json["width_mm"] <= 0:
#         raise ValueError(f"width_mm must be positive, got {fig_json['width_mm']}")
# 
#     if fig_json["height_mm"] <= 0:
#         raise ValueError(f"height_mm must be positive, got {fig_json['height_mm']}")
# 
#     # Optional fields validation
#     if "nrows" in fig_json:
#         if not isinstance(fig_json["nrows"], int) or fig_json["nrows"] <= 0:
#             raise ValueError("nrows must be a positive integer")
# 
#     if "ncols" in fig_json:
#         if not isinstance(fig_json["ncols"], int) or fig_json["ncols"] <= 0:
#             raise ValueError("ncols must be a positive integer")
# 
#     if "dpi" in fig_json:
#         if not isinstance(fig_json["dpi"], int) or fig_json["dpi"] <= 0:
#             raise ValueError("dpi must be a positive integer")
# 
#     if "axes" in fig_json:
#         if not isinstance(fig_json["axes"], list):
#             raise ValueError("axes must be a list")
# 
#     return True
# 
# 
# def validate_plot_data(plot_data: Dict[str, Any]) -> bool:
#     """
#     Validate plot data contains required fields for plot type.
# 
#     Parameters
#     ----------
#     plot_data : Dict[str, Any]
#         Plot configuration with data
# 
#     Returns
#     -------
#     bool
#         True if valid, raises ValueError otherwise
# 
#     Raises
#     ------
#     ValueError
#         If plot data is invalid
#     """
#     if "plot_type" not in plot_data:
#         raise ValueError("Plot must specify plot_type")
# 
#     plot_type = plot_data["plot_type"]
#     data = plot_data.get("data", {})
# 
#     # Type-specific requirements
#     if plot_type in ["line", "scatter", "errorbar"]:
#         if "x" not in data or "y" not in data:
#             raise ValueError(f"{plot_type} requires 'x' and 'y' data")
# 
#         # Validate arrays have same length
#         x_len = len(data["x"]) if hasattr(data["x"], "__len__") else 1
#         y_len = len(data["y"]) if hasattr(data["y"], "__len__") else 1
# 
#         if x_len != y_len:
#             raise ValueError(f"x and y data must have same length: {x_len} != {y_len}")
# 
#     elif plot_type in ["bar", "barh"]:
#         if "x" not in data:
#             raise ValueError(f"{plot_type} requires 'x' data")
# 
#         if "height" not in data and "y" not in data:
#             raise ValueError(f"{plot_type} requires 'height' or 'y' data")
# 
#     elif plot_type == "hist":
#         if "x" not in data:
#             raise ValueError("hist requires 'x' data")
# 
#     elif plot_type in ["heatmap", "imshow"]:
#         if "z" not in data and "img" not in data:
#             raise ValueError(f"{plot_type} requires 'z' or 'img' data")
# 
#     elif plot_type in ["contour", "contourf"]:
#         if "x" not in data or "y" not in data or "z" not in data:
#             raise ValueError(f"{plot_type} requires 'x', 'y', and 'z' data")
# 
#     return True
# 
# 
# def check_schema_version(fig_json: Dict[str, Any]) -> str:
#     """
#     Check and return schema version.
# 
#     Parameters
#     ----------
#     fig_json : Dict[str, Any]
#         Figure JSON
# 
#     Returns
#     -------
#     str
#         Schema version (defaults to "1.0.0" if not specified)
#     """
#     return fig_json.get("schema_version", "1.0.0")
# 
# 
# def validate_color(color: str) -> bool:
#     """
#     Validate color specification.
# 
#     Parameters
#     ----------
#     color : str
#         Color specification (name, hex, rgb, etc.)
# 
#     Returns
#     -------
#     bool
#         True if valid, raises ValueError otherwise
#     """
#     if not isinstance(color, str):
#         raise ValueError(f"Color must be a string, got {type(color)}")
# 
#     # Basic validation - matplotlib will do deeper validation
#     if not color:
#         raise ValueError("Color cannot be empty string")
# 
#     return True
# 
# 
# def validate_axes_layout(nrows: int, ncols: int, num_axes: int) -> bool:
#     """
#     Validate axes layout is consistent.
# 
#     Parameters
#     ----------
#     nrows : int
#         Number of rows
#     ncols : int
#         Number of columns
#     num_axes : int
#         Number of axes configurations
# 
#     Returns
#     -------
#     bool
#         True if valid, raises ValueError otherwise
#     """
#     max_axes = nrows * ncols
# 
#     if num_axes > max_axes:
#         raise ValueError(
#             f"Too many axes: {num_axes} axes for {nrows}x{ncols} layout "
#             f"(max {max_axes})"
#         )
# 
#     return True
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_utils/_validate.py
# --------------------------------------------------------------------------------
