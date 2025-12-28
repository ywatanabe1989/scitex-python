# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/schema/_plot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-13 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/schema/_plot.py
# 
# """
# Plot Schema - Canonical source of truth for plot specifications.
# 
# This module defines the layered schema architecture for plots:
# 
# 1. PlotSpec (spec.json) - Semantic definition: WHAT to plot
#    - Traces with type and column mappings
#    - Axes configuration (labels, limits)
#    - Data source reference
# 
# 2. PlotStyle (style.json) - Appearance: HOW it looks
#    - Theme and colors
#    - Line widths, marker sizes
#    - Font settings
# 
# 3. PlotGeometry (cache/geometry_px.json) - Rendered positions: WHERE (cache)
#    - Pixel coordinates
#    - Path data for hit testing
#    - Bounding boxes in px
# 
# 4. RenderManifest (cache/render_manifest.json) - Render metadata
#    - DPI, figure size
#    - Source hash for cache invalidation
# 
# Design Principles:
# - Canonical data uses ratio (0-1) for axes bbox, mm for panel size
# - px data is ALWAYS derived/cached, never source of truth
# - Traces are semantic (boxplot, heatmap) not decomposed (line segments)
# """
# 
# from dataclasses import dataclass, field, asdict
# from typing import Dict, Any, List, Optional, Literal, Union
# from datetime import datetime
# import json
# 
# 
# # Schema versions
# PLOT_SPEC_VERSION = "1.0.0"
# PLOT_STYLE_VERSION = "1.0.0"
# PLOT_GEOMETRY_VERSION = "1.0.0"
# 
# # DPI fallback for legacy data without explicit DPI
# # Note: For dynamic DPI resolution, use scitex.plt.styles.get_default_dpi()
# # This constant is only used as a fallback when parsing data without DPI info
# DPI_FALLBACK = 300
# 
# 
# # =============================================================================
# # Type Aliases
# # =============================================================================
# 
# TraceType = Literal[
#     # Line-based
#     "line", "step", "stem",
#     # Scatter-based
#     "scatter", "hexbin",
#     # Distribution
#     "histogram", "kde", "ecdf", "boxplot", "violinplot", "joyplot",
#     # Categorical
#     "bar", "barh",
#     # 2D/Grid
#     "heatmap", "imshow", "contour", "contourf", "pcolormesh",
#     # Statistical
#     "errorbar", "fill_between", "mean_std", "mean_ci", "median_iqr",
#     # Vector
#     "quiver", "streamplot",
#     # Special
#     "pie", "raster", "rectangle",
#     # Generic fallback
#     "unknown",
# ]
# 
# CoordinateSpace = Literal["panel", "figure", "data"]
# 
# 
# # =============================================================================
# # Bounding Box Specs
# # =============================================================================
# 
# 
# @dataclass
# class BboxRatio:
#     """
#     Bounding box in normalized coordinates (0-1).
# 
#     This is the CANONICAL representation for axes position within a panel.
#     """
#     x0: float
#     y0: float
#     width: float
#     height: float
#     space: CoordinateSpace = "panel"
# 
#     @property
#     def x1(self) -> float:
#         return self.x0 + self.width
# 
#     @property
#     def y1(self) -> float:
#         return self.y0 + self.height
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "x0": self.x0,
#             "y0": self.y0,
#             "width": self.width,
#             "height": self.height,
#             "space": self.space,
#         }
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "BboxRatio":
#         # Handle both width/height and x1/y1 formats
#         if "width" not in data and "x1" in data:
#             data = data.copy()
#             data["width"] = data["x1"] - data["x0"]
#             data["height"] = data["y1"] - data["y0"]
#             data.pop("x1", None)
#             data.pop("y1", None)
#         return cls(**{k: v for k, v in data.items() if k in ["x0", "y0", "width", "height", "space"]})
# 
# 
# @dataclass
# class BboxPx:
#     """
#     Bounding box in pixel coordinates.
# 
#     This is DERIVED/CACHED, not canonical.
#     """
#     x0: float
#     y0: float
#     width: float
#     height: float
# 
#     @property
#     def x1(self) -> float:
#         return self.x0 + self.width
# 
#     @property
#     def y1(self) -> float:
#         return self.y0 + self.height
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "x0": self.x0,
#             "y0": self.y0,
#             "x1": self.x1,
#             "y1": self.y1,
#             "width": self.width,
#             "height": self.height,
#         }
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "BboxPx":
#         if "width" not in data and "x1" in data:
#             data = data.copy()
#             data["width"] = data["x1"] - data["x0"]
#             data["height"] = data["y1"] - data["y0"]
#         return cls(
#             x0=data["x0"],
#             y0=data["y0"],
#             width=data.get("width", 0),
#             height=data.get("height", 0),
#         )
# 
# 
# # =============================================================================
# # Trace Specification (Semantic)
# # =============================================================================
# 
# 
# @dataclass
# class TraceSpec:
#     """
#     Semantic specification for a single trace/artist.
# 
#     This captures WHAT the user intended to plot, not how it was rendered.
# 
#     Parameters
#     ----------
#     id : str
#         Unique identifier for this trace
#     type : TraceType
#         Semantic type (boxplot, heatmap, line, etc.)
#     x_col : str, optional
#         Column name for x data (line, scatter, bar, etc.)
#     y_col : str, optional
#         Column name for y data (line, scatter)
#     data_cols : list, optional
#         Column names for multi-column data (boxplot, violinplot)
#     value_col : str, optional
#         Column name for values (heatmap, contour)
#     u_col, v_col : str, optional
#         Column names for vector components (quiver, streamplot)
#     label : str, optional
#         Legend label
#     group : str, optional
#         Grouping identifier for related traces
# 
#     Examples
#     --------
#     >>> # Line plot
#     >>> TraceSpec(id="line-0", type="line", x_col="time", y_col="signal", label="EEG")
# 
#     >>> # Boxplot with 4 groups
#     >>> TraceSpec(id="box-0", type="boxplot", data_cols=["A", "B", "C", "D"])
# 
#     >>> # Heatmap
#     >>> TraceSpec(id="hmap-0", type="heatmap", x_col="x", y_col="y", value_col="z")
# 
#     >>> # Quiver (vector field)
#     >>> TraceSpec(id="quiv-0", type="quiver", x_col="x", y_col="y", u_col="u", v_col="v")
#     """
#     id: str
#     type: TraceType
# 
#     # Column mappings (usage depends on trace type)
#     x_col: Optional[str] = None
#     y_col: Optional[str] = None
#     data_cols: Optional[List[str]] = None  # For boxplot, violin, etc.
#     value_col: Optional[str] = None  # For heatmap, contour
#     u_col: Optional[str] = None  # For quiver
#     v_col: Optional[str] = None  # For quiver
# 
#     # Metadata
#     label: Optional[str] = None
#     group: Optional[str] = None
#     axes_index: int = 0  # Which axes this trace belongs to
# 
#     # Additional type-specific parameters
#     extra: Dict[str, Any] = field(default_factory=dict)
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {
#             "id": self.id,
#             "type": self.type,
#             "axes_index": self.axes_index,
#         }
#         # Only include non-None fields
#         if self.x_col:
#             result["x_col"] = self.x_col
#         if self.y_col:
#             result["y_col"] = self.y_col
#         if self.data_cols:
#             result["data_cols"] = self.data_cols
#         if self.value_col:
#             result["value_col"] = self.value_col
#         if self.u_col:
#             result["u_col"] = self.u_col
#         if self.v_col:
#             result["v_col"] = self.v_col
#         if self.label:
#             result["label"] = self.label
#         if self.group:
#             result["group"] = self.group
#         if self.extra:
#             result["extra"] = self.extra
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "TraceSpec":
#         return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
# 
# 
# # =============================================================================
# # Axes Specification (Semantic)
# # =============================================================================
# 
# 
# @dataclass
# class AxesLimits:
#     """Axis limits specification."""
#     x: Optional[List[float]] = None  # [xmin, xmax]
#     y: Optional[List[float]] = None  # [ymin, ymax]
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {}
#         if self.x:
#             result["x"] = self.x
#         if self.y:
#             result["y"] = self.y
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "AxesLimits":
#         return cls(x=data.get("x"), y=data.get("y"))
# 
# 
# @dataclass
# class AxesLabels:
#     """Axes labels specification."""
#     xlabel: Optional[str] = None
#     ylabel: Optional[str] = None
#     title: Optional[str] = None
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {}
#         if self.xlabel:
#             result["xlabel"] = self.xlabel
#         if self.ylabel:
#             result["ylabel"] = self.ylabel
#         if self.title:
#             result["title"] = self.title
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "AxesLabels":
#         return cls(**{k: v for k, v in data.items() if k in ["xlabel", "ylabel", "title"]})
# 
# 
# @dataclass
# class AxesSpecItem:
#     """
#     Specification for a single axes within a plot.
# 
#     Parameters
#     ----------
#     id : str
#         Unique identifier (e.g., "ax0", "colorbar")
#     bbox : BboxRatio
#         Position in normalized coordinates (0-1) within the panel
#     labels : AxesLabels
#         Axis labels and title
#     limits : AxesLimits, optional
#         Axis limits (auto if not specified)
#     role : str
#         Role of this axes ("main", "colorbar", "inset", etc.)
#     linked_to : str, optional
#         ID of axes this is linked to (e.g., colorbar linked to heatmap axes)
#     """
#     id: str
#     bbox: BboxRatio
#     labels: AxesLabels = field(default_factory=AxesLabels)
#     limits: Optional[AxesLimits] = None
#     role: str = "main"  # "main", "colorbar", "inset", "twinx", "twiny"
#     linked_to: Optional[str] = None
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {
#             "id": self.id,
#             "bbox": self.bbox.to_dict(),
#             "labels": self.labels.to_dict(),
#             "role": self.role,
#         }
#         if self.limits:
#             result["limits"] = self.limits.to_dict()
#         if self.linked_to:
#             result["linked_to"] = self.linked_to
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "AxesSpecItem":
#         data_copy = data.copy()
#         if "bbox" in data_copy:
#             data_copy["bbox"] = BboxRatio.from_dict(data_copy["bbox"])
#         else:
#             # Default bbox
#             data_copy["bbox"] = BboxRatio(x0=0.15, y0=0.15, width=0.7, height=0.7)
#         if "labels" in data_copy:
#             data_copy["labels"] = AxesLabels.from_dict(data_copy["labels"])
#         if "limits" in data_copy and data_copy["limits"]:
#             data_copy["limits"] = AxesLimits.from_dict(data_copy["limits"])
#         return cls(**{k: v for k, v in data_copy.items() if k in cls.__dataclass_fields__})
# 
# 
# # =============================================================================
# # Data Source Specification
# # =============================================================================
# 
# 
# @dataclass
# class DataSourceSpec:
#     """
#     Specification for the data source.
# 
#     Parameters
#     ----------
#     csv : str
#         Relative path to CSV file
#     format : str
#         Data format ("wide" or "long")
#     hash : str, optional
#         Content hash for integrity verification
#     """
#     csv: str
#     format: str = "wide"
#     hash: Optional[str] = None
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {"csv": self.csv, "format": self.format}
#         if self.hash:
#             result["hash"] = self.hash
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "DataSourceSpec":
#         # Handle legacy "source" or "path" keys
#         csv = data.get("csv") or data.get("source") or data.get("path", "")
#         return cls(
#             csv=csv,
#             format=data.get("format", "wide"),
#             hash=data.get("hash"),
#         )
# 
# 
# # =============================================================================
# # PlotSpec - Main Semantic Specification
# # =============================================================================
# 
# 
# @dataclass
# class PlotSpec:
#     """
#     Complete semantic specification for a plot.
# 
#     This is the SOURCE OF TRUTH stored in spec.json.
#     Contains only semantic information about WHAT to plot.
# 
#     Parameters
#     ----------
#     plot_id : str
#         Unique identifier for this plot
#     data : DataSourceSpec
#         Data source specification
#     axes : list of AxesSpecItem
#         Axes configurations
#     traces : list of TraceSpec
#         Trace/artist specifications
# 
#     Examples
#     --------
#     >>> spec = PlotSpec(
#     ...     plot_id="panel_A",
#     ...     data=DataSourceSpec(csv="data.csv"),
#     ...     axes=[AxesSpecItem(id="ax0", bbox=BboxRatio(0.15, 0.15, 0.7, 0.7))],
#     ...     traces=[TraceSpec(id="line-0", type="line", x_col="x", y_col="y")]
#     ... )
#     >>> spec.to_json()
#     """
#     plot_id: str
#     data: DataSourceSpec
#     axes: List[AxesSpecItem] = field(default_factory=list)
#     traces: List[TraceSpec] = field(default_factory=list)
# 
#     # Schema metadata
#     scitex_schema: str = "scitex.plt.spec"
#     scitex_schema_version: str = PLOT_SPEC_VERSION
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "schema": {
#                 "name": self.scitex_schema,
#                 "version": self.scitex_schema_version,
#             },
#             "plot_id": self.plot_id,
#             "data": self.data.to_dict(),
#             "axes": [ax.to_dict() for ax in self.axes],
#             "traces": [tr.to_dict() for tr in self.traces],
#         }
# 
#     def to_json(self, indent: int = 2) -> str:
#         return json.dumps(self.to_dict(), indent=indent)
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "PlotSpec":
#         schema_info = data.get("schema", {})
#         return cls(
#             plot_id=data.get("plot_id", ""),
#             data=DataSourceSpec.from_dict(data.get("data", {})),
#             axes=[AxesSpecItem.from_dict(ax) for ax in data.get("axes", [])],
#             traces=[TraceSpec.from_dict(tr) for tr in data.get("traces", [])],
#             scitex_schema=schema_info.get("name", "scitex.plt.spec"),
#             scitex_schema_version=schema_info.get("version", PLOT_SPEC_VERSION),
#         )
# 
#     @classmethod
#     def from_json(cls, json_str: str) -> "PlotSpec":
#         return cls.from_dict(json.loads(json_str))
# 
# 
# # =============================================================================
# # PlotStyle - Appearance Specification
# # =============================================================================
# 
# 
# @dataclass
# class TraceStyleSpec:
#     """Style overrides for a specific trace."""
#     trace_id: str
#     color: Optional[str] = None
#     linewidth: Optional[float] = None
#     linestyle: Optional[str] = None
#     marker: Optional[str] = None
#     markersize: Optional[float] = None
#     alpha: Optional[float] = None
#     extra: Dict[str, Any] = field(default_factory=dict)
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {"trace_id": self.trace_id}
#         for field_name in ["color", "linewidth", "linestyle", "marker", "markersize", "alpha"]:
#             val = getattr(self, field_name)
#             if val is not None:
#                 result[field_name] = val
#         if self.extra:
#             result["extra"] = self.extra
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "TraceStyleSpec":
#         return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
# 
# 
# @dataclass
# class ThemeSpec:
#     """Theme specification."""
#     mode: str = "light"  # "light", "dark", "auto"
#     colors: Dict[str, str] = field(default_factory=lambda: {
#         "background": "transparent",
#         "axes_bg": "white",
#         "text": "black",
#         "spine": "black",
#         "tick": "black",
#     })
#     palette: Optional[str] = None  # Color palette name
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {"mode": self.mode, "colors": self.colors}
#         if self.palette:
#             result["palette"] = self.palette
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "ThemeSpec":
#         return cls(
#             mode=data.get("mode", "light"),
#             colors=data.get("colors", {}),
#             palette=data.get("palette"),
#         )
# 
# 
# @dataclass
# class FontSpec:
#     """Font specification."""
#     family: str = "sans-serif"
#     size_pt: float = 7.0
#     title_size_pt: float = 8.0
#     label_size_pt: float = 7.0
#     tick_size_pt: float = 6.0
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return asdict(self)
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "FontSpec":
#         return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
# 
# 
# @dataclass
# class SizeSpec:
#     """Panel size specification (canonical in mm)."""
#     width_mm: float = 80.0
#     height_mm: float = 68.0
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return asdict(self)
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "SizeSpec":
#         return cls(
#             width_mm=data.get("width_mm", 80.0),
#             height_mm=data.get("height_mm", 68.0),
#         )
# 
# 
# # Valid matplotlib legend location strings
# LegendLocation = Literal[
#     "best", "upper right", "upper left", "lower left", "lower right",
#     "right", "center left", "center right", "lower center", "upper center",
#     "center",
# ]
# 
# 
# @dataclass
# class LegendSpec:
#     """
#     Legend configuration specification.
# 
#     Parameters
#     ----------
#     visible : bool
#         Whether to show the legend (default True)
#     location : str
#         Legend location. Valid values:
#         - "best" (auto-placement)
#         - "upper right", "upper left", "lower right", "lower left"
#         - "right", "center left", "center right"
#         - "upper center", "lower center", "center"
#     frameon : bool
#         Whether to draw a frame around the legend
#     fontsize : float, optional
#         Font size for legend text (in points)
#     ncols : int
#         Number of columns in the legend
#     title : str, optional
#         Legend title
#     """
#     visible: bool = True
#     location: str = "best"
#     frameon: bool = True
#     fontsize: Optional[float] = None
#     ncols: int = 1
#     title: Optional[str] = None
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {
#             "visible": self.visible,
#             "location": self.location,
#             "frameon": self.frameon,
#             "ncols": self.ncols,
#         }
#         if self.fontsize is not None:
#             result["fontsize"] = self.fontsize
#         if self.title is not None:
#             result["title"] = self.title
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "LegendSpec":
#         # Handle backward compatibility: boolean legend value
#         if isinstance(data, bool):
#             return cls(visible=data)
#         return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
# 
# 
# @dataclass
# class PlotStyle:
#     """
#     Appearance specification for a plot.
# 
#     Stored in style.json. Contains HOW the plot looks.
#     Only stores overrides from defaults.
# 
#     Parameters
#     ----------
#     theme : ThemeSpec
#         Theme configuration
#     size : SizeSpec
#         Panel size in mm (canonical unit)
#     font : FontSpec
#         Font settings
#     traces : list of TraceStyleSpec
#         Per-trace style overrides
#     legend : LegendSpec
#         Legend configuration (visibility, location, styling)
#     grid : bool
#         Whether to show grid lines
#     """
#     theme: ThemeSpec = field(default_factory=ThemeSpec)
#     size: SizeSpec = field(default_factory=SizeSpec)
#     font: FontSpec = field(default_factory=FontSpec)
#     traces: List[TraceStyleSpec] = field(default_factory=list)
#     legend: LegendSpec = field(default_factory=LegendSpec)
# 
#     # Axes-level overrides
#     grid: bool = False
# 
#     # Schema metadata
#     scitex_schema: str = "scitex.plt.style"
#     scitex_schema_version: str = PLOT_STYLE_VERSION
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "schema": {
#                 "name": self.scitex_schema,
#                 "version": self.scitex_schema_version,
#             },
#             "theme": self.theme.to_dict(),
#             "size": self.size.to_dict(),
#             "font": self.font.to_dict(),
#             "traces": [tr.to_dict() for tr in self.traces],
#             "legend": self.legend.to_dict(),
#             "grid": self.grid,
#         }
# 
#     def to_json(self, indent: int = 2) -> str:
#         return json.dumps(self.to_dict(), indent=indent)
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "PlotStyle":
#         # Handle backward compatibility: legend can be bool or dict
#         legend_data = data.get("legend", True)
#         if isinstance(legend_data, bool):
#             legend = LegendSpec(visible=legend_data)
#         elif isinstance(legend_data, dict):
#             legend = LegendSpec.from_dict(legend_data)
#         else:
#             legend = LegendSpec()
# 
#         return cls(
#             theme=ThemeSpec.from_dict(data.get("theme", {})),
#             size=SizeSpec.from_dict(data.get("size", {})),
#             font=FontSpec.from_dict(data.get("font", {})),
#             traces=[TraceStyleSpec.from_dict(tr) for tr in data.get("traces", [])],
#             legend=legend,
#             grid=data.get("grid", False),
#         )
# 
#     @classmethod
#     def from_json(cls, json_str: str) -> "PlotStyle":
#         return cls.from_dict(json.loads(json_str))
# 
# 
# # =============================================================================
# # PlotGeometry - Cached Render Output
# # =============================================================================
# 
# 
# @dataclass
# class RenderedArtist:
#     """
#     Cached pixel-level data for a rendered artist.
# 
#     This is DERIVED from PlotSpec + PlotStyle, not source of truth.
#     """
#     id: str
#     type: str
#     axes_index: int
#     label: Optional[str] = None
#     bbox_px: Optional[BboxPx] = None
#     path_px: Optional[List[List[float]]] = None  # [[x, y], [x, y], ...]
#     extra: Dict[str, Any] = field(default_factory=dict)
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {
#             "id": self.id,
#             "type": self.type,
#             "axes_index": self.axes_index,
#         }
#         if self.label:
#             result["label"] = self.label
#         if self.bbox_px:
#             result["bbox_px"] = self.bbox_px.to_dict()
#         if self.path_px:
#             result["path_px"] = self.path_px
#         if self.extra:
#             result.update(self.extra)
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "RenderedArtist":
#         data_copy = data.copy()
#         if "bbox_px" in data_copy and data_copy["bbox_px"]:
#             data_copy["bbox_px"] = BboxPx.from_dict(data_copy["bbox_px"])
#         return cls(**{k: v for k, v in data_copy.items() if k in cls.__dataclass_fields__})
# 
# 
# @dataclass
# class RenderedAxes:
#     """Cached pixel-level data for rendered axes."""
#     id: str
#     xlim: List[float]
#     ylim: List[float]
#     bbox_px: BboxPx
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "id": self.id,
#             "xlim": self.xlim,
#             "ylim": self.ylim,
#             "bbox_px": self.bbox_px.to_dict(),
#         }
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "RenderedAxes":
#         return cls(
#             id=data.get("id", "ax0"),
#             xlim=data.get("xlim", [0, 1]),
#             ylim=data.get("ylim", [0, 1]),
#             bbox_px=BboxPx.from_dict(data.get("bbox_px", {"x0": 0, "y0": 0, "width": 100, "height": 100})),
#         )
# 
# 
# @dataclass
# class HitRegionEntry:
#     """Entry in the hit region color map."""
#     id: int
#     type: str
#     label: str
#     axes_index: int
#     rgb: List[int]
#     group_id: Optional[str] = None
#     role: str = "standalone"
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return asdict(self)
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "HitRegionEntry":
#         return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
# 
# 
# @dataclass
# class SelectableRegion:
#     """Selectable region for GUI interaction."""
#     bbox_px: List[float]  # [x0, y0, x1, y1]
#     text: Optional[str] = None
#     fontsize: Optional[float] = None
#     color: Optional[str] = None
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {"bbox_px": self.bbox_px}
#         if self.text:
#             result["text"] = self.text
#         if self.fontsize:
#             result["fontsize"] = self.fontsize
#         if self.color:
#             result["color"] = self.color
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "SelectableRegion":
#         return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
# 
# 
# @dataclass
# class PlotGeometry:
#     """
#     Cached geometry data for a rendered plot.
# 
#     Stored in cache/geometry_px.json. This is DERIVED output.
#     Can be deleted and regenerated from PlotSpec + PlotStyle.
# 
#     Parameters
#     ----------
#     source_hash : str
#         Hash of spec + style that produced this geometry
#     figure_px : tuple
#         Rendered figure size (width, height) in pixels
#     dpi : int
#         DPI used for rendering
#     axes : list of RenderedAxes
#         Pixel-level axes data
#     artists : list of RenderedArtist
#         Pixel-level artist data
#     hit_regions : dict
#         Hit testing data (color_map, groups)
#     selectable_regions : dict
#         GUI-selectable regions
#     """
#     source_hash: str
#     figure_px: List[int]  # [width, height]
#     dpi: int
#     axes: List[RenderedAxes] = field(default_factory=list)
#     artists: List[RenderedArtist] = field(default_factory=list)
#     hit_regions: Dict[str, Any] = field(default_factory=dict)
#     selectable_regions: Dict[str, Any] = field(default_factory=dict)
#     crop_box: Optional[Dict[str, int]] = None
# 
#     # Schema metadata
#     scitex_schema: str = "scitex.plt.geometry"
#     scitex_schema_version: str = PLOT_GEOMETRY_VERSION
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "schema": {
#                 "name": self.scitex_schema,
#                 "version": self.scitex_schema_version,
#             },
#             "source_hash": self.source_hash,
#             "figure_px": self.figure_px,
#             "dpi": self.dpi,
#             "axes": [ax.to_dict() for ax in self.axes],
#             "artists": [ar.to_dict() for ar in self.artists],
#             "hit_regions": self.hit_regions,
#             "selectable_regions": self.selectable_regions,
#             "crop_box": self.crop_box,
#         }
# 
#     def to_json(self, indent: int = 2) -> str:
#         return json.dumps(self.to_dict(), indent=indent)
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "PlotGeometry":
#         return cls(
#             source_hash=data.get("source_hash", ""),
#             figure_px=data.get("figure_px", [944, 803]),
#             dpi=data.get("dpi", DPI_FALLBACK),
#             axes=[RenderedAxes.from_dict(ax) for ax in data.get("axes", [])],
#             artists=[RenderedArtist.from_dict(ar) for ar in data.get("artists", [])],
#             hit_regions=data.get("hit_regions", {}),
#             selectable_regions=data.get("selectable_regions", {}),
#             crop_box=data.get("crop_box"),
#         )
# 
#     @classmethod
#     def from_json(cls, json_str: str) -> "PlotGeometry":
#         return cls.from_dict(json.loads(json_str))
# 
# 
# # =============================================================================
# # RenderManifest - Render Configuration and Metadata
# # =============================================================================
# 
# 
# @dataclass
# class RenderManifest:
#     """
#     Manifest for rendered outputs.
# 
#     Stored in cache/render_manifest.json.
#     Contains metadata about how the render was produced.
#     """
#     source_hash: str  # Hash of spec + style
#     panel_size_mm: List[float]  # [width, height]
# 
#     # Output files
#     overview_png: Optional[str] = None
#     overview_svg: Optional[str] = None
#     hitmap_png: Optional[str] = None
#     hitmap_svg: Optional[str] = None
# 
#     # Render settings
#     dpi: int = DPI_FALLBACK  # Use scitex.plt.styles.get_default_dpi() for dynamic resolution
#     render_px: Optional[List[int]] = None  # [width, height]
#     crop_margin_mm: float = 1.0
# 
#     # Timestamps
#     rendered_at: Optional[str] = None
# 
#     # Schema metadata
#     scitex_schema: str = "scitex.plt.render_manifest"
#     scitex_schema_version: str = "1.0.0"
# 
#     def __post_init__(self):
#         if self.rendered_at is None:
#             self.rendered_at = datetime.now().isoformat()
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "schema": {
#                 "name": self.scitex_schema,
#                 "version": self.scitex_schema_version,
#             },
#             "source_hash": self.source_hash,
#             "panel_size_mm": self.panel_size_mm,
#             "overview_png": self.overview_png,
#             "overview_svg": self.overview_svg,
#             "hitmap_png": self.hitmap_png,
#             "hitmap_svg": self.hitmap_svg,
#             "dpi": self.dpi,
#             "render_px": self.render_px,
#             "crop_margin_mm": self.crop_margin_mm,
#             "rendered_at": self.rendered_at,
#         }
# 
#     def to_json(self, indent: int = 2) -> str:
#         return json.dumps(self.to_dict(), indent=indent)
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "RenderManifest":
#         return cls(**{k: v for k, v in data.items()
#                      if k in cls.__dataclass_fields__ and k != "scitex_schema" and k != "scitex_schema_version"})
# 
#     @classmethod
#     def from_json(cls, json_str: str) -> "RenderManifest":
#         return cls.from_dict(json.loads(json_str))
# 
# 
# # =============================================================================
# # Public API
# # =============================================================================
# 
# __all__ = [
#     # Version constants
#     "PLOT_SPEC_VERSION",
#     "PLOT_STYLE_VERSION",
#     "PLOT_GEOMETRY_VERSION",
#     # Type aliases
#     "TraceType",
#     "CoordinateSpace",
#     "LegendLocation",
#     # Bbox classes
#     "BboxRatio",
#     "BboxPx",
#     # Spec classes (canonical)
#     "TraceSpec",
#     "AxesLimits",
#     "AxesLabels",
#     "AxesSpecItem",
#     "DataSourceSpec",
#     "PlotSpec",
#     # Style classes
#     "TraceStyleSpec",
#     "ThemeSpec",
#     "FontSpec",
#     "SizeSpec",
#     "LegendSpec",
#     "PlotStyle",
#     # Geometry classes (cache)
#     "RenderedArtist",
#     "RenderedAxes",
#     "HitRegionEntry",
#     "SelectableRegion",
#     "PlotGeometry",
#     # Manifest
#     "RenderManifest",
# ]
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/schema/_plot.py
# --------------------------------------------------------------------------------
