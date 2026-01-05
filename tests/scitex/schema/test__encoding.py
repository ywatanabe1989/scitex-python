# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/schema/_encoding.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-19
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/schema/_encoding.py
# 
# """
# Encoding Schema - Data to Visual Mapping for Scientific Rigor.
# 
# This module defines the encoding layer that maps data columns to visual channels.
# This separation ensures scientific reproducibility by explicitly documenting
# how data is transformed into visual representation.
# 
# Encoding (encoding.json) - Data→Visual Mapping:
#   - Column bindings (x, y, color, size, shape)
#   - Data transformations (log, normalize, bin)
#   - Scale types (linear, log, categorical)
#   - Missing value handling
# """
# 
# import json
# from dataclasses import dataclass, field
# from typing import Any, Dict, List, Optional
# 
# ENCODING_VERSION = "1.0.0"
# 
# 
# @dataclass
# class ChannelBinding:
#     """
#     Binding from data column to visual channel.
# 
#     Parameters
#     ----------
#     channel : str
#         Visual channel (x, y, color, size, shape, opacity, etc.)
#     column : str
#         Data column name
#     scale : str
#         Scale type (linear, log, sqrt, categorical, ordinal)
#     domain : list, optional
#         Data domain [min, max] or list of categories
#     range : list, optional
#         Visual range (e.g., [0, 1] for opacity, color palette for color)
#     transform : str, optional
#         Data transformation (log, sqrt, normalize, zscore, rank)
#     """
# 
#     channel: str
#     column: str
#     scale: str = "linear"
#     domain: Optional[List[Any]] = None
#     range: Optional[List[Any]] = None
#     transform: Optional[str] = None
#     missing_value: Optional[str] = "drop"  # "drop", "zero", "mean", "interpolate"
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {
#             "channel": self.channel,
#             "column": self.column,
#             "scale": self.scale,
#         }
#         if self.domain is not None:
#             result["domain"] = self.domain
#         if self.range is not None:
#             result["range"] = self.range
#         if self.transform:
#             result["transform"] = self.transform
#         if self.missing_value != "drop":
#             result["missing_value"] = self.missing_value
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "ChannelBinding":
#         return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
# 
# 
# @dataclass
# class TraceEncoding:
#     """
#     Encoding specification for a single trace.
# 
#     Parameters
#     ----------
#     trace_id : str
#         ID of the trace this encoding applies to
#     bindings : list of ChannelBinding
#         Column to channel bindings
#     aggregate : str, optional
#         Aggregation method (mean, median, sum, count, etc.)
#     group_by : list, optional
#         Columns to group by before aggregation
#     """
# 
#     trace_id: str
#     bindings: List[ChannelBinding] = field(default_factory=list)
#     aggregate: Optional[str] = None
#     group_by: Optional[List[str]] = None
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {
#             "trace_id": self.trace_id,
#             "bindings": [b.to_dict() for b in self.bindings],
#         }
#         if self.aggregate:
#             result["aggregate"] = self.aggregate
#         if self.group_by:
#             result["group_by"] = self.group_by
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "TraceEncoding":
#         bindings = [ChannelBinding.from_dict(b) for b in data.get("bindings", [])]
#         return cls(
#             trace_id=data.get("trace_id", ""),
#             bindings=bindings,
#             aggregate=data.get("aggregate"),
#             group_by=data.get("group_by"),
#         )
# 
# 
# @dataclass
# class PlotEncoding:
#     """
#     Complete encoding specification for a plot.
# 
#     Stored in encoding.json. Documents how data maps to visual channels
#     for scientific reproducibility.
# 
#     Parameters
#     ----------
#     traces : list of TraceEncoding
#         Per-trace encoding specifications
#     data_hash : str, optional
#         Hash of source data for integrity verification
#     """
# 
#     traces: List[TraceEncoding] = field(default_factory=list)
#     data_hash: Optional[str] = None
# 
#     # Schema metadata
#     scitex_schema: str = "scitex.plt.encoding"
#     scitex_schema_version: str = ENCODING_VERSION
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {
#             "schema": {
#                 "name": self.scitex_schema,
#                 "version": self.scitex_schema_version,
#             },
#             "traces": [t.to_dict() for t in self.traces],
#         }
#         if self.data_hash:
#             result["data_hash"] = self.data_hash
#         return result
# 
#     def to_json(self, indent: int = 2) -> str:
#         return json.dumps(self.to_dict(), indent=indent)
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "PlotEncoding":
#         return cls(
#             traces=[TraceEncoding.from_dict(t) for t in data.get("traces", [])],
#             data_hash=data.get("data_hash"),
#         )
# 
#     @classmethod
#     def from_json(cls, json_str: str) -> "PlotEncoding":
#         return cls.from_dict(json.loads(json_str))
# 
#     @classmethod
#     def from_spec_and_style(cls, spec: dict, style: dict) -> "PlotEncoding":
#         """
#         Create encoding from spec and style for backward compatibility.
# 
#         Extracts column bindings from PlotSpec traces and color/size mappings
#         from PlotStyle trace overrides.
#         """
#         traces = []
#         spec_traces = spec.get("traces", [])
#         style_traces = {t.get("trace_id"): t for t in style.get("traces", [])}
# 
#         for trace in spec_traces:
#             trace_id = trace.get("id", "")
#             bindings = []
# 
#             # X column
#             if trace.get("x_col"):
#                 bindings.append(
#                     ChannelBinding(
#                         channel="x",
#                         column=trace["x_col"],
#                     )
#                 )
# 
#             # Y column
#             if trace.get("y_col"):
#                 bindings.append(
#                     ChannelBinding(
#                         channel="y",
#                         column=trace["y_col"],
#                     )
#                 )
# 
#             # Data columns (for boxplot, etc.)
#             for i, col in enumerate(trace.get("data_cols", [])):
#                 bindings.append(
#                     ChannelBinding(
#                         channel=f"data_{i}",
#                         column=col,
#                     )
#                 )
# 
#             # Value column (for heatmap)
#             if trace.get("value_col"):
#                 bindings.append(
#                     ChannelBinding(
#                         channel="value",
#                         column=trace["value_col"],
#                     )
#                 )
# 
#             # Vector columns (for quiver)
#             if trace.get("u_col"):
#                 bindings.append(
#                     ChannelBinding(
#                         channel="u",
#                         column=trace["u_col"],
#                     )
#                 )
#             if trace.get("v_col"):
#                 bindings.append(
#                     ChannelBinding(
#                         channel="v",
#                         column=trace["v_col"],
#                     )
#                 )
# 
#             # Style overrides that are data-driven
#             style_trace = style_traces.get(trace_id, {})
#             if style_trace.get("color_by"):
#                 bindings.append(
#                     ChannelBinding(
#                         channel="color",
#                         column=style_trace["color_by"],
#                     )
#                 )
#             if style_trace.get("size_by"):
#                 bindings.append(
#                     ChannelBinding(
#                         channel="size",
#                         column=style_trace["size_by"],
#                     )
#                 )
# 
#             if bindings:
#                 traces.append(
#                     TraceEncoding(
#                         trace_id=trace_id,
#                         bindings=bindings,
#                     )
#                 )
# 
#         return cls(traces=traces)
# 
# 
# __all__ = [
#     "ENCODING_VERSION",
#     "ChannelBinding",
#     "TraceEncoding",
#     "PlotEncoding",
# ]
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/schema/_encoding.py
# --------------------------------------------------------------------------------
