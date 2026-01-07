# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_FTS.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_FTS.py
# 
# """FTS Bundle Class - Main entry point for FTS bundles.
# 
# Structure (identical for all kinds):
# - canonical/: Source of truth (spec.json, encoding.json, theme.json)
# - payload/: Data files (empty for composites)
# - artifacts/: Exports and cache
# - children/: Embedded child bundles (empty for leaves)
# """
# 
# import uuid
# from pathlib import Path
# from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
# 
# from ._children import ValidationError, embed_child, load_embedded_children
# from ._dataclasses import DataInfo, Node, SizeMM
# from ._loader import load_bundle_components
# from ._saver import compute_canonical_hash, compute_theme_hash, save_bundle_components, save_render_outputs
# from ._storage import Storage, get_storage
# from .._fig import Encoding, Theme
# from .._stats import Stats
# 
# if TYPE_CHECKING:
#     from matplotlib.figure import Figure as MplFigure
# 
# 
# class FTS:
#     """Figure-Table-Statistics Bundle - Self-contained figure/plot/stats package.
# 
#     Attributes:
#         node: Node metadata (kind, children, layout, payload_schema, etc.)
#         encoding: Encoding specification (traces, channels)
#         theme: Theme specification (colors, fonts)
#         stats: Statistics (for kind=stats)
#         data_info: Data info metadata
#     """
# 
#     def __init__(
#         self,
#         path: Union[str, Path],
#         create: bool = False,
#         kind: str = "plot",
#         name: Optional[str] = None,
#         size_mm: Optional[Dict[str, float]] = None,
#         # Legacy support
#         node_type: Optional[str] = None,
#     ):
#         """Initialize FTS bundle.
# 
#         Args:
#             path: Bundle path (directory or .zip file)
#             create: If True, create new bundle; if False, load existing
#             kind: Bundle kind (plot, figure, table, stats, group, collection)
#             name: Bundle name (default: stem of path)
#             size_mm: Figure size in mm (e.g., {"width": 170, "height": 85})
#             node_type: DEPRECATED - use 'kind' instead
#         """
#         self._path = Path(path)
#         self._is_zip = self._path.suffix == ".zip"
#         self._node: Optional[Node] = None
#         self._encoding: Optional[Encoding] = None
#         self._theme: Optional[Theme] = None
#         self._stats: Optional[Stats] = None
#         self._data_info: Optional[DataInfo] = None
#         self._dirty = False
#         self._storage: Optional[Storage] = None
# 
#         # Handle legacy node_type parameter
#         if node_type is not None:
#             kind = node_type
# 
#         if create:
#             self._create_new(kind, name, size_mm)
#         else:
#             self._load()
# 
#     @property
#     def path(self) -> Path:
#         """Bundle path (directory or ZIP)."""
#         return self._path
# 
#     @property
#     def is_zip(self) -> bool:
#         """Whether bundle is a ZIP file."""
#         return self._is_zip
# 
#     @property
#     def bundle_type(self) -> str:
#         """Bundle kind (figure, plot, table, etc.)."""
#         return self._node.kind if self._node else "unknown"
# 
#     @property
#     def is_dirty(self) -> bool:
#         """Whether bundle has unsaved changes."""
#         return self._dirty
# 
#     @property
#     def storage(self) -> Storage:
#         """Get storage for this bundle."""
#         if self._storage is None:
#             self._storage = get_storage(self._path)
#         return self._storage
# 
#     @property
#     def node(self) -> Optional[Node]:
#         """Node metadata."""
#         return self._node
# 
#     @node.setter
#     def node(self, value: Union[Node, Dict[str, Any]]):
#         if isinstance(value, dict):
#             self._node = Node.from_dict(value)
#         else:
#             self._node = value
#         self._dirty = True
# 
#     @property
#     def encoding(self) -> Optional[Encoding]:
#         """Encoding specification (typed object)."""
#         return self._encoding
# 
#     @encoding.setter
#     def encoding(self, value: Union[Encoding, Dict[str, Any]]):
#         if isinstance(value, dict):
#             self._encoding = Encoding.from_dict(value)
#         else:
#             self._encoding = value
#         self._dirty = True
# 
#     @property
#     def encoding_dict(self) -> Optional[Dict[str, Any]]:
#         """Encoding as dictionary (for serialization)."""
#         return self._encoding.to_dict() if self._encoding else None
# 
#     @property
#     def theme(self) -> Optional[Theme]:
#         """Theme specification (typed object)."""
#         return self._theme
# 
#     @theme.setter
#     def theme(self, value: Union[Theme, Dict[str, Any]]):
#         if isinstance(value, dict):
#             self._theme = Theme.from_dict(value)
#         else:
#             self._theme = value
#         self._dirty = True
# 
#     @property
#     def theme_dict(self) -> Optional[Dict[str, Any]]:
#         """Theme as dictionary (for serialization)."""
#         return self._theme.to_dict() if self._theme else None
# 
#     @property
#     def stats(self) -> Optional[Stats]:
#         """Statistics."""
#         return self._stats
# 
#     @stats.setter
#     def stats(self, value: Union[Stats, Dict[str, Any]]):
#         if isinstance(value, dict):
#             self._stats = Stats.from_dict(value)
#         else:
#             self._stats = value
#         self._dirty = True
# 
#     @property
#     def data_info(self) -> Optional[DataInfo]:
#         """Data info metadata."""
#         return self._data_info
# 
#     @data_info.setter
#     def data_info(self, value: Union[DataInfo, Dict[str, Any]]):
#         if isinstance(value, dict):
#             self._data_info = DataInfo.from_dict(value)
#         else:
#             self._data_info = value
#         self._dirty = True
# 
#     def _create_new(
#         self,
#         kind: str,
#         name: Optional[str],
#         size_mm: Optional[Dict[str, float]],
#     ):
#         """Create a new bundle."""
#         bundle_id = str(uuid.uuid4())
#         if name is None:
#             name = self._path.stem
# 
#         # Determine payload_schema for leaf kinds
#         # Note: payload_schema is optional. For plots without data, it's None.
#         # For plots with data, from_matplotlib will set it.
#         payload_schema = None
#         if kind in Node.LEAF_KINDS and kind != "plot":
#             # Only auto-set for non-plot leaf kinds
#             payload_schema_map = {
#                 "table": "scitex.fts.payload.table@1",
#                 "stats": "scitex.fts.payload.stats@1",
#             }
#             payload_schema = payload_schema_map.get(kind)
# 
#         self._node = Node(
#             id=bundle_id,
#             kind=kind,
#             name=name,
#             size_mm=SizeMM.from_dict(size_mm) if size_mm else None,
#             payload_schema=payload_schema,
#         )
#         self._encoding = Encoding()
#         self._theme = Theme()
#         self._stats = Stats()
#         self._dirty = True
# 
#     def _load(self):
#         """Load existing bundle."""
#         if not self._path.exists():
#             raise FileNotFoundError(f"FTS bundle not found: {self._path}")
# 
#         (
#             self._node,
#             self._encoding,
#             self._theme,
#             self._stats,
#             self._data_info,
#         ) = load_bundle_components(self._path)
# 
#     def add_child(
#         self,
#         child: Union[str, Path, "FTS"],
#         row: int = 0,
#         col: int = 0,
#         label: Optional[str] = None,
#         row_span: int = 1,
#         col_span: int = 1,
#         **kwargs,
#     ) -> str:
#         """Add and embed a child bundle. Returns child_name in children/."""
#         if not self.node.is_composite_kind():
#             raise TypeError(f"kind={self.node.kind} cannot have children")
# 
#         # Get child path
#         if isinstance(child, FTS):
#             child_path = child.path
#         else:
#             child_path = Path(child)
# 
#         # Embed child into children/ directory
#         # Returns (child_name, child_id) tuple
#         child_name, child_id = embed_child(self.storage, child_path)
# 
#         # Add to node.children
#         self._node.children.append(child_name)
# 
#         # Initialize layout if needed
#         if self._node.layout is None:
#             self._node.layout = {"rows": 2, "cols": 2, "panels": []}
# 
#         # Update grid size if needed
#         self._node.layout["rows"] = max(self._node.layout.get("rows", 1), row + row_span)
#         self._node.layout["cols"] = max(self._node.layout.get("cols", 1), col + col_span)
# 
#         # Add to layout.panels
#         panel_info = {
#             "child": child_name,
#             "child_id": child_id,  # Full UUID for identity tracking
#             "row": row,
#             "col": col,
#             "row_span": row_span,
#             "col_span": col_span,
#             **kwargs,
#         }
#         if label:
#             panel_info["label"] = label
# 
#         self._node.layout["panels"].append(panel_info)
#         self._dirty = True
# 
#         return child_name
# 
#     def load_children(self) -> Dict[str, "FTS"]:
#         """Load embedded children. Returns dict: child_name -> FTS."""
#         return load_embedded_children(self._path)
# 
#     def render(self) -> Optional["MplFigure"]:
#         """Render figure. Composite renders children, leaf renders from encoding."""
#         if self._node is None:
#             return None
# 
#         if self._node.is_composite_kind():
#             return self._render_composite()
#         elif self._node.is_leaf_kind():
#             return self._render_from_encoding()
# 
#         return None
# 
#     def _render_composite(self) -> Optional["MplFigure"]:
#         """Render composite figure with children."""
#         if not self._node.children:
#             return None  # Empty container
# 
#         from .._fig._composite import render_composite
# 
#         children = self.load_children()
#         size_mm = self._node.size_mm.to_dict() if self._node.size_mm else None
# 
#         fig, geometry = render_composite(
#             children=children,
#             layout=self._node.layout or {"rows": 1, "cols": 1, "panels": []},
#             size_mm=size_mm,
#             theme=self._theme,
#         )
# 
#         return fig
# 
#     def _render_from_encoding(self) -> Optional["MplFigure"]:
#         """Render leaf figure from encoding + payload."""
#         if self._encoding is None:
#             return None
# 
#         import scitex.plt as splt
# 
#         size_mm = self._node.size_mm.to_dict() if self._node.size_mm else {"width": 85, "height": 85}
# 
#         # Use scitex.plt for proper styling (3-4 ticks, etc.)
#         fig, ax = splt.subplots(
#             figsize_mm=(size_mm.get("width", 85), size_mm.get("height", 85))
#         )
# 
#         # Load data from payload
#         data = self._load_payload_data()
# 
#         # Render traces
#         from .._fig._backend._render import render_traces
# 
#         traces = self._encoding.traces if self._encoding.traces else []
#         for trace in traces:
#             render_traces(ax, trace, data, self._theme)
# 
#         # Apply labels from encoding axes config (if available)
#         # Note: Unit validation happens in scitex.plt via UnitAwareMixin.set_xlabel/set_ylabel
#         if self._encoding.axes:
#             if "x" in self._encoding.axes and self._encoding.axes["x"].title:
#                 ax.set_xlabel(self._encoding.axes["x"].title)
#             if "y" in self._encoding.axes and self._encoding.axes["y"].title:
#                 ax.set_ylabel(self._encoding.axes["y"].title)
# 
#         fig.tight_layout()
#         return fig
# 
#     def _load_payload_data(self) -> Optional["pd.DataFrame"]:
#         """Load data from payload/data.csv or legacy data/data.csv."""
#         import pandas as pd
#         from io import StringIO
# 
#         # Try new path first, then legacy
#         for path in ["payload/data.csv", "data/data.csv"]:
#             if self.storage.exists(path):
#                 csv_bytes = self.storage.read(path)
#                 return pd.read_csv(StringIO(csv_bytes.decode("utf-8")))
#         return None
# 
#     def validate(self, level: str = "schema") -> List[str]:
#         """Validate bundle. Returns list of error messages (empty if valid)."""
#         errors = []
# 
#         # Node logical validation
#         if self._node:
#             errors.extend(self._node.validate())
# 
#         # Storage-level validation - check required payload files
#         if self._node and self._node.is_leaf_kind():
#             required_file = self._node.get_required_payload_file()
#             if required_file:
#                 # Check both new structure (payload/) and legacy structure (data/)
#                 # Legacy sio.save() uses data/data.csv, new FTS uses payload/data.csv
#                 legacy_paths = {
#                     "payload/data.csv": "data/data.csv",
#                     "payload/table.csv": "data/table.csv",
#                     "payload/stats.json": "stats/stats.json",
#                 }
#                 legacy_path = legacy_paths.get(required_file)
#                 if not self.storage.exists(required_file):
#                     if not legacy_path or not self.storage.exists(legacy_path):
#                         errors.append(f"Missing required payload file: {required_file}")
# 
#         # NOTE: For composite kinds, do NOT validate payload/ emptiness by listing files.
#         # Payload prohibition is enforced purely via payload_schema is None (in Node.validate).
# 
#         # Recursively validate embedded children
#         if self._node and self._node.is_composite_kind() and self._node.children:
#             children = self.load_children()
#             for child_name, child in children.items():
#                 child_errors = child.validate(level)
#                 errors.extend([f"{child_name}: {e}" for e in child_errors])
# 
#         # Schema validation for other components
#         if level in ("semantic", "strict"):
#             # Additional semantic validation
#             if self._encoding and self._node:
#                 if self._node.is_composite_kind() and self._encoding.traces:
#                     errors.append("Composite kinds should not have encoding traces")
# 
#         return errors
# 
#     def save(
#         self,
#         path: Optional[Union[str, Path]] = None,
#         validate: bool = True,
#         validation_level: str = "schema",
#         render: bool = True,
#     ):
#         """Save bundle to disk.
# 
#         Args:
#             path: Override save path
#             validate: Run validation before saving
#             validation_level: Validation level
#             render: Generate exports/cache (default True).
#                     Set False for WIP saves (faster, spec/payload/children only).
#         """
#         if path:
#             self._path = Path(path)
#             self._is_zip = self._path.suffix == ".zip"
#             self._storage = None  # Reset storage
# 
#         # Validate before saving
#         if validate:
#             errors = self.validate(level=validation_level)
#             if errors:
#                 raise ValidationError(f"Validation failed: {errors}")
# 
#         # Update modified timestamp
#         if self._node:
#             self._node.touch()
# 
#         # Save canonical files
#         save_bundle_components(
#             self._path,
#             node=self._node,
#             encoding=self._encoding,
#             theme=self._theme,
#             stats=self._stats,
#             data_info=self._data_info,
#             render=render,
#         )
# 
#         # Render and save exports/cache (optional)
#         if render:
#             fig = self.render()
#             if fig:
#                 source_hash = compute_canonical_hash(self.storage)
#                 theme_hash = compute_theme_hash(self._theme)
#                 save_render_outputs(
#                     self.storage,
#                     fig,
#                     geometry={},  # TODO: Generate proper geometry
#                     source_hash=source_hash,
#                     theme_hash=theme_hash,
#                 )
#                 import matplotlib.pyplot as plt
#                 from matplotlib.figure import Figure as MplFigure
# 
#                 # Handle FigWrapper from scitex.plt
#                 if isinstance(fig, MplFigure):
#                     plt.close(fig)
#                 elif hasattr(fig, "figure") and isinstance(fig.figure, MplFigure):
#                     plt.close(fig.figure)
#                 else:
#                     plt.close(fig)
# 
#         self._dirty = False
# 
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert bundle to dictionary."""
#         result = {
#             "path": str(self._path),
#             "is_zip": self._is_zip,
#             "kind": self.bundle_type,
#         }
#         if self._node:
#             result["node"] = self._node.to_dict()
#         if self._encoding:
#             result["encoding"] = self._encoding.to_dict()
#         if self._theme:
#             result["theme"] = self._theme.to_dict()
#         if self._stats:
#             result["stats"] = self._stats.to_dict()
#         if self._data_info:
#             result["data_info"] = self._data_info.to_dict()
#         return result
# 
#     def __enter__(self) -> "FTS":
#         """Enter context manager."""
#         return self
# 
#     def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
#         """Exit context manager, auto-saving if dirty and no exception."""
#         if exc_type is None and self._dirty:
#             self.save()
#         return False
# 
#     def __repr__(self) -> str:
#         dirty_marker = "*" if self._dirty else ""
#         kind = self._node.kind if self._node else "unknown"
#         return f"FTS({self._path!r}, kind={kind!r}){dirty_marker}"
# 
# 
# # =============================================================================
# # Factory Functions
# # =============================================================================
# 
# # Import from_matplotlib from helper module (single source of truth)
# from ._mpl_helpers import from_matplotlib
# 
# 
# def load_bundle(path: Union[str, Path]) -> FTS:
#     """Load an existing FTS bundle."""
#     return FTS(path)
# 
# 
# def create_bundle(
#     path: Union[str, Path],
#     kind: str = "plot",
#     name: Optional[str] = None,
#     size_mm: Optional[Dict[str, float]] = None,
#     # Legacy support
#     node_type: Optional[str] = None,
# ) -> FTS:
#     """Create a new FTS bundle."""
#     if node_type is not None:
#         kind = node_type
#     return FTS(path, create=True, kind=kind, name=name, size_mm=size_mm)
# 
# 
# __all__ = ["FTS", "load_bundle", "create_bundle", "from_matplotlib"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_FTS.py
# --------------------------------------------------------------------------------
