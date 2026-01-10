#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/io/bundle/_dataclasses/test__Spec.py

"""Tests for Spec dataclass."""

import json
from datetime import datetime

import pytest


class TestSpecCreation:
    """Test Spec instantiation."""

    def test_create_minimal_spec(self):
        """Test creating spec with minimal required fields."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="test-1", kind="plot")

        assert spec.id == "test-1"
        assert spec.kind == "plot"
        assert spec.bbox_norm is not None
        assert spec.created_at is not None
        assert spec.modified_at is not None

    def test_create_plot_spec(self):
        """Test creating a plot-type spec."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="plot-1", kind="plot", name="Line Plot")

        assert spec.kind == "plot"
        assert spec.name == "Line Plot"

    def test_create_table_spec(self):
        """Test creating a table-type spec."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="table-1", kind="table", name="Demographics Table")

        assert spec.kind == "table"
        assert spec.name == "Demographics Table"

    def test_create_figure_spec(self):
        """Test creating a figure-type spec (container)."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(
            id="fig-1",
            kind="figure",
            name="Multi-panel Figure",
            children=["panel-a", "panel-b"],
        )

        assert spec.kind == "figure"
        assert len(spec.children) == 2
        assert "panel-a" in spec.children

    def test_create_with_size_mm(self):
        """Test creating spec with size specification."""
        from scitex.io.bundle._dataclasses import SizeMM, Spec

        size = SizeMM(width=80, height=60)
        spec = Spec(id="sized-1", kind="plot", size_mm=size)

        assert spec.size_mm is not None
        assert spec.size_mm.width == 80
        assert spec.size_mm.height == 60

    def test_timestamp_auto_generation(self):
        """Test that timestamps are auto-generated."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="ts-test", kind="plot")

        assert spec.created_at is not None
        assert spec.modified_at is not None
        assert spec.created_at == spec.modified_at


class TestSpecSerialization:
    """Test Spec to_dict and from_dict."""

    def test_to_dict_minimal(self):
        """Test to_dict with minimal spec."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="test-1", kind="plot")
        d = spec.to_dict()

        assert d["id"] == "test-1"
        assert d["kind"] == "plot"
        assert "bbox_norm" in d
        assert "created_at" in d
        assert "modified_at" in d

    def test_to_dict_with_name(self):
        """Test to_dict includes name when set."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="test-1", kind="table", name="Test Table")
        d = spec.to_dict()

        assert d["name"] == "Test Table"

    def test_to_dict_with_children(self):
        """Test to_dict includes children."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="fig-1", kind="figure", children=["a", "b", "c"])
        d = spec.to_dict()

        assert d["children"] == ["a", "b", "c"]

    def test_to_dict_with_size_mm(self):
        """Test to_dict includes size_mm."""
        from scitex.io.bundle._dataclasses import SizeMM, Spec

        spec = Spec(id="test-1", kind="plot", size_mm=SizeMM(width=100, height=80))
        d = spec.to_dict()

        assert "size_mm" in d
        assert d["size_mm"]["width"] == 100
        assert d["size_mm"]["height"] == 80

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        from scitex.io.bundle._dataclasses import Spec

        data = {"id": "from-dict-1", "kind": "plot"}
        spec = Spec.from_dict(data)

        assert spec.id == "from-dict-1"
        assert spec.kind == "plot"

    def test_from_dict_with_all_fields(self):
        """Test from_dict with all fields."""
        from scitex.io.bundle._dataclasses import Spec

        data = {
            "id": "full-spec",
            "kind": "figure",
            "name": "Full Table",
            "bbox_norm": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8},
            "size_mm": {"width": 170, "height": 120},
            "children": ["child-1", "child-2"],
            "refs": {"data": "data.csv", "stats": "stats.json"},
            "created_at": "2025-12-20T00:00:00Z",
            "modified_at": "2025-12-20T01:00:00Z",
        }
        spec = Spec.from_dict(data)

        assert spec.id == "full-spec"
        assert spec.kind == "figure"
        assert spec.name == "Full Table"
        assert spec.children == ["child-1", "child-2"]
        assert spec.created_at == "2025-12-20T00:00:00Z"

    def test_roundtrip_serialization(self):
        """Test that to_dict -> from_dict preserves data."""
        from scitex.io.bundle._dataclasses import SizeMM, Spec

        original = Spec(
            id="roundtrip-1",
            kind="figure",
            name="Roundtrip Test",
            size_mm=SizeMM(width=80, height=60),
            children=["a", "b"],
        )

        d = original.to_dict()
        restored = Spec.from_dict(d)

        assert restored.id == original.id
        assert restored.kind == original.kind
        assert restored.name == original.name
        assert restored.children == original.children


class TestSpecTouch:
    """Test Spec touch method."""

    def test_touch_updates_modified(self):
        """Test that touch updates modified_at."""
        import time

        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="touch-test", kind="plot")
        original_modified = spec.modified_at

        time.sleep(0.01)  # Small delay to ensure different timestamp
        spec.touch()

        assert spec.modified_at != original_modified
        assert spec.created_at != spec.modified_at


class TestSpecKinds:
    """Test various spec kinds."""

    def test_all_valid_kinds(self):
        """Test creating specs with all valid kinds."""
        from scitex.io.bundle._dataclasses import Spec

        valid_kinds = [
            "figure",
            "plot",
            "table",
            "stats",
            "image",
            "text",
            "shape",
        ]

        for spec_kind in valid_kinds:
            spec = Spec(id=f"{spec_kind}-test", kind=spec_kind)
            assert spec.kind == spec_kind

    def test_table_kind_integration(self):
        """Test table kind works with full workflow."""
        from scitex.io.bundle._dataclasses import Spec

        # Create table spec
        table_spec = Spec(
            id="demographics",
            kind="table",
            name="Subject Demographics",
        )

        # Serialize
        d = table_spec.to_dict()
        assert d["kind"] == "table"

        # Deserialize
        restored = Spec.from_dict(d)
        assert restored.kind == "table"
        assert restored.name == "Subject Demographics"


class TestSpecBBox:
    """Test Spec bounding box handling."""

    def test_default_bbox(self):
        """Test default bounding box values."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="bbox-test", kind="plot")

        assert spec.bbox_norm is not None
        # BBox uses x0, y0, x1, y1 format
        assert hasattr(spec.bbox_norm, "x0")
        assert hasattr(spec.bbox_norm, "y0")
        assert hasattr(spec.bbox_norm, "x1")
        assert hasattr(spec.bbox_norm, "y1")

    def test_custom_bbox(self):
        """Test custom bounding box."""
        from scitex.io.bundle._dataclasses import BBox, Spec

        # BBox uses x0, y0, x1, y1 format
        bbox = BBox(x0=0.1, y0=0.2, x1=0.7, y1=0.7)
        spec = Spec(id="bbox-test", kind="plot", bbox_norm=bbox)

        assert spec.bbox_norm.x0 == 0.1
        assert spec.bbox_norm.y0 == 0.2
        assert spec.bbox_norm.x1 == 0.7
        assert spec.bbox_norm.y1 == 0.7


class TestSpecRefs:
    """Test Spec refs handling."""

    def test_default_refs(self):
        """Test default refs."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="refs-test", kind="plot")

        assert spec.refs is not None

    def test_refs_from_dict(self):
        """Test refs from dictionary."""
        from scitex.io.bundle._dataclasses import Spec

        data = {
            "id": "refs-test",
            "kind": "table",
            "refs": {
                "data": "data/table1.csv",
                "stats": "stats/table1_stats.json",
            },
        }
        spec = Spec.from_dict(data)

        # Refs should be parsed
        assert spec.refs is not None

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/bundle/_dataclasses/_Spec.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-21
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/bundle/_dataclasses/_Spec.py
# 
# """Spec - Core bundle specification model with kind-based constraints."""
# 
# from dataclasses import dataclass, field
# from datetime import datetime
# from typing import Any, ClassVar, Dict, List, Optional, Set
# 
# from ._Axes import Axes
# from ._BBox import BBox
# from ._SizeMM import SizeMM
# from ._SpecRefs import SpecRefs
# 
# 
# @dataclass
# class TextContent:
#     """Text content for kind=text specs."""
# 
#     content: str = ""
#     fontsize: Optional[float] = None
#     fontweight: Optional[str] = None  # "normal" | "bold"
#     ha: str = "center"  # "left" | "center" | "right"
#     va: str = "center"  # "top" | "center" | "bottom"
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {"content": self.content, "ha": self.ha, "va": self.va}
#         if self.fontsize is not None:
#             result["fontsize"] = self.fontsize
#         if self.fontweight is not None:
#             result["fontweight"] = self.fontweight
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "TextContent":
#         return cls(
#             content=data.get("content", ""),
#             fontsize=data.get("fontsize"),
#             fontweight=data.get("fontweight"),
#             ha=data.get("ha", "center"),
#             va=data.get("va", "center"),
#         )
# 
# 
# @dataclass
# class ShapeParams:
#     """Shape parameters for kind=shape specs."""
# 
#     shape_type: str = "rectangle"  # "rectangle" | "ellipse" | "arrow" | "line"
#     color: str = "#000000"
#     linewidth: float = 1.0
#     fill: bool = False
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "shape_type": self.shape_type,
#             "color": self.color,
#             "linewidth": self.linewidth,
#             "fill": self.fill,
#         }
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "ShapeParams":
#         return cls(
#             shape_type=data.get("shape_type", "rectangle"),
#             color=data.get("color", "#000000"),
#             linewidth=data.get("linewidth", 1.0),
#             fill=data.get("fill", False),
#         )
# 
# 
# @dataclass
# class Spec:
#     """Core bundle specification model with kind-based constraints.
# 
#     The central structural element of a bundle.
#     Stored in canonical/spec.json.
# 
#     Kind categories:
#     - Data leaf kinds (plot, table, stats): require payload data files
#     - Annotation leaf kinds (text, shape): no payload required, params in spec
#     - Image leaf kinds (image): require payload image file
#     - Composite kinds (figure): contain children, no payload
# 
#     All bundles have IDENTICAL directory structure:
#     - canonical/: Source of truth (spec.json, encoding.json, theme.json, etc.)
#     - payload/: ALWAYS exists (empty for composites/annotations, populated for data/image)
#     - artifacts/: Derived files (exports/, cache/)
#     - children/: ALWAYS exists (empty for leaves, populated for composites)
#     """
# 
#     # Required fields
#     id: str
#     kind: str  # "figure" | "plot" | "table" | "stats" | "text" | "shape" | "image"
# 
#     # Schema versioning for forward compatibility
#     scitex_schema: str = "scitex.io.bundle.spec"
#     scitex_schema_version: str = "1.0.0"
# 
#     # Children and layout (for composite kinds)
#     children: List[str] = field(default_factory=list)
#     layout: Optional[Dict] = None
# 
#     # Payload schema (for data leaf kinds)
#     payload_schema: Optional[str] = None
# 
#     # Visual properties
#     bbox_norm: BBox = field(default_factory=BBox)
#     name: Optional[str] = None
#     size_mm: Optional[SizeMM] = None
#     axes: Optional[Axes] = None
# 
#     # Kind-specific content
#     text: Optional[TextContent] = None  # For kind=text
#     shape: Optional[ShapeParams] = None  # For kind=shape
# 
#     # References and timestamps
#     refs: SpecRefs = field(default_factory=SpecRefs)
#     created_at: Optional[str] = None
#     modified_at: Optional[str] = None
# 
#     # === Kind Constants ===
#     # Data leaf kinds: require payload data files, forbid children
#     DATA_LEAF_KINDS: ClassVar[Set[str]] = {"plot", "table", "stats"}
#     # Annotation leaf kinds: no payload required, forbid children
#     ANNOTATION_LEAF_KINDS: ClassVar[Set[str]] = {"text", "shape"}
#     # Image leaf kinds: require payload image file, forbid children
#     IMAGE_LEAF_KINDS: ClassVar[Set[str]] = {"image"}
#     # All leaf kinds (for convenience)
#     LEAF_KINDS: ClassVar[Set[str]] = (
#         DATA_LEAF_KINDS | ANNOTATION_LEAF_KINDS | IMAGE_LEAF_KINDS
#     )
#     # Composite kinds: allow children, forbid payload
#     COMPOSITE_KINDS: ClassVar[Set[str]] = {"figure"}
#     # All valid kinds
#     ALL_KINDS: ClassVar[Set[str]] = LEAF_KINDS | COMPOSITE_KINDS
# 
#     # Payload schema -> required file mapping
#     PAYLOAD_REQUIRED_FILES: ClassVar[Dict[str, str]] = {
#         "scitex.io.bundle.payload.plot@1": "payload/data.csv",
#         "scitex.io.bundle.payload.table@1": "payload/table.csv",
#         "scitex.io.bundle.payload.stats@1": "payload/stats.json",
#         "scitex.io.bundle.payload.image@1": "payload/image.png",
#     }
# 
#     def __post_init__(self):
#         if self.created_at is None:
#             self.created_at = datetime.utcnow().isoformat() + "Z"
#         if self.modified_at is None:
#             self.modified_at = self.created_at
# 
#     # === Kind Methods ===
# 
#     def is_leaf_kind(self) -> bool:
#         """Check if this is any leaf kind (forbids children)."""
#         return self.kind in self.LEAF_KINDS
# 
#     def is_data_leaf_kind(self) -> bool:
#         """Check if this is a data leaf kind (requires payload data)."""
#         return self.kind in self.DATA_LEAF_KINDS
# 
#     def is_annotation_leaf_kind(self) -> bool:
#         """Check if this is an annotation leaf kind (no payload required)."""
#         return self.kind in self.ANNOTATION_LEAF_KINDS
# 
#     def is_image_leaf_kind(self) -> bool:
#         """Check if this is an image leaf kind (requires payload image)."""
#         return self.kind in self.IMAGE_LEAF_KINDS
# 
#     def is_composite_kind(self) -> bool:
#         """Check if this is a composite kind (allows children, forbids payload)."""
#         return self.kind in self.COMPOSITE_KINDS
# 
#     def get_required_payload_file(self) -> Optional[str]:
#         """Get required payload file path based on payload_schema."""
#         return self.PAYLOAD_REQUIRED_FILES.get(self.payload_schema)
# 
#     # === Validation ===
# 
#     def validate(self) -> List[str]:
#         """Validate logical constraints (not file existence).
# 
#         Returns:
#             List of error messages (empty if valid)
#         """
#         errors = []
# 
#         # Check: kind is valid
#         if self.kind not in self.ALL_KINDS:
#             errors.append(
#                 f"Unknown kind: {self.kind}. Valid kinds: {sorted(self.ALL_KINDS)}"
#             )
#             return errors  # Early return - other checks don't make sense
# 
#         # Check: children list has no duplicates
#         if len(self.children) != len(set(self.children)):
#             errors.append("children list has duplicates")
# 
#         if self.is_leaf_kind():
#             # All leaf kinds: children must be empty
#             if self.children:
#                 errors.append(f"kind={self.kind} cannot have children")
# 
#             # Data leaf kinds: payload_schema is optional but recommended
#             # Annotation leaf kinds: should not have payload_schema
#             if self.is_annotation_leaf_kind() and self.payload_schema:
#                 errors.append(f"kind={self.kind} should not have payload_schema")
# 
#         elif self.is_composite_kind():
#             # Composite kinds: payload_schema must be None
#             if self.payload_schema:
#                 errors.append(f"kind={self.kind} should not have payload_schema")
# 
#             # Validate layout if present
#             if self.layout:
#                 panels = self.layout.get("panels", [])
#                 panel_children = [p.get("child") for p in panels]
# 
#                 # Check: panel child references must be subset of children
#                 for child_ref in panel_children:
#                     if child_ref not in self.children:
#                         errors.append(
#                             f"layout.panels references unknown child: {child_ref}"
#                         )
# 
#                 # Check: no duplicate panel child references
#                 if len(panel_children) != len(set(panel_children)):
#                     errors.append("layout.panels has duplicate child references")
# 
#         return errors
# 
#     # === Serialization ===
# 
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary for JSON serialization."""
#         result = {
#             "id": self.id,
#             "kind": self.kind,
#             "scitex_schema": self.scitex_schema,
#             "scitex_schema_version": self.scitex_schema_version,
#             "bbox_norm": self.bbox_norm.to_dict(),
#         }
# 
#         if self.name:
#             result["name"] = self.name
#         if self.size_mm:
#             result["size_mm"] = self.size_mm.to_dict()
#         if self.axes:
#             result["axes"] = self.axes.to_dict()
#         if self.children:
#             result["children"] = self.children
#         if self.layout:
#             result["layout"] = self.layout
#         if self.payload_schema:
#             result["payload_schema"] = self.payload_schema
#         if self.text:
#             result["text"] = self.text.to_dict()
#         if self.shape:
#             result["shape"] = self.shape.to_dict()
# 
#         result["refs"] = self.refs.to_dict()
#         result["created_at"] = self.created_at
#         result["modified_at"] = self.modified_at
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "Spec":
#         """Create Spec from dictionary."""
#         # Handle legacy 'type' field
#         kind = data.get("kind") or data.get("type", "plot")
# 
#         # Handle payload_schema
#         payload_schema = data.get("payload_schema")
# 
#         # Handle size_mm from different formats
#         size_mm_data = data.get("size_mm")
#         if size_mm_data is None and "size" in data:
#             size = data["size"]
#             if isinstance(size, dict):
#                 size_mm_data = {
#                     "width": size.get("width_mm", size.get("width", 85)),
#                     "height": size.get("height_mm", size.get("height", 85)),
#                 }
# 
#         # Handle name from 'title' field (legacy format)
#         name = data.get("name") or data.get("title")
# 
#         # Handle text content
#         text = None
#         if "text" in data and isinstance(data["text"], dict):
#             text = TextContent.from_dict(data["text"])
# 
#         # Handle shape params
#         shape = None
#         if "shape" in data and isinstance(data["shape"], dict):
#             shape = ShapeParams.from_dict(data["shape"])
# 
#         return cls(
#             id=data.get("id", "unknown"),
#             kind=kind,
#             scitex_schema=data.get("scitex_schema", "scitex.io.bundle.spec"),
#             scitex_schema_version=data.get("scitex_schema_version", "1.0.0"),
#             children=data.get("children", []),
#             layout=data.get("layout"),
#             payload_schema=payload_schema,
#             bbox_norm=BBox.from_dict(data.get("bbox_norm", {})),
#             name=name,
#             size_mm=SizeMM.from_dict(size_mm_data) if size_mm_data else None,
#             axes=Axes.from_dict(data["axes"]) if "axes" in data else None,
#             text=text,
#             shape=shape,
#             refs=SpecRefs.from_dict(data.get("refs", {})),
#             created_at=data.get("created_at"),
#             modified_at=data.get("modified_at"),
#         )
# 
#     def touch(self):
#         """Update modified timestamp."""
#         self.modified_at = datetime.utcnow().isoformat() + "Z"
# 
# 
# __all__ = ["Spec", "TextContent", "ShapeParams"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/bundle/_dataclasses/_Spec.py
# --------------------------------------------------------------------------------
