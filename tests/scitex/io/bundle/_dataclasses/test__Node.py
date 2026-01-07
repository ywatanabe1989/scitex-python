#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/fsb/_bundle/_dataclasses/test__Node.py

"""Tests for Node dataclass."""

import json
from datetime import datetime

import pytest


class TestNodeCreation:
    """Test Node instantiation."""

    def test_create_minimal_node(self):
        """Test creating node with minimal required fields."""
        from scitex.io.bundle._bundle._dataclasses import Node

        node = Node(id="test-1", type="plot")

        assert node.id == "test-1"
        assert node.type == "plot"
        assert node.bbox_norm is not None
        assert node.created_at is not None
        assert node.modified_at is not None

    def test_create_plot_node(self):
        """Test creating a plot-type node."""
        from scitex.io.bundle._bundle._dataclasses import Node

        node = Node(id="plot-1", type="plot", name="Line Plot")

        assert node.type == "plot"
        assert node.name == "Line Plot"

    def test_create_table_node(self):
        """Test creating a table-type node."""
        from scitex.io.bundle._bundle._dataclasses import Node

        node = Node(id="table-1", type="table", name="Demographics Table")

        assert node.type == "table"
        assert node.name == "Demographics Table"

    def test_create_figure_node(self):
        """Test creating a figure-type node (container)."""
        from scitex.io.bundle._bundle._dataclasses import Node

        node = Node(
            id="fig-1",
            type="figure",
            name="Multi-panel Figure",
            children=["panel-a", "panel-b"],
        )

        assert node.type == "figure"
        assert len(node.children) == 2
        assert "panel-a" in node.children

    def test_create_with_size_mm(self):
        """Test creating node with size specification."""
        from scitex.io.bundle._bundle._dataclasses import Node, SizeMM

        size = SizeMM(width=80, height=60)
        node = Node(id="sized-1", type="plot", size_mm=size)

        assert node.size_mm is not None
        assert node.size_mm.width == 80
        assert node.size_mm.height == 60

    def test_timestamp_auto_generation(self):
        """Test that timestamps are auto-generated."""
        from scitex.io.bundle._bundle._dataclasses import Node

        node = Node(id="ts-test", type="plot")

        assert node.created_at is not None
        assert node.modified_at is not None
        assert node.created_at == node.modified_at


class TestNodeSerialization:
    """Test Node to_dict and from_dict."""

    def test_to_dict_minimal(self):
        """Test to_dict with minimal node."""
        from scitex.io.bundle._bundle._dataclasses import Node

        node = Node(id="test-1", type="plot")
        d = node.to_dict()

        assert d["id"] == "test-1"
        assert d["type"] == "plot"
        assert "bbox_norm" in d
        assert "created_at" in d
        assert "modified_at" in d

    def test_to_dict_with_name(self):
        """Test to_dict includes name when set."""
        from scitex.io.bundle._bundle._dataclasses import Node

        node = Node(id="test-1", type="table", name="Test Table")
        d = node.to_dict()

        assert d["name"] == "Test Table"

    def test_to_dict_with_children(self):
        """Test to_dict includes children."""
        from scitex.io.bundle._bundle._dataclasses import Node

        node = Node(id="fig-1", type="figure", children=["a", "b", "c"])
        d = node.to_dict()

        assert d["children"] == ["a", "b", "c"]

    def test_to_dict_with_size_mm(self):
        """Test to_dict includes size_mm."""
        from scitex.io.bundle._bundle._dataclasses import Node, SizeMM

        node = Node(id="test-1", type="plot", size_mm=SizeMM(width=100, height=80))
        d = node.to_dict()

        assert "size_mm" in d
        assert d["size_mm"]["width"] == 100
        assert d["size_mm"]["height"] == 80

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        from scitex.io.bundle._bundle._dataclasses import Node

        data = {"id": "from-dict-1", "type": "plot"}
        node = Node.from_dict(data)

        assert node.id == "from-dict-1"
        assert node.type == "plot"

    def test_from_dict_with_all_fields(self):
        """Test from_dict with all fields."""
        from scitex.io.bundle._bundle._dataclasses import Node

        data = {
            "id": "full-node",
            "type": "table",
            "name": "Full Table",
            "bbox_norm": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8},
            "size_mm": {"width": 170, "height": 120},
            "children": ["child-1", "child-2"],
            "refs": {"data": "data.csv", "stats": "stats.json"},
            "created_at": "2025-12-20T00:00:00Z",
            "modified_at": "2025-12-20T01:00:00Z",
        }
        node = Node.from_dict(data)

        assert node.id == "full-node"
        assert node.type == "table"
        assert node.name == "Full Table"
        assert node.children == ["child-1", "child-2"]
        assert node.created_at == "2025-12-20T00:00:00Z"

    def test_roundtrip_serialization(self):
        """Test that to_dict -> from_dict preserves data."""
        from scitex.io.bundle._bundle._dataclasses import Node, SizeMM

        original = Node(
            id="roundtrip-1",
            type="table",
            name="Roundtrip Test",
            size_mm=SizeMM(width=80, height=60),
            children=["a", "b"],
        )

        d = original.to_dict()
        restored = Node.from_dict(d)

        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.name == original.name
        assert restored.children == original.children


class TestNodeTouch:
    """Test Node touch method."""

    def test_touch_updates_modified(self):
        """Test that touch updates modified_at."""
        from scitex.io.bundle._bundle._dataclasses import Node
        import time

        node = Node(id="touch-test", type="plot")
        original_modified = node.modified_at

        time.sleep(0.01)  # Small delay to ensure different timestamp
        node.touch()

        assert node.modified_at != original_modified
        assert node.created_at != node.modified_at


class TestNodeTypes:
    """Test various node types."""

    def test_all_valid_types(self):
        """Test creating nodes with all valid types."""
        from scitex.io.bundle._bundle._dataclasses import Node

        valid_types = ["figure", "plot", "table", "stats", "image", "text", "shape", "symbol", "comment", "equation"]

        for node_type in valid_types:
            node = Node(id=f"{node_type}-test", type=node_type)
            assert node.type == node_type

    def test_table_type_integration(self):
        """Test table type works with full workflow."""
        from scitex.io.bundle._bundle._dataclasses import Node

        # Create table node
        table_node = Node(
            id="demographics",
            type="table",
            name="Subject Demographics",
        )

        # Serialize
        d = table_node.to_dict()
        assert d["type"] == "table"

        # Deserialize
        restored = Node.from_dict(d)
        assert restored.type == "table"
        assert restored.name == "Subject Demographics"


class TestNodeBBox:
    """Test Node bounding box handling."""

    def test_default_bbox(self):
        """Test default bounding box values."""
        from scitex.io.bundle._bundle._dataclasses import Node

        node = Node(id="bbox-test", type="plot")

        assert node.bbox_norm is not None
        # BBox uses x0, y0, x1, y1 format
        assert hasattr(node.bbox_norm, "x0")
        assert hasattr(node.bbox_norm, "y0")
        assert hasattr(node.bbox_norm, "x1")
        assert hasattr(node.bbox_norm, "y1")

    def test_custom_bbox(self):
        """Test custom bounding box."""
        from scitex.io.bundle._bundle._dataclasses import Node, BBox

        # BBox uses x0, y0, x1, y1 format
        bbox = BBox(x0=0.1, y0=0.2, x1=0.7, y1=0.7)
        node = Node(id="bbox-test", type="plot", bbox_norm=bbox)

        assert node.bbox_norm.x0 == 0.1
        assert node.bbox_norm.y0 == 0.2
        assert node.bbox_norm.x1 == 0.7
        assert node.bbox_norm.y1 == 0.7


class TestNodeRefs:
    """Test Node refs handling."""

    def test_default_refs(self):
        """Test default refs."""
        from scitex.io.bundle._bundle._dataclasses import Node

        node = Node(id="refs-test", type="plot")

        assert node.refs is not None

    def test_refs_from_dict(self):
        """Test refs from dictionary."""
        from scitex.io.bundle._bundle._dataclasses import Node

        data = {
            "id": "refs-test",
            "type": "table",
            "refs": {
                "data": "data/table1.csv",
                "stats": "stats/table1_stats.json",
            },
        }
        node = Node.from_dict(data)

        # Refs should be parsed
        assert node.refs is not None

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_dataclasses/_Node.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_dataclasses/_Node.py
# 
# """Node - Core FTS Node model with kind-based constraints."""
# 
# from dataclasses import dataclass, field
# from datetime import datetime
# from typing import Any, ClassVar, Dict, List, Optional, Set
# 
# from ._Axes import Axes
# from ._BBox import BBox
# from ._NodeRefs import NodeRefs
# from ._SizeMM import SizeMM
# 
# 
# @dataclass
# class Node:
#     """Core FTS Node model with kind-based constraints.
# 
#     The central structural element of an FTS bundle.
#     Stored in canonical/spec.json.
# 
#     Kind categories:
#     - Leaf kinds (plot, table, stats): children empty, payload required
#     - Composite kinds (figure, group, collection): payload empty, children optional
# 
#     All bundles have IDENTICAL directory structure:
#     - canonical/: Source of truth (spec.json, encoding.json, theme.json, etc.)
#     - payload/: ALWAYS exists (empty for composites, populated for leaves)
#     - artifacts/: Derived files (exports/, cache/)
#     - children/: ALWAYS exists (empty for leaves, populated for composites)
#     """
# 
#     # Required fields
#     id: str
#     kind: str  # "plot" | "figure" | "table" | "stats" | "group" | "collection"
# 
#     # Schema versioning for forward compatibility
#     scitex_schema: str = "scitex.fts.spec"
#     scitex_schema_version: str = "1.0.0"
# 
#     # Children and layout (for composite kinds)
#     children: List[str] = field(default_factory=list)  # UUID-based names (e.g., "690b1931.zip")
#     layout: Optional[Dict] = None  # {rows, cols, panels: [{child, child_id, label, row, col}], ...}
# 
#     # Payload schema (for leaf kinds)
#     payload_schema: Optional[str] = None  # e.g., "scitex.fts.payload.plot@1"
# 
#     # Visual properties
#     bbox_norm: BBox = field(default_factory=BBox)
#     name: Optional[str] = None
#     size_mm: Optional[SizeMM] = None
#     axes: Optional[Axes] = None
# 
#     # References and timestamps
#     refs: NodeRefs = field(default_factory=NodeRefs)
#     created_at: Optional[str] = None
#     modified_at: Optional[str] = None
# 
#     # === Kind Constants ===
#     # Leaf kinds: require payload/, forbid children
#     LEAF_KINDS: ClassVar[Set[str]] = {"plot", "table", "stats"}
#     # Composite kinds: forbid payload/, allow children (can be empty for WIP)
#     COMPOSITE_KINDS: ClassVar[Set[str]] = {"figure", "group", "collection"}
# 
#     # Payload schema -> required file mapping
#     PAYLOAD_REQUIRED_FILES: ClassVar[Dict[str, str]] = {
#         "scitex.fts.payload.plot@1": "payload/data.csv",
#         "scitex.fts.payload.table@1": "payload/table.csv",
#         "scitex.fts.payload.stats@1": "payload/stats.json",
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
#         """Check if this is a leaf kind (requires payload, forbids children)."""
#         return self.kind in self.LEAF_KINDS
# 
#     def is_composite_kind(self) -> bool:
#         """Check if this is a composite kind (forbids payload, allows children)."""
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
#         # Check: children list has no duplicates
#         if len(self.children) != len(set(self.children)):
#             errors.append("children list has duplicates")
# 
#         if self.is_leaf_kind():
#             # Leaf kinds: children must be empty
#             # Note: payload_schema is optional for backwards compatibility
#             # with legacy sio.save() bundles that may not have data files
#             if self.children:
#                 errors.append(f"kind={self.kind} cannot have children")
# 
#         elif self.is_composite_kind():
#             # Composite kinds: payload_schema must be None
#             # (payload prohibition enforced via payload_schema, NOT by listing files)
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
#                         errors.append(f"layout.panels references unknown child: {child_ref}")
# 
#                 # Check: no duplicate panel child references
#                 if len(panel_children) != len(set(panel_children)):
#                     errors.append("layout.panels has duplicate child references")
# 
#         else:
#             errors.append(f"Unknown kind: {self.kind}")
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
# 
#         result["refs"] = self.refs.to_dict()
#         result["created_at"] = self.created_at
#         result["modified_at"] = self.modified_at
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "Node":
#         """Create Node from dictionary."""
#         # Handle legacy 'type' field for backwards compatibility
#         kind = data.get("kind") or data.get("type", "plot")
# 
#         # Handle payload_schema - do NOT infer for legacy bundles
#         # Legacy sio.save() bundles may not have data files, so we don't
#         # assume payload_schema based on kind. It's optional.
#         payload_schema = data.get("payload_schema")
# 
#         # Handle size_mm from different formats (size vs size_mm)
#         size_mm_data = data.get("size_mm")
#         if size_mm_data is None and "size" in data:
#             # Legacy sio.save() format: size.width_mm, size.height_mm
#             size = data["size"]
#             if isinstance(size, dict):
#                 size_mm_data = {
#                     "width": size.get("width_mm", size.get("width", 85)),
#                     "height": size.get("height_mm", size.get("height", 85)),
#                 }
# 
#         # Handle name from 'title' field (legacy sio.save() format)
#         name = data.get("name") or data.get("title")
# 
#         return cls(
#             id=data.get("id", "unknown"),
#             kind=kind,
#             scitex_schema=data.get("scitex_schema", "scitex.fts.spec"),
#             scitex_schema_version=data.get("scitex_schema_version", "1.0.0"),
#             children=data.get("children", []),
#             layout=data.get("layout"),
#             payload_schema=payload_schema,
#             bbox_norm=BBox.from_dict(data.get("bbox_norm", {})),
#             name=name,
#             size_mm=SizeMM.from_dict(size_mm_data) if size_mm_data else None,
#             axes=Axes.from_dict(data["axes"]) if "axes" in data else None,
#             refs=NodeRefs.from_dict(data.get("refs", {})),
#             created_at=data.get("created_at"),
#             modified_at=data.get("modified_at"),
#         )
# 
#     def touch(self):
#         """Update modified timestamp."""
#         self.modified_at = datetime.utcnow().isoformat() + "Z"
# 
# 
# __all__ = ["Node"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_dataclasses/_Node.py
# --------------------------------------------------------------------------------
