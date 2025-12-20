#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_dataclasses/_Node.py

"""Node - Core FTS Node model with kind-based constraints."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Set

from ._Axes import Axes
from ._BBox import BBox
from ._NodeRefs import NodeRefs
from ._SizeMM import SizeMM


@dataclass
class Node:
    """Core FTS Node model with kind-based constraints.

    The central structural element of an FTS bundle.
    Stored in canonical/spec.json.

    Kind categories:
    - Leaf kinds (plot, table, stats): children empty, payload required
    - Composite kinds (figure, group, collection): payload empty, children optional

    All bundles have IDENTICAL directory structure:
    - canonical/: Source of truth (spec.json, encoding.json, theme.json, etc.)
    - payload/: ALWAYS exists (empty for composites, populated for leaves)
    - artifacts/: Derived files (exports/, cache/)
    - children/: ALWAYS exists (empty for leaves, populated for composites)
    """

    # Required fields
    id: str
    kind: str  # "plot" | "figure" | "table" | "stats" | "group" | "collection"

    # Schema versioning for forward compatibility
    scitex_schema: str = "scitex.fts.spec"
    scitex_schema_version: str = "1.0.0"

    # Children and layout (for composite kinds)
    children: List[str] = field(default_factory=list)  # UUID-based names (e.g., "690b1931.zip")
    layout: Optional[Dict] = None  # {rows, cols, panels: [{child, child_id, label, row, col}], ...}

    # Payload schema (for leaf kinds)
    payload_schema: Optional[str] = None  # e.g., "scitex.fts.payload.plot@1"

    # Visual properties
    bbox_norm: BBox = field(default_factory=BBox)
    name: Optional[str] = None
    size_mm: Optional[SizeMM] = None
    axes: Optional[Axes] = None

    # References and timestamps
    refs: NodeRefs = field(default_factory=NodeRefs)
    created_at: Optional[str] = None
    modified_at: Optional[str] = None

    # === Kind Constants ===
    # Leaf kinds: require payload/, forbid children
    LEAF_KINDS: ClassVar[Set[str]] = {"plot", "table", "stats"}
    # Composite kinds: forbid payload/, allow children (can be empty for WIP)
    COMPOSITE_KINDS: ClassVar[Set[str]] = {"figure", "group", "collection"}

    # Payload schema -> required file mapping
    PAYLOAD_REQUIRED_FILES: ClassVar[Dict[str, str]] = {
        "scitex.fts.payload.plot@1": "payload/data.csv",
        "scitex.fts.payload.table@1": "payload/table.csv",
        "scitex.fts.payload.stats@1": "payload/stats.json",
    }

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if self.modified_at is None:
            self.modified_at = self.created_at

    # === Kind Methods ===

    def is_leaf_kind(self) -> bool:
        """Check if this is a leaf kind (requires payload, forbids children)."""
        return self.kind in self.LEAF_KINDS

    def is_composite_kind(self) -> bool:
        """Check if this is a composite kind (forbids payload, allows children)."""
        return self.kind in self.COMPOSITE_KINDS

    def get_required_payload_file(self) -> Optional[str]:
        """Get required payload file path based on payload_schema."""
        return self.PAYLOAD_REQUIRED_FILES.get(self.payload_schema)

    # === Validation ===

    def validate(self) -> List[str]:
        """Validate logical constraints (not file existence).

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check: children list has no duplicates
        if len(self.children) != len(set(self.children)):
            errors.append("children list has duplicates")

        if self.is_leaf_kind():
            # Leaf kinds: children must be empty
            # Note: payload_schema is optional for backwards compatibility
            # with legacy sio.save() bundles that may not have data files
            if self.children:
                errors.append(f"kind={self.kind} cannot have children")

        elif self.is_composite_kind():
            # Composite kinds: payload_schema must be None
            # (payload prohibition enforced via payload_schema, NOT by listing files)
            if self.payload_schema:
                errors.append(f"kind={self.kind} should not have payload_schema")

            # Validate layout if present
            if self.layout:
                panels = self.layout.get("panels", [])
                panel_children = [p.get("child") for p in panels]

                # Check: panel child references must be subset of children
                for child_ref in panel_children:
                    if child_ref not in self.children:
                        errors.append(f"layout.panels references unknown child: {child_ref}")

                # Check: no duplicate panel child references
                if len(panel_children) != len(set(panel_children)):
                    errors.append("layout.panels has duplicate child references")

        else:
            errors.append(f"Unknown kind: {self.kind}")

        return errors

    # === Serialization ===

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "kind": self.kind,
            "scitex_schema": self.scitex_schema,
            "scitex_schema_version": self.scitex_schema_version,
            "bbox_norm": self.bbox_norm.to_dict(),
        }

        if self.name:
            result["name"] = self.name
        if self.size_mm:
            result["size_mm"] = self.size_mm.to_dict()
        if self.axes:
            result["axes"] = self.axes.to_dict()
        if self.children:
            result["children"] = self.children
        if self.layout:
            result["layout"] = self.layout
        if self.payload_schema:
            result["payload_schema"] = self.payload_schema

        result["refs"] = self.refs.to_dict()
        result["created_at"] = self.created_at
        result["modified_at"] = self.modified_at
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """Create Node from dictionary."""
        # Handle legacy 'type' field for backwards compatibility
        kind = data.get("kind") or data.get("type", "plot")

        # Handle payload_schema - do NOT infer for legacy bundles
        # Legacy sio.save() bundles may not have data files, so we don't
        # assume payload_schema based on kind. It's optional.
        payload_schema = data.get("payload_schema")

        # Handle size_mm from different formats (size vs size_mm)
        size_mm_data = data.get("size_mm")
        if size_mm_data is None and "size" in data:
            # Legacy sio.save() format: size.width_mm, size.height_mm
            size = data["size"]
            if isinstance(size, dict):
                size_mm_data = {
                    "width": size.get("width_mm", size.get("width", 85)),
                    "height": size.get("height_mm", size.get("height", 85)),
                }

        # Handle name from 'title' field (legacy sio.save() format)
        name = data.get("name") or data.get("title")

        return cls(
            id=data.get("id", "unknown"),
            kind=kind,
            scitex_schema=data.get("scitex_schema", "scitex.fts.spec"),
            scitex_schema_version=data.get("scitex_schema_version", "1.0.0"),
            children=data.get("children", []),
            layout=data.get("layout"),
            payload_schema=payload_schema,
            bbox_norm=BBox.from_dict(data.get("bbox_norm", {})),
            name=name,
            size_mm=SizeMM.from_dict(size_mm_data) if size_mm_data else None,
            axes=Axes.from_dict(data["axes"]) if "axes" in data else None,
            refs=NodeRefs.from_dict(data.get("refs", {})),
            created_at=data.get("created_at"),
            modified_at=data.get("modified_at"),
        )

    def touch(self):
        """Update modified timestamp."""
        self.modified_at = datetime.utcnow().isoformat() + "Z"


__all__ = ["Node"]

# EOF
