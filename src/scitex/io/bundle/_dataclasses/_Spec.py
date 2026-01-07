#!/usr/bin/env python3
# Timestamp: 2025-12-21
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/bundle/_dataclasses/_Spec.py

"""Spec - Core bundle specification model with kind-based constraints."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Set

from ._Axes import Axes
from ._BBox import BBox
from ._SizeMM import SizeMM
from ._SpecRefs import SpecRefs


@dataclass
class TextContent:
    """Text content for kind=text specs."""

    content: str = ""
    fontsize: Optional[float] = None
    fontweight: Optional[str] = None  # "normal" | "bold"
    ha: str = "center"  # "left" | "center" | "right"
    va: str = "center"  # "top" | "center" | "bottom"

    def to_dict(self) -> Dict[str, Any]:
        result = {"content": self.content, "ha": self.ha, "va": self.va}
        if self.fontsize is not None:
            result["fontsize"] = self.fontsize
        if self.fontweight is not None:
            result["fontweight"] = self.fontweight
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextContent":
        return cls(
            content=data.get("content", ""),
            fontsize=data.get("fontsize"),
            fontweight=data.get("fontweight"),
            ha=data.get("ha", "center"),
            va=data.get("va", "center"),
        )


@dataclass
class ShapeParams:
    """Shape parameters for kind=shape specs."""

    shape_type: str = "rectangle"  # "rectangle" | "ellipse" | "arrow" | "line"
    color: str = "#000000"
    linewidth: float = 1.0
    fill: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape_type": self.shape_type,
            "color": self.color,
            "linewidth": self.linewidth,
            "fill": self.fill,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShapeParams":
        return cls(
            shape_type=data.get("shape_type", "rectangle"),
            color=data.get("color", "#000000"),
            linewidth=data.get("linewidth", 1.0),
            fill=data.get("fill", False),
        )


@dataclass
class Spec:
    """Core bundle specification model with kind-based constraints.

    The central structural element of a bundle.
    Stored in canonical/spec.json.

    Kind categories:
    - Data leaf kinds (plot, table, stats): require payload data files
    - Annotation leaf kinds (text, shape): no payload required, params in spec
    - Image leaf kinds (image): require payload image file
    - Composite kinds (figure): contain children, no payload

    All bundles have IDENTICAL directory structure:
    - canonical/: Source of truth (spec.json, encoding.json, theme.json, etc.)
    - payload/: ALWAYS exists (empty for composites/annotations, populated for data/image)
    - artifacts/: Derived files (exports/, cache/)
    - children/: ALWAYS exists (empty for leaves, populated for composites)
    """

    # Required fields
    id: str
    kind: str  # "figure" | "plot" | "table" | "stats" | "text" | "shape" | "image"

    # Schema versioning for forward compatibility
    scitex_schema: str = "scitex.io.bundle.spec"
    scitex_schema_version: str = "1.0.0"

    # Children and layout (for composite kinds)
    children: List[str] = field(default_factory=list)
    layout: Optional[Dict] = None

    # Payload schema (for data leaf kinds)
    payload_schema: Optional[str] = None

    # Visual properties
    bbox_norm: BBox = field(default_factory=BBox)
    name: Optional[str] = None
    size_mm: Optional[SizeMM] = None
    axes: Optional[Axes] = None

    # Kind-specific content
    text: Optional[TextContent] = None  # For kind=text
    shape: Optional[ShapeParams] = None  # For kind=shape

    # References and timestamps
    refs: SpecRefs = field(default_factory=SpecRefs)
    created_at: Optional[str] = None
    modified_at: Optional[str] = None

    # === Kind Constants ===
    # Data leaf kinds: require payload data files, forbid children
    DATA_LEAF_KINDS: ClassVar[Set[str]] = {"plot", "table", "stats"}
    # Annotation leaf kinds: no payload required, forbid children
    ANNOTATION_LEAF_KINDS: ClassVar[Set[str]] = {"text", "shape"}
    # Image leaf kinds: require payload image file, forbid children
    IMAGE_LEAF_KINDS: ClassVar[Set[str]] = {"image"}
    # All leaf kinds (for convenience)
    LEAF_KINDS: ClassVar[Set[str]] = (
        DATA_LEAF_KINDS | ANNOTATION_LEAF_KINDS | IMAGE_LEAF_KINDS
    )
    # Composite kinds: allow children, forbid payload
    COMPOSITE_KINDS: ClassVar[Set[str]] = {"figure"}
    # All valid kinds
    ALL_KINDS: ClassVar[Set[str]] = LEAF_KINDS | COMPOSITE_KINDS

    # Payload schema -> required file mapping
    PAYLOAD_REQUIRED_FILES: ClassVar[Dict[str, str]] = {
        "scitex.io.bundle.payload.plot@1": "payload/data.csv",
        "scitex.io.bundle.payload.table@1": "payload/table.csv",
        "scitex.io.bundle.payload.stats@1": "payload/stats.json",
        "scitex.io.bundle.payload.image@1": "payload/image.png",
    }

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if self.modified_at is None:
            self.modified_at = self.created_at

    # === Kind Methods ===

    def is_leaf_kind(self) -> bool:
        """Check if this is any leaf kind (forbids children)."""
        return self.kind in self.LEAF_KINDS

    def is_data_leaf_kind(self) -> bool:
        """Check if this is a data leaf kind (requires payload data)."""
        return self.kind in self.DATA_LEAF_KINDS

    def is_annotation_leaf_kind(self) -> bool:
        """Check if this is an annotation leaf kind (no payload required)."""
        return self.kind in self.ANNOTATION_LEAF_KINDS

    def is_image_leaf_kind(self) -> bool:
        """Check if this is an image leaf kind (requires payload image)."""
        return self.kind in self.IMAGE_LEAF_KINDS

    def is_composite_kind(self) -> bool:
        """Check if this is a composite kind (allows children, forbids payload)."""
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

        # Check: kind is valid
        if self.kind not in self.ALL_KINDS:
            errors.append(
                f"Unknown kind: {self.kind}. Valid kinds: {sorted(self.ALL_KINDS)}"
            )
            return errors  # Early return - other checks don't make sense

        # Check: children list has no duplicates
        if len(self.children) != len(set(self.children)):
            errors.append("children list has duplicates")

        if self.is_leaf_kind():
            # All leaf kinds: children must be empty
            if self.children:
                errors.append(f"kind={self.kind} cannot have children")

            # Data leaf kinds: payload_schema is optional but recommended
            # Annotation leaf kinds: should not have payload_schema
            if self.is_annotation_leaf_kind() and self.payload_schema:
                errors.append(f"kind={self.kind} should not have payload_schema")

        elif self.is_composite_kind():
            # Composite kinds: payload_schema must be None
            if self.payload_schema:
                errors.append(f"kind={self.kind} should not have payload_schema")

            # Validate layout if present
            if self.layout:
                panels = self.layout.get("panels", [])
                panel_children = [p.get("child") for p in panels]

                # Check: panel child references must be subset of children
                for child_ref in panel_children:
                    if child_ref not in self.children:
                        errors.append(
                            f"layout.panels references unknown child: {child_ref}"
                        )

                # Check: no duplicate panel child references
                if len(panel_children) != len(set(panel_children)):
                    errors.append("layout.panels has duplicate child references")

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
        if self.text:
            result["text"] = self.text.to_dict()
        if self.shape:
            result["shape"] = self.shape.to_dict()

        result["refs"] = self.refs.to_dict()
        result["created_at"] = self.created_at
        result["modified_at"] = self.modified_at
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Spec":
        """Create Spec from dictionary."""
        # Handle legacy 'type' field
        kind = data.get("kind") or data.get("type", "plot")

        # Handle payload_schema
        payload_schema = data.get("payload_schema")

        # Handle size_mm from different formats
        size_mm_data = data.get("size_mm")
        if size_mm_data is None and "size" in data:
            size = data["size"]
            if isinstance(size, dict):
                size_mm_data = {
                    "width": size.get("width_mm", size.get("width", 85)),
                    "height": size.get("height_mm", size.get("height", 85)),
                }

        # Handle name from 'title' field (legacy format)
        name = data.get("name") or data.get("title")

        # Handle text content
        text = None
        if "text" in data and isinstance(data["text"], dict):
            text = TextContent.from_dict(data["text"])

        # Handle shape params
        shape = None
        if "shape" in data and isinstance(data["shape"], dict):
            shape = ShapeParams.from_dict(data["shape"])

        return cls(
            id=data.get("id", "unknown"),
            kind=kind,
            scitex_schema=data.get("scitex_schema", "scitex.io.bundle.spec"),
            scitex_schema_version=data.get("scitex_schema_version", "1.0.0"),
            children=data.get("children", []),
            layout=data.get("layout"),
            payload_schema=payload_schema,
            bbox_norm=BBox.from_dict(data.get("bbox_norm", {})),
            name=name,
            size_mm=SizeMM.from_dict(size_mm_data) if size_mm_data else None,
            axes=Axes.from_dict(data["axes"]) if "axes" in data else None,
            text=text,
            shape=shape,
            refs=SpecRefs.from_dict(data.get("refs", {})),
            created_at=data.get("created_at"),
            modified_at=data.get("modified_at"),
        )

    def touch(self):
        """Update modified timestamp."""
        self.modified_at = datetime.utcnow().isoformat() + "Z"


__all__ = ["Spec", "TextContent", "ShapeParams"]

# EOF
