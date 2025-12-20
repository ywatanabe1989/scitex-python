#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_dataclasses/Node.py

"""Node - Core FSB Node model."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ._Axes import Axes
from ._BBox import BBox
from ._NodeRefs import NodeRefs
from ._SizeMM import SizeMM


@dataclass
class Node:
    """Core FSB Node model.

    The central structural element of an FTS bundle.
    Stored in node.json at bundle root.

    Node types:
    - figure: Container for multiple children (plots, text, shapes)
    - plot: Single visualization with data
    - text: Text element
    - shape: Geometric shape
    - image: Raster image
    - stats: Statistical results bundle
    """

    id: str
    type: str  # figure, plot, text, shape, image, stats
    bbox_norm: BBox = field(default_factory=BBox)
    name: Optional[str] = None
    size_mm: Optional[SizeMM] = None
    axes: Optional[Axes] = None
    children: List[str] = field(default_factory=list)
    refs: NodeRefs = field(default_factory=NodeRefs)
    created_at: Optional[str] = None
    modified_at: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if self.modified_at is None:
            self.modified_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "type": self.type,
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
        result["refs"] = self.refs.to_dict()
        result["created_at"] = self.created_at
        result["modified_at"] = self.modified_at
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        return cls(
            id=data.get("id", "unknown"),
            type=data.get("type", "plot"),
            bbox_norm=BBox.from_dict(data.get("bbox_norm", {})),
            name=data.get("name"),
            size_mm=SizeMM.from_dict(data["size_mm"]) if "size_mm" in data else None,
            axes=Axes.from_dict(data["axes"]) if "axes" in data else None,
            children=data.get("children", []),
            refs=NodeRefs.from_dict(data.get("refs", {})),
            created_at=data.get("created_at"),
            modified_at=data.get("modified_at"),
        )

    def touch(self):
        """Update modified timestamp."""
        self.modified_at = datetime.utcnow().isoformat() + "Z"


__all__ = ["Node"]

# EOF
