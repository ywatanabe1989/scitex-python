#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/schema/_canvas.py
# Time-stamp: "2024-12-09 08:25:00 (ywatanabe)"
"""
Canvas Schemas for SciTeX.

Defines the wire format for canvas specifications that are:
- Saved to .canvas directories
- Shared between vis module and cloud GUI
- Used for multi-panel figure composition

These are lightweight dataclasses that mirror the JSON structure,
separate from the Canvas class which provides the OO interface.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime


# Schema version for canvas format
CANVAS_SCHEMA_VERSION = "0.1.0"


@dataclass
class PanelPositionSpec:
    """Position specification for a panel."""

    x_mm: float = 0.0
    y_mm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PanelPositionSpec":
        return cls(**data)


@dataclass
class PanelSizeSpec:
    """Size specification for a panel."""

    width_mm: float = 50.0
    height_mm: float = 50.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PanelSizeSpec":
        return cls(**data)


@dataclass
class PanelClipSpec:
    """Clipping specification for a panel."""

    enabled: bool = False
    x_mm: float = 0.0
    y_mm: float = 0.0
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PanelClipSpec":
        return cls(**data)


@dataclass
class PanelLabelSpec:
    """Label specification for a panel (e.g., "A", "B", "C")."""

    text: str = ""
    position: str = "top-left"  # "top-left", "top-right", "bottom-left", "bottom-right"
    fontsize: int = 12
    fontweight: str = "bold"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PanelLabelSpec":
        return cls(**data)


@dataclass
class PanelBorderSpec:
    """Border specification for a panel."""

    visible: bool = False
    color: str = "#000000"
    width_mm: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PanelBorderSpec":
        return cls(**data)


@dataclass
class PanelSpec:
    """
    Specification for a single panel in a canvas.

    A panel can be either:
    - "scitex": A SciTeX figure (has accompanying .json metadata)
    - "image": A plain image file (PNG, JPG, SVG)
    """

    name: str
    type: str = "image"  # "scitex" or "image"
    position: PanelPositionSpec = field(default_factory=PanelPositionSpec)
    size: PanelSizeSpec = field(default_factory=PanelSizeSpec)
    z_index: int = 0
    rotation_deg: float = 0.0
    opacity: float = 1.0
    flip_h: bool = False
    flip_v: bool = False
    visible: bool = True
    clip: PanelClipSpec = field(default_factory=PanelClipSpec)
    label: PanelLabelSpec = field(default_factory=PanelLabelSpec)
    border: PanelBorderSpec = field(default_factory=PanelBorderSpec)
    source: Optional[str] = None  # File path/suffix for the panel source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "position": self.position.to_dict(),
            "size": self.size.to_dict(),
            "z_index": self.z_index,
            "rotation_deg": self.rotation_deg,
            "opacity": self.opacity,
            "flip_h": self.flip_h,
            "flip_v": self.flip_v,
            "visible": self.visible,
            "clip": self.clip.to_dict(),
            "label": self.label.to_dict(),
            "border": self.border.to_dict(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PanelSpec":
        data_copy = data.copy()

        # Convert nested specs
        if "position" in data_copy and isinstance(data_copy["position"], dict):
            data_copy["position"] = PanelPositionSpec.from_dict(data_copy["position"])

        if "size" in data_copy and isinstance(data_copy["size"], dict):
            data_copy["size"] = PanelSizeSpec.from_dict(data_copy["size"])

        if "clip" in data_copy and isinstance(data_copy["clip"], dict):
            data_copy["clip"] = PanelClipSpec.from_dict(data_copy["clip"])

        if "label" in data_copy and isinstance(data_copy["label"], dict):
            data_copy["label"] = PanelLabelSpec.from_dict(data_copy["label"])

        if "border" in data_copy and isinstance(data_copy["border"], dict):
            data_copy["border"] = PanelBorderSpec.from_dict(data_copy["border"])

        return cls(**data_copy)


@dataclass
class CanvasAnnotationSpec:
    """
    Specification for an annotation on the canvas.

    Supports various annotation types:
    - "text": Plain text
    - "arrow": Arrow with optional text
    - "bracket": Statistical comparison bracket
    - "line": Simple line
    - "rectangle": Rectangle shape
    - "legend": Custom legend
    """

    type: str  # "text", "arrow", "bracket", "line", "rectangle", "legend"
    id: Optional[str] = None

    # Common properties
    x_mm: float = 0.0
    y_mm: float = 0.0
    text: Optional[str] = None
    fontsize: int = 10
    color: str = "#000000"

    # Type-specific properties (stored in extra)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.type,
            "id": self.id,
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
            "text": self.text,
            "fontsize": self.fontsize,
            "color": self.color,
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanvasAnnotationSpec":
        known_fields = {"type", "id", "x_mm", "y_mm", "text", "fontsize", "color"}
        extra = {k: v for k, v in data.items() if k not in known_fields}
        core = {k: v for k, v in data.items() if k in known_fields}
        core["extra"] = extra
        return cls(**core)


@dataclass
class CanvasTitleSpec:
    """Specification for canvas title."""

    text: str = ""
    position: Dict[str, float] = field(default_factory=lambda: {"x_mm": 0, "y_mm": 0})
    fontsize: int = 14

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanvasTitleSpec":
        return cls(**data)


@dataclass
class CanvasCaptionSpec:
    """Specification for canvas caption (figure legend)."""

    text: str = ""
    render: bool = False  # Whether to render caption in output
    position: Dict[str, float] = field(default_factory=lambda: {"x_mm": 0, "y_mm": 0})
    fontsize: int = 10
    width_mm: Optional[float] = None  # For text wrapping

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanvasCaptionSpec":
        return cls(**data)


@dataclass
class CanvasBackgroundSpec:
    """Specification for canvas background."""

    color: str = "#ffffff"
    grid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanvasBackgroundSpec":
        return cls(**data)


@dataclass
class CanvasMetadataSpec:
    """Specification for canvas metadata."""

    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    author: str = ""
    description: str = ""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanvasMetadataSpec":
        return cls(**data)


@dataclass
class DataFileSpec:
    """Specification for a data file reference with integrity hash."""

    path: str  # Relative path within canvas directory
    hash: str  # Format: "sha256:{hex_digest}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataFileSpec":
        return cls(**data)


@dataclass
class CanvasSizeSpec:
    """Size specification for canvas."""

    width_mm: float = 180.0
    height_mm: float = 240.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanvasSizeSpec":
        return cls(**data)


@dataclass
class CanvasSpec:
    """
    Complete specification for a SciTeX canvas.

    This is the wire format for canvas.json files.
    The Canvas class in scitex.canvas provides the OO interface.

    Examples
    --------
    >>> spec = CanvasSpec(
    ...     canvas_name="fig1",
    ...     size=CanvasSizeSpec(width_mm=180, height_mm=120)
    ... )
    >>> spec.panels.append(PanelSpec(name="a", source="plot.png"))
    >>> json_dict = spec.to_dict()
    """

    canvas_name: str
    scitex_schema: str = "scitex.schema.canvas"
    scitex_schema_version: str = CANVAS_SCHEMA_VERSION
    size: CanvasSizeSpec = field(default_factory=CanvasSizeSpec)
    background: CanvasBackgroundSpec = field(default_factory=CanvasBackgroundSpec)
    panels: List[PanelSpec] = field(default_factory=list)
    annotations: List[CanvasAnnotationSpec] = field(default_factory=list)
    title: CanvasTitleSpec = field(default_factory=CanvasTitleSpec)
    caption: CanvasCaptionSpec = field(default_factory=CanvasCaptionSpec)
    data_files: List[DataFileSpec] = field(default_factory=list)
    metadata: CanvasMetadataSpec = field(default_factory=CanvasMetadataSpec)
    manual_overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scitex_schema": self.scitex_schema,
            "scitex_schema_version": self.scitex_schema_version,
            "canvas_name": self.canvas_name,
            "size": self.size.to_dict(),
            "background": self.background.to_dict(),
            "panels": [p.to_dict() for p in self.panels],
            "annotations": [a.to_dict() for a in self.annotations],
            "title": self.title.to_dict(),
            "caption": self.caption.to_dict(),
            "data_files": [d.to_dict() for d in self.data_files],
            "metadata": self.metadata.to_dict(),
            "manual_overrides": self.manual_overrides,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanvasSpec":
        """Create from dictionary."""
        return cls(
            canvas_name=data.get("canvas_name", ""),
            scitex_schema=data.get("scitex_schema", "scitex.schema.canvas"),
            scitex_schema_version=data.get("scitex_schema_version", CANVAS_SCHEMA_VERSION),
            size=CanvasSizeSpec.from_dict(data.get("size", {})),
            background=CanvasBackgroundSpec.from_dict(data.get("background", {})),
            panels=[PanelSpec.from_dict(p) for p in data.get("panels", [])],
            annotations=[
                CanvasAnnotationSpec.from_dict(a) for a in data.get("annotations", [])
            ],
            title=CanvasTitleSpec.from_dict(data.get("title", {})),
            caption=CanvasCaptionSpec.from_dict(data.get("caption", {})),
            data_files=[DataFileSpec.from_dict(d) for d in data.get("data_files", [])],
            metadata=CanvasMetadataSpec.from_dict(data.get("metadata", {})),
            manual_overrides=data.get("manual_overrides", {}),
        )

    def validate(self) -> bool:
        """
        Validate the canvas specification.

        Returns
        -------
        bool
            True if valid

        Raises
        ------
        ValueError
            If validation fails
        """
        if not self.canvas_name:
            raise ValueError("Canvas name is required")

        if self.size.width_mm <= 0 or self.size.height_mm <= 0:
            raise ValueError("Canvas size must be positive")

        # Validate panels
        panel_names = set()
        for panel in self.panels:
            if panel.name in panel_names:
                raise ValueError(f"Duplicate panel name: {panel.name}")
            panel_names.add(panel.name)

        return True


__all__ = [
    "CANVAS_SCHEMA_VERSION",
    # Panel specs
    "PanelPositionSpec",
    "PanelSizeSpec",
    "PanelClipSpec",
    "PanelLabelSpec",
    "PanelBorderSpec",
    "PanelSpec",
    # Canvas specs
    "CanvasAnnotationSpec",
    "CanvasTitleSpec",
    "CanvasCaptionSpec",
    "CanvasBackgroundSpec",
    "CanvasMetadataSpec",
    "CanvasSizeSpec",
    "CanvasSpec",
    # Data file spec
    "DataFileSpec",
]


# EOF
