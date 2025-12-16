#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-15
# Author: ywatanabe / Claude
# File: scitex/diagram/_schema.py

"""
Schema definitions for SciTeX Diagram.

The schema defines paper-specific constraints that Mermaid/Graphviz don't know:
- Paper layout (single/double column, max width)
- Reading direction preferences
- Node emphasis for scientific communication
- Semantic layer grouping
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Literal


class DiagramType(Enum):
    """Semantic type of diagram - affects layout strategy."""
    WORKFLOW = "workflow"       # Sequential process, prefer LR/TB flow
    DECISION = "decision"       # Decision tree, prefer TB with branches
    PIPELINE = "pipeline"       # Data pipeline, strict LR with stages
    HIERARCHY = "hierarchy"     # Tree structure, TB with levels
    COMPARISON = "comparison"   # Side-by-side, two columns


class ColumnLayout(Enum):
    """Paper column layout."""
    SINGLE = "single"   # Full width (~170mm)
    DOUBLE = "double"   # Half width (~85mm)


class SpacingLevel(Enum):
    """Abstract spacing levels - mapped to backend-specific values."""
    TIGHT = "tight"       # Publication: minimal whitespace
    COMPACT = "compact"
    MEDIUM = "medium"
    LARGE = "large"


class PaperMode(Enum):
    """Paper mode affects layout density and edge visibility."""
    DRAFT = "draft"           # Full arrows, visible bidirectional, medium spacing
    PUBLICATION = "publication"  # Compact, return edges hidden/dotted


@dataclass
class PaperConstraints:
    """Paper-specific constraints that affect layout."""
    column: ColumnLayout = ColumnLayout.SINGLE
    max_width_mm: int = 170
    reading_direction: Literal["left_to_right", "top_to_bottom"] = "left_to_right"
    mode: PaperMode = PaperMode.DRAFT  # draft: full details, publication: compact
    emphasize: List[str] = field(default_factory=list)  # Node IDs to highlight

    # Scientific communication hints
    main_flow: List[str] = field(default_factory=list)  # Critical path nodes
    secondary_flow: List[str] = field(default_factory=list)  # Supporting elements
    return_edges: List[tuple] = field(default_factory=list)  # Edges to hide in publication


@dataclass
class LayoutHints:
    """Abstract layout hints - compiled to backend directives."""
    layers: List[List[str]] = field(default_factory=list)  # Nodes grouped by rank
    alignment: Dict[str, str] = field(default_factory=dict)  # Node alignment hints
    layer_gap: SpacingLevel = SpacingLevel.MEDIUM
    node_gap: SpacingLevel = SpacingLevel.MEDIUM

    # Subgraph organization
    groups: Dict[str, List[str]] = field(default_factory=dict)  # Named groups


@dataclass
class NodeSpec:
    """Specification for a single node."""
    id: str
    label: str
    shape: Literal["box", "rounded", "diamond", "circle", "stadium"] = "box"
    emphasis: Literal["normal", "primary", "success", "warning", "muted"] = "normal"

    def short_label(self, max_chars: int = 20) -> str:
        """Return truncated label for compact layouts."""
        if len(self.label) <= max_chars:
            return self.label
        return self.label[:max_chars-3] + "..."


@dataclass
class EdgeSpec:
    """Specification for an edge between nodes."""
    source: str
    target: str
    label: Optional[str] = None
    style: Literal["solid", "dashed", "dotted"] = "solid"
    arrow: Literal["normal", "none", "open"] = "normal"


@dataclass
class DiagramSpec:
    """Complete diagram specification - the semantic layer."""

    # Metadata
    type: DiagramType = DiagramType.WORKFLOW
    title: str = ""

    # Paper constraints
    paper: PaperConstraints = field(default_factory=PaperConstraints)

    # Layout hints
    layout: LayoutHints = field(default_factory=LayoutHints)

    # Content
    nodes: List[NodeSpec] = field(default_factory=list)
    edges: List[EdgeSpec] = field(default_factory=list)

    # Theme
    theme: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "DiagramSpec":
        """Create DiagramSpec from dictionary (parsed YAML)."""
        spec = cls()

        # Parse type
        if "type" in data:
            spec.type = DiagramType(data["type"])

        spec.title = data.get("title", "")

        # Parse paper constraints
        if "paper" in data:
            p = data["paper"]
            spec.paper = PaperConstraints(
                column=ColumnLayout(p.get("column", "single")),
                max_width_mm=p.get("max_width_mm", 170),
                reading_direction=p.get("reading_direction", "left_to_right"),
                mode=PaperMode(p.get("mode", "draft")),
                emphasize=p.get("emphasize", []),
                main_flow=p.get("main_flow", []),
                secondary_flow=p.get("secondary_flow", []),
                return_edges=[tuple(e) for e in p.get("return_edges", [])],
            )

        # Parse layout hints
        if "layout" in data:
            lt = data["layout"]
            spec.layout = LayoutHints(
                layers=lt.get("layers", []),
                alignment=lt.get("alignment", {}),
                layer_gap=SpacingLevel(lt.get("layer_gap", "medium")),
                node_gap=SpacingLevel(lt.get("node_gap", "medium")),
                groups=lt.get("groups", {}),
            )

        # Parse nodes
        for n in data.get("nodes", []):
            spec.nodes.append(NodeSpec(
                id=n["id"],
                label=n.get("label", n["id"]),
                shape=n.get("shape", "box"),
                emphasis=n.get("emphasis", "normal"),
            ))

        # Parse edges
        for e in data.get("edges", []):
            spec.edges.append(EdgeSpec(
                source=e["from"] if "from" in e else e["source"],
                target=e["to"] if "to" in e else e["target"],
                label=e.get("label"),
                style=e.get("style", "solid"),
                arrow=e.get("arrow", "normal"),
            ))

        # Theme
        spec.theme = data.get("theme", {})

        return spec
