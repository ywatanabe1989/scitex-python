#!/usr/bin/env python3
"""Tests for scitex.diagram._schema"""

import pytest

from scitex.diagram._schema import (
    ColumnLayout,
    DiagramSpec,
    DiagramType,
    EdgeSpec,
    LayoutHints,
    NodeSpec,
    PaperConstraints,
    PaperMode,
    SpacingLevel,
)


class TestDiagramType:
    def test_workflow_type(self):
        assert DiagramType.WORKFLOW.value == "workflow"

    def test_decision_type(self):
        assert DiagramType.DECISION.value == "decision"

    def test_pipeline_type(self):
        assert DiagramType.PIPELINE.value == "pipeline"

    def test_hierarchy_type(self):
        assert DiagramType.HIERARCHY.value == "hierarchy"

    def test_comparison_type(self):
        assert DiagramType.COMPARISON.value == "comparison"

    def test_all_types_exist(self):
        expected = {"workflow", "decision", "pipeline", "hierarchy", "comparison"}
        actual = {t.value for t in DiagramType}
        assert actual == expected


class TestColumnLayout:
    def test_single_column(self):
        assert ColumnLayout.SINGLE.value == "single"

    def test_double_column(self):
        assert ColumnLayout.DOUBLE.value == "double"

    def test_all_layouts_exist(self):
        expected = {"single", "double"}
        actual = {c.value for c in ColumnLayout}
        assert actual == expected


class TestSpacingLevel:
    def test_tight_spacing(self):
        assert SpacingLevel.TIGHT.value == "tight"

    def test_compact_spacing(self):
        assert SpacingLevel.COMPACT.value == "compact"

    def test_medium_spacing(self):
        assert SpacingLevel.MEDIUM.value == "medium"

    def test_large_spacing(self):
        assert SpacingLevel.LARGE.value == "large"

    def test_all_levels_exist(self):
        expected = {"tight", "compact", "medium", "large"}
        actual = {s.value for s in SpacingLevel}
        assert actual == expected


class TestPaperMode:
    def test_draft_mode(self):
        assert PaperMode.DRAFT.value == "draft"

    def test_publication_mode(self):
        assert PaperMode.PUBLICATION.value == "publication"


class TestNodeSpec:
    def test_create_node(self):
        node = NodeSpec(id="test", label="Test Node")
        assert node.id == "test"
        assert node.label == "Test Node"
        assert node.shape == "box"
        assert node.emphasis == "normal"

    def test_short_label_no_truncation(self):
        node = NodeSpec(id="test", label="Short")
        assert node.short_label(20) == "Short"

    def test_short_label_with_truncation(self):
        node = NodeSpec(id="test", label="This is a very long label")
        result = node.short_label(10)
        assert len(result) == 10
        assert result.endswith("...")


class TestEdgeSpec:
    def test_create_edge(self):
        edge = EdgeSpec(source="a", target="b")
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.style == "solid"
        assert edge.arrow == "normal"

    def test_edge_with_label(self):
        edge = EdgeSpec(source="a", target="b", label="connects")
        assert edge.label == "connects"

    def test_edge_dashed_style(self):
        edge = EdgeSpec(source="a", target="b", style="dashed")
        assert edge.style == "dashed"

    def test_edge_dotted_style(self):
        edge = EdgeSpec(source="a", target="b", style="dotted")
        assert edge.style == "dotted"

    def test_edge_arrow_none(self):
        edge = EdgeSpec(source="a", target="b", arrow="none")
        assert edge.arrow == "none"


class TestPaperConstraints:
    def test_default_constraints(self):
        pc = PaperConstraints()
        assert pc.column == ColumnLayout.SINGLE
        assert pc.max_width_mm == 170
        assert pc.reading_direction == "left_to_right"
        assert pc.mode == PaperMode.DRAFT
        assert pc.emphasize == []
        assert pc.main_flow == []
        assert pc.secondary_flow == []
        assert pc.return_edges == []

    def test_double_column(self):
        pc = PaperConstraints(column=ColumnLayout.DOUBLE)
        assert pc.column == ColumnLayout.DOUBLE
        assert pc.max_width_mm == 170

    def test_publication_mode(self):
        pc = PaperConstraints(mode=PaperMode.PUBLICATION)
        assert pc.mode == PaperMode.PUBLICATION

    def test_top_to_bottom_direction(self):
        pc = PaperConstraints(reading_direction="top_to_bottom")
        assert pc.reading_direction == "top_to_bottom"

    def test_with_emphasize_nodes(self):
        pc = PaperConstraints(emphasize=["node1", "node2"])
        assert "node1" in pc.emphasize
        assert "node2" in pc.emphasize

    def test_with_main_flow(self):
        pc = PaperConstraints(main_flow=["a", "b", "c"])
        assert pc.main_flow == ["a", "b", "c"]

    def test_with_return_edges(self):
        pc = PaperConstraints(return_edges=[("b", "a"), ("d", "c")])
        assert ("b", "a") in pc.return_edges


class TestLayoutHints:
    def test_default_hints(self):
        lh = LayoutHints()
        assert lh.layers == []
        assert lh.alignment == {}
        assert lh.layer_gap == SpacingLevel.MEDIUM
        assert lh.node_gap == SpacingLevel.MEDIUM
        assert lh.groups == {}

    def test_with_layers(self):
        lh = LayoutHints(layers=[["a", "b"], ["c", "d"]])
        assert len(lh.layers) == 2
        assert "a" in lh.layers[0]

    def test_with_groups(self):
        lh = LayoutHints(groups={"Processing": ["p1", "p2"], "Output": ["o1"]})
        assert "Processing" in lh.groups
        assert lh.groups["Processing"] == ["p1", "p2"]

    def test_tight_spacing(self):
        lh = LayoutHints(layer_gap=SpacingLevel.TIGHT, node_gap=SpacingLevel.TIGHT)
        assert lh.layer_gap == SpacingLevel.TIGHT
        assert lh.node_gap == SpacingLevel.TIGHT

    def test_with_alignment(self):
        lh = LayoutHints(alignment={"a": "left", "b": "center"})
        assert lh.alignment["a"] == "left"


class TestDiagramSpec:
    def test_default_spec(self):
        spec = DiagramSpec()
        assert spec.type == DiagramType.WORKFLOW
        assert spec.title == ""
        assert len(spec.nodes) == 0
        assert len(spec.edges) == 0

    def test_from_dict_basic(self):
        data = {
            "type": "workflow",
            "title": "Test Diagram",
            "nodes": [
                {"id": "a", "label": "Node A"},
                {"id": "b", "label": "Node B"},
            ],
            "edges": [
                {"from": "a", "to": "b"},
            ],
        }
        spec = DiagramSpec.from_dict(data)
        assert spec.type == DiagramType.WORKFLOW
        assert spec.title == "Test Diagram"
        assert len(spec.nodes) == 2
        assert len(spec.edges) == 1

    def test_from_dict_with_paper_constraints(self):
        data = {
            "type": "decision",
            "paper": {
                "column": "double",
                "mode": "publication",
                "emphasize": ["node1", "node2"],
            },
            "nodes": [],
            "edges": [],
        }
        spec = DiagramSpec.from_dict(data)
        assert spec.paper.column == ColumnLayout.DOUBLE
        assert spec.paper.mode == PaperMode.PUBLICATION
        assert spec.paper.emphasize == ["node1", "node2"]

    def test_from_dict_with_layers(self):
        data = {
            "type": "workflow",
            "layout": {
                "layers": [["a", "b"], ["c"]],
                "layer_gap": "tight",
            },
            "nodes": [],
            "edges": [],
        }
        spec = DiagramSpec.from_dict(data)
        assert spec.layout.layers == [["a", "b"], ["c"]]
        assert spec.layout.layer_gap == SpacingLevel.TIGHT


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/diagram/_schema.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: 2025-12-15
# # Author: ywatanabe / Claude
# # File: scitex/diagram/_schema.py
#
# """
# Schema definitions for SciTeX Diagram.
#
# The schema defines paper-specific constraints that Mermaid/Graphviz don't know:
# - Paper layout (single/double column, max width)
# - Reading direction preferences
# - Node emphasis for scientific communication
# - Semantic layer grouping
# """
#
# from dataclasses import dataclass, field
# from enum import Enum
# from typing import List, Dict, Optional, Literal
#
#
# class DiagramType(Enum):
#     """Semantic type of diagram - affects layout strategy."""
#     WORKFLOW = "workflow"       # Sequential process, prefer LR/TB flow
#     DECISION = "decision"       # Decision tree, prefer TB with branches
#     PIPELINE = "pipeline"       # Data pipeline, strict LR with stages
#     HIERARCHY = "hierarchy"     # Tree structure, TB with levels
#     COMPARISON = "comparison"   # Side-by-side, two columns
#
#
# class ColumnLayout(Enum):
#     """Paper column layout."""
#     SINGLE = "single"   # Full width (~170mm)
#     DOUBLE = "double"   # Half width (~85mm)
#
#
# class SpacingLevel(Enum):
#     """Abstract spacing levels - mapped to backend-specific values."""
#     TIGHT = "tight"       # Publication: minimal whitespace
#     COMPACT = "compact"
#     MEDIUM = "medium"
#     LARGE = "large"
#
#
# class PaperMode(Enum):
#     """Paper mode affects layout density and edge visibility."""
#     DRAFT = "draft"           # Full arrows, visible bidirectional, medium spacing
#     PUBLICATION = "publication"  # Compact, return edges hidden/dotted
#
#
# @dataclass
# class PaperConstraints:
#     """Paper-specific constraints that affect layout."""
#     column: ColumnLayout = ColumnLayout.SINGLE
#     max_width_mm: int = 170
#     reading_direction: Literal["left_to_right", "top_to_bottom"] = "left_to_right"
#     mode: PaperMode = PaperMode.DRAFT  # draft: full details, publication: compact
#     emphasize: List[str] = field(default_factory=list)  # Node IDs to highlight
#
#     # Scientific communication hints
#     main_flow: List[str] = field(default_factory=list)  # Critical path nodes
#     secondary_flow: List[str] = field(default_factory=list)  # Supporting elements
#     return_edges: List[tuple] = field(default_factory=list)  # Edges to hide in publication
#
#
# @dataclass
# class LayoutHints:
#     """Abstract layout hints - compiled to backend directives."""
#     layers: List[List[str]] = field(default_factory=list)  # Nodes grouped by rank
#     alignment: Dict[str, str] = field(default_factory=dict)  # Node alignment hints
#     layer_gap: SpacingLevel = SpacingLevel.MEDIUM
#     node_gap: SpacingLevel = SpacingLevel.MEDIUM
#
#     # Subgraph organization
#     groups: Dict[str, List[str]] = field(default_factory=dict)  # Named groups
#
#
# @dataclass
# class NodeSpec:
#     """Specification for a single node."""
#     id: str
#     label: str
#     shape: Literal["box", "rounded", "diamond", "circle", "stadium"] = "box"
#     emphasis: Literal["normal", "primary", "success", "warning", "muted"] = "normal"
#
#     def short_label(self, max_chars: int = 20) -> str:
#         """Return truncated label for compact layouts."""
#         if len(self.label) <= max_chars:
#             return self.label
#         return self.label[:max_chars-3] + "..."
#
#
# @dataclass
# class EdgeSpec:
#     """Specification for an edge between nodes."""
#     source: str
#     target: str
#     label: Optional[str] = None
#     style: Literal["solid", "dashed", "dotted"] = "solid"
#     arrow: Literal["normal", "none", "open"] = "normal"
#
#
# @dataclass
# class DiagramSpec:
#     """Complete diagram specification - the semantic layer."""
#
#     # Metadata
#     type: DiagramType = DiagramType.WORKFLOW
#     title: str = ""
#
#     # Paper constraints
#     paper: PaperConstraints = field(default_factory=PaperConstraints)
#
#     # Layout hints
#     layout: LayoutHints = field(default_factory=LayoutHints)
#
#     # Content
#     nodes: List[NodeSpec] = field(default_factory=list)
#     edges: List[EdgeSpec] = field(default_factory=list)
#
#     # Theme
#     theme: Dict[str, str] = field(default_factory=dict)
#
#     @classmethod
#     def from_dict(cls, data: dict) -> "DiagramSpec":
#         """Create DiagramSpec from dictionary (parsed YAML)."""
#         spec = cls()
#
#         # Parse type
#         if "type" in data:
#             spec.type = DiagramType(data["type"])
#
#         spec.title = data.get("title", "")
#
#         # Parse paper constraints
#         if "paper" in data:
#             p = data["paper"]
#             spec.paper = PaperConstraints(
#                 column=ColumnLayout(p.get("column", "single")),
#                 max_width_mm=p.get("max_width_mm", 170),
#                 reading_direction=p.get("reading_direction", "left_to_right"),
#                 mode=PaperMode(p.get("mode", "draft")),
#                 emphasize=p.get("emphasize", []),
#                 main_flow=p.get("main_flow", []),
#                 secondary_flow=p.get("secondary_flow", []),
#                 return_edges=[tuple(e) for e in p.get("return_edges", [])],
#             )
#
#         # Parse layout hints
#         if "layout" in data:
#             lt = data["layout"]
#             spec.layout = LayoutHints(
#                 layers=lt.get("layers", []),
#                 alignment=lt.get("alignment", {}),
#                 layer_gap=SpacingLevel(lt.get("layer_gap", "medium")),
#                 node_gap=SpacingLevel(lt.get("node_gap", "medium")),
#                 groups=lt.get("groups", {}),
#             )
#
#         # Parse nodes
#         for n in data.get("nodes", []):
#             spec.nodes.append(NodeSpec(
#                 id=n["id"],
#                 label=n.get("label", n["id"]),
#                 shape=n.get("shape", "box"),
#                 emphasis=n.get("emphasis", "normal"),
#             ))
#
#         # Parse edges
#         for e in data.get("edges", []):
#             spec.edges.append(EdgeSpec(
#                 source=e["from"] if "from" in e else e["source"],
#                 target=e["to"] if "to" in e else e["target"],
#                 label=e.get("label"),
#                 style=e.get("style", "solid"),
#                 arrow=e.get("arrow", "normal"),
#             ))
#
#         # Theme
#         spec.theme = data.get("theme", {})
#
#         return spec

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/diagram/_schema.py
# --------------------------------------------------------------------------------
