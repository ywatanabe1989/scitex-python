#!/usr/bin/env python3
"""Tests for scitex.diagram._schema"""

import pytest

pytest.importorskip(
    "scitex.diagram._schema", reason="scitex.diagram._schema module not available"
)

from scitex.diagram._schema import (  # noqa: E402
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
