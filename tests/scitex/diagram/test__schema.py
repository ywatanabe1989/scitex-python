#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex.diagram._schema"""

import pytest
from scitex.diagram._schema import (
    DiagramType, ColumnLayout, SpacingLevel, PaperMode,
    PaperConstraints, LayoutHints, NodeSpec, EdgeSpec, DiagramSpec
)


class TestDiagramType:
    def test_workflow_type(self):
        assert DiagramType.WORKFLOW.value == "workflow"

    def test_decision_type(self):
        assert DiagramType.DECISION.value == "decision"

    def test_all_types_exist(self):
        expected = {"workflow", "decision", "pipeline", "hierarchy", "comparison"}
        actual = {t.value for t in DiagramType}
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
    pytest.main([__file__, "-v"])
