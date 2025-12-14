#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex.diagram._diagram"""

import pytest
import tempfile
from pathlib import Path
from scitex.diagram import Diagram, PaperMode


class TestDiagramCreation:
    def test_create_empty_diagram(self):
        d = Diagram(type="workflow")
        assert d.spec.type.value == "workflow"
        assert len(d.spec.nodes) == 0

    def test_create_with_title(self):
        d = Diagram(type="decision", title="Test Diagram")
        assert d.spec.title == "Test Diagram"


class TestAddNodes:
    def test_add_simple_node(self):
        d = Diagram(type="workflow")
        d.add_node("a", "Node A")
        assert len(d.spec.nodes) == 1
        assert d.spec.nodes[0].id == "a"
        assert d.spec.nodes[0].label == "Node A"

    def test_add_node_with_shape(self):
        d = Diagram(type="workflow")
        d.add_node("a", "Node A", shape="diamond")
        assert d.spec.nodes[0].shape == "diamond"

    def test_add_node_with_emphasis(self):
        d = Diagram(type="workflow")
        d.add_node("a", "Node A", emphasis="primary")
        assert d.spec.nodes[0].emphasis == "primary"


class TestAddEdges:
    def test_add_simple_edge(self):
        d = Diagram(type="workflow")
        d.add_node("a", "A")
        d.add_node("b", "B")
        d.add_edge("a", "b")
        assert len(d.spec.edges) == 1
        assert d.spec.edges[0].source == "a"
        assert d.spec.edges[0].target == "b"

    def test_add_edge_with_label(self):
        d = Diagram(type="workflow")
        d.add_edge("a", "b", label="connects")
        assert d.spec.edges[0].label == "connects"


class TestGroups:
    def test_set_group(self):
        d = Diagram(type="workflow")
        d.add_node("a", "A")
        d.add_node("b", "B")
        d.set_group("Group1", ["a", "b"])
        assert "Group1" in d.spec.layout.groups
        assert d.spec.layout.groups["Group1"] == ["a", "b"]


class TestEmphasize:
    def test_emphasize_nodes(self):
        d = Diagram(type="workflow")
        d.add_node("a", "A")
        d.add_node("b", "B")
        d.emphasize("a", "b")
        assert "a" in d.spec.paper.emphasize
        assert "b" in d.spec.paper.emphasize


class TestToMermaid:
    def test_simple_mermaid_output(self):
        d = Diagram(type="workflow")
        d.add_node("a", "Node A")
        d.add_node("b", "Node B")
        d.add_edge("a", "b")
        mmd = d.to_mermaid()
        assert "graph" in mmd
        assert "a" in mmd
        assert "b" in mmd
        assert "-->" in mmd

    def test_mermaid_with_groups(self):
        d = Diagram(type="workflow")
        d.add_node("a", "A")
        d.add_node("b", "B")
        d.set_group("Test Group", ["a", "b"])
        mmd = d.to_mermaid()
        assert "subgraph" in mmd
        assert "Test Group" in mmd


class TestToGraphviz:
    def test_simple_graphviz_output(self):
        d = Diagram(type="workflow")
        d.add_node("a", "Node A")
        d.add_node("b", "Node B")
        d.add_edge("a", "b")
        dot = d.to_graphviz()
        assert "digraph" in dot
        assert "rankdir" in dot
        assert "a ->" in dot or "a->" in dot

    def test_publication_mode_tight_spacing(self):
        d = Diagram(type="workflow")
        d.add_node("a", "A")
        d.spec.paper.mode = PaperMode.PUBLICATION
        dot = d.to_graphviz()
        assert "ranksep=0.3" in dot
        assert "nodesep=0.2" in dot


class TestSplit:
    def test_no_split_when_small(self):
        d = Diagram(type="workflow")
        d.add_node("a", "A")
        d.add_node("b", "B")
        parts = d.split(max_nodes=10)
        assert len(parts) == 1

    def test_split_large_diagram(self):
        d = Diagram(type="workflow")
        for i in range(10):
            d.add_node(f"n{i}", f"Node {i}")
        d.set_group("Group1", ["n0", "n1", "n2"])
        d.set_group("Group2", ["n3", "n4", "n5"])
        d.set_group("Group3", ["n6", "n7", "n8", "n9"])
        parts = d.split(max_nodes=4)
        assert len(parts) >= 2


class TestFileIO:
    def test_to_mermaid_file(self):
        d = Diagram(type="workflow")
        d.add_node("a", "A")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mmd"
            d.to_mermaid(path)
            assert path.exists()
            content = path.read_text()
            assert "graph" in content

    def test_to_yaml_file(self):
        d = Diagram(type="workflow", title="Test")
        d.add_node("a", "A")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yaml"
            d.to_yaml(path)
            assert path.exists()
            content = path.read_text()
            assert "type: workflow" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
