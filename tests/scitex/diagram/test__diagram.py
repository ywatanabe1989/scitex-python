#!/usr/bin/env python3
"""Tests for scitex.diagram._diagram"""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip(
    "scitex.diagram._diagram", reason="scitex.diagram._diagram module not available"
)

from scitex.diagram import (  # noqa: E402  # type: ignore[attr-defined]
    Diagram,
    PaperMode,
)


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

    def test_to_graphviz_file(self):
        d = Diagram(type="workflow")
        d.add_node("a", "A")
        d.add_node("b", "B")
        d.add_edge("a", "b")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dot"
            d.to_graphviz(path)
            assert path.exists()
            content = path.read_text()
            assert "digraph" in content


class TestFromYaml:
    def test_from_yaml_basic(self):
        yaml_content = """
type: workflow
title: Test Workflow
nodes:
  - id: a
    label: Node A
  - id: b
    label: Node B
edges:
  - from: a
    to: b
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yaml"
            path.write_text(yaml_content)
            d = Diagram.from_yaml(path)
            assert d.spec.type.value == "workflow"
            assert d.spec.title == "Test Workflow"
            assert len(d.spec.nodes) == 2
            assert len(d.spec.edges) == 1

    def test_from_yaml_with_paper_constraints(self):
        yaml_content = """
type: decision
paper:
  column: double
  mode: publication
  reading_direction: top_to_bottom
nodes: []
edges: []
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yaml"
            path.write_text(yaml_content)
            d = Diagram.from_yaml(path)
            assert d.spec.type.value == "decision"
            assert d.spec.paper.column.value == "double"
            assert d.spec.paper.mode.value == "publication"

    def test_from_yaml_with_groups(self):
        yaml_content = """
type: pipeline
layout:
  groups:
    Input: [a, b]
    Output: [c, d]
nodes:
  - id: a
    label: A
  - id: b
    label: B
  - id: c
    label: C
  - id: d
    label: D
edges: []
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yaml"
            path.write_text(yaml_content)
            d = Diagram.from_yaml(path)
            assert "Input" in d.spec.layout.groups
            assert d.spec.layout.groups["Input"] == ["a", "b"]


class TestFromMermaid:
    def test_from_mermaid_basic(self):
        mermaid_content = """graph LR
    A["Node A"] --> B["Node B"]
    B --> C["Node C"]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mmd"
            path.write_text(mermaid_content)
            d = Diagram.from_mermaid(path)
            assert len(d.spec.nodes) >= 2
            assert len(d.spec.edges) >= 1

    def test_from_mermaid_with_diagram_type(self):
        mermaid_content = """graph TB
    Start["Begin"] --> Process["Process"]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mmd"
            path.write_text(mermaid_content)
            d = Diagram.from_mermaid(path, diagram_type="decision")
            assert d.spec.type.value == "decision"

    def test_from_mermaid_with_edge_labels(self):
        mermaid_content = """graph LR
    A["Node A"] -->|yes| B["Node B"]
    A -->|no| C["Node C"]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mmd"
            path.write_text(mermaid_content)
            d = Diagram.from_mermaid(path)
            assert len(d.spec.edges) >= 1
            labeled_edges = [e for e in d.spec.edges if e.label]
            assert len(labeled_edges) >= 1

    def test_from_mermaid_with_dashed_edges(self):
        # Parser currently handles edges on separate lines from node definitions
        mermaid_content = """graph LR
    A["Start"]
    B["End"]
    A -.-> B
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mmd"
            path.write_text(mermaid_content)
            d = Diagram.from_mermaid(path)
            dashed_edges = [e for e in d.spec.edges if e.style == "dashed"]
            assert len(dashed_edges) >= 1


class TestYamlRoundTrip:
    def test_yaml_roundtrip(self):
        """Test that saving to YAML and loading back preserves structure."""
        d1 = Diagram(type="workflow", title="Roundtrip Test")
        d1.add_node("a", "Node A", shape="rounded")
        d1.add_node("b", "Node B", emphasis="primary")
        d1.add_edge("a", "b", label="connects")
        d1.set_group("Processing", ["a", "b"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roundtrip.yaml"
            d1.to_yaml(path)
            d2 = Diagram.from_yaml(path)

            assert d2.spec.title == "Roundtrip Test"
            assert len(d2.spec.nodes) == 2
            assert len(d2.spec.edges) == 1


class TestDiagramTypes:
    def test_all_diagram_types(self):
        for dtype in ["workflow", "decision", "pipeline", "hierarchy"]:
            d = Diagram(type=dtype)
            assert d.spec.type.value == dtype

    def test_column_parameter(self):
        d = Diagram(type="workflow", column="double")
        assert d.spec.paper.column == "double"


class TestEdgeStyles:
    def test_dashed_edge(self):
        d = Diagram(type="workflow")
        d.add_edge("a", "b", style="dashed")
        assert d.spec.edges[0].style == "dashed"

    def test_edge_with_label(self):
        d = Diagram(type="workflow")
        d.add_edge("a", "b", label="yes")
        assert d.spec.edges[0].label == "yes"


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
