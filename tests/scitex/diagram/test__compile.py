#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex.diagram._compile"""

import pytest
from scitex.diagram._schema import DiagramSpec, NodeSpec, EdgeSpec, PaperMode
from scitex.diagram._compile import compile_to_mermaid, compile_to_graphviz


class TestCompileToMermaid:
    def test_empty_diagram(self):
        spec = DiagramSpec()
        result = compile_to_mermaid(spec)
        assert "graph" in result
        assert "%%{init:" in result

    def test_with_nodes(self):
        spec = DiagramSpec()
        spec.nodes = [
            NodeSpec(id="a", label="Node A"),
            NodeSpec(id="b", label="Node B"),
        ]
        result = compile_to_mermaid(spec)
        assert "a" in result
        assert "b" in result
        assert "Node A" in result

    def test_with_edges(self):
        spec = DiagramSpec()
        spec.nodes = [
            NodeSpec(id="a", label="A"),
            NodeSpec(id="b", label="B"),
        ]
        spec.edges = [EdgeSpec(source="a", target="b")]
        result = compile_to_mermaid(spec)
        assert "-->" in result

    def test_dashed_edge(self):
        spec = DiagramSpec()
        spec.edges = [EdgeSpec(source="a", target="b", style="dashed")]
        result = compile_to_mermaid(spec)
        assert "-.->" in result

    def test_with_groups(self):
        spec = DiagramSpec()
        spec.nodes = [NodeSpec(id="a", label="A")]
        spec.layout.groups = {"TestGroup": ["a"]}
        result = compile_to_mermaid(spec)
        assert "subgraph" in result
        assert "TestGroup" in result


class TestCompileToGraphviz:
    def test_empty_diagram(self):
        spec = DiagramSpec()
        result = compile_to_graphviz(spec)
        assert "digraph" in result
        assert "rankdir" in result

    def test_with_nodes(self):
        spec = DiagramSpec()
        spec.nodes = [
            NodeSpec(id="a", label="Node A"),
            NodeSpec(id="b", label="Node B"),
        ]
        result = compile_to_graphviz(spec)
        assert 'label="Node A"' in result
        assert 'label="Node B"' in result

    def test_publication_mode_spacing(self):
        spec = DiagramSpec()
        spec.paper.mode = PaperMode.PUBLICATION
        result = compile_to_graphviz(spec)
        assert "ranksep=0.3" in result
        assert "nodesep=0.2" in result

    def test_layers_generate_rank_same(self):
        spec = DiagramSpec()
        spec.nodes = [
            NodeSpec(id="a", label="A"),
            NodeSpec(id="b", label="B"),
        ]
        spec.layout.layers = [["a", "b"]]
        spec.paper.mode = PaperMode.PUBLICATION
        result = compile_to_graphviz(spec)
        assert "rank=same" in result

    def test_style_comma_separated(self):
        """Test that style attributes are comma-separated, not overwritten."""
        spec = DiagramSpec()
        spec.nodes = [NodeSpec(id="a", label="A", shape="rounded", emphasis="primary")]
        spec.paper.emphasize = ["a"]
        result = compile_to_graphviz(spec)
        # Should have style="filled,rounded" not separate style attributes
        assert 'style="filled,rounded"' in result or 'style="rounded,filled"' in result


class TestSanitizeId:
    def test_spaces_replaced(self):
        from scitex.diagram._compile import _sanitize_id
        assert _sanitize_id("hello world") == "hello_world"

    def test_special_chars_replaced(self):
        from scitex.diagram._compile import _sanitize_id
        assert _sanitize_id("(A) Test") == "A_Test"

    def test_multiple_underscores_collapsed(self):
        from scitex.diagram._compile import _sanitize_id
        assert _sanitize_id("a---b") == "a_b"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
