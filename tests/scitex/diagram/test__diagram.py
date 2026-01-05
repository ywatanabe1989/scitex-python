#!/usr/bin/env python3
"""Tests for scitex.diagram._diagram"""

import tempfile
from pathlib import Path

import pytest

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

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/diagram/_diagram.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: 2025-12-15
# # Author: ywatanabe / Claude
# # File: scitex/diagram/_diagram.py
#
# """
# Main Diagram class - the user-facing API.
# """
#
# from pathlib import Path
# from typing import Optional, Union, List
# import yaml
# import re
#
# from scitex.diagram._schema import DiagramSpec, DiagramType, NodeSpec, EdgeSpec, PaperMode
# from scitex.diagram._compile import compile_to_mermaid, compile_to_graphviz
# from scitex.diagram._presets import get_preset
# from scitex.diagram._split import split_diagram, SplitConfig, SplitStrategy, SplitResult
#
#
# class Diagram:
#     """
#     Paper-optimized diagram with semantic specification.
#
#     This class provides the main interface for creating diagrams
#     that compile to Mermaid or Graphviz with paper-appropriate
#     layout constraints.
#
#     Examples
#     --------
#     >>> # From YAML spec
#     >>> d = Diagram.from_yaml("workflow.diagram.yaml")
#     >>> d.to_mermaid("workflow.mmd")
#
#     >>> # From existing Mermaid (parse and enhance)
#     >>> d = Diagram.from_mermaid("existing.mmd", diagram_type="workflow")
#     >>> d.spec.paper.column = "double"
#     >>> d.to_mermaid("enhanced.mmd")
#
#     >>> # Programmatic creation
#     >>> d = Diagram(type="pipeline")
#     >>> d.add_node("input", "Raw Data")
#     >>> d.add_node("process", "Transform", emphasis="primary")
#     >>> d.add_node("output", "Results")
#     >>> d.add_edge("input", "process")
#     >>> d.add_edge("process", "output")
#     >>> print(d.to_mermaid())
#     """
#
#     def __init__(
#         self,
#         type: str = "workflow",
#         title: str = "",
#         column: str = "single",
#     ):
#         """
#         Initialize a new diagram.
#
#         Parameters
#         ----------
#         type : str
#             Diagram type: workflow, decision, pipeline, hierarchy.
#         title : str
#             Diagram title.
#         column : str
#             Paper column: single or double.
#         """
#         self.spec = DiagramSpec(
#             type=DiagramType(type),
#             title=title,
#         )
#         self.spec.paper.column = column
#
#     @classmethod
#     def from_yaml(cls, path: Union[str, Path]) -> "Diagram":
#         """
#         Load diagram from YAML specification file.
#
#         Parameters
#         ----------
#         path : str or Path
#             Path to YAML file.
#
#         Returns
#         -------
#         Diagram
#             Loaded diagram.
#         """
#         path = Path(path)
#         with open(path, "r", encoding="utf-8") as f:
#             data = yaml.safe_load(f)
#
#         diagram = cls.__new__(cls)
#         diagram.spec = DiagramSpec.from_dict(data)
#         return diagram
#
#     @classmethod
#     def from_mermaid(
#         cls,
#         path: Union[str, Path],
#         diagram_type: str = "workflow",
#     ) -> "Diagram":
#         """
#         Parse existing Mermaid file and create enhanced Diagram.
#
#         This allows upgrading existing Mermaid files with SciTeX
#         paper constraints while preserving the original structure.
#
#         Parameters
#         ----------
#         path : str or Path
#             Path to .mmd file.
#         diagram_type : str
#             Inferred diagram type.
#
#         Returns
#         -------
#         Diagram
#             Parsed diagram (can be enhanced and re-exported).
#         """
#         path = Path(path)
#         with open(path, "r", encoding="utf-8") as f:
#             content = f.read()
#
#         diagram = cls(type=diagram_type)
#         diagram._parse_mermaid(content)
#         return diagram
#
#     def _parse_mermaid(self, content: str):
#         """Parse Mermaid content to extract structure."""
#         # Extract nodes
#         # Pattern: ID["Label"] or ID("Label") etc.
#         node_pattern = r'(\w+)\s*[\[\(\{<][\"\']?([^"\'\]\)\}>]+)[\"\']?[\]\)\}>]'
#
#         for match in re.finditer(node_pattern, content):
#             node_id = match.group(1)
#             label = match.group(2).strip()
#             # Skip if it looks like a subgraph name
#             if node_id.lower() not in ["subgraph", "end", "style", "graph", "direction"]:
#                 # Check for duplicates
#                 existing = [n for n in self.spec.nodes if n.id == node_id]
#                 if not existing:
#                     self.spec.nodes.append(NodeSpec(id=node_id, label=label))
#
#         # Extract edges
#         # Pattern: A --> B or A -->|label| B
#         edge_pattern = r'(\w+)\s*(-->|-.->|-----|---)\s*(?:\|["\']?([^|"\']+)["\']?\|)?\s*(\w+)'
#
#         for match in re.finditer(edge_pattern, content):
#             source = match.group(1)
#             arrow = match.group(2)
#             label = match.group(3)
#             target = match.group(4)
#
#             style = "solid"
#             if "-.->" in arrow or "-.." in arrow:
#                 style = "dashed"
#
#             self.spec.edges.append(EdgeSpec(
#                 source=source,
#                 target=target,
#                 label=label.strip() if label else None,
#                 style=style,
#             ))
#
#         # Extract subgraphs as groups
#         subgraph_pattern = r'subgraph\s+(\w+)\s*\[?"?([^"\]]*)"?\]?'
#         for match in re.finditer(subgraph_pattern, content):
#             group_id = match.group(1)
#             group_name = match.group(2) or group_id
#             self.spec.layout.groups[group_name] = []
#
#     def add_node(
#         self,
#         id: str,
#         label: str,
#         shape: str = "box",
#         emphasis: str = "normal",
#     ):
#         """Add a node to the diagram."""
#         self.spec.nodes.append(NodeSpec(
#             id=id,
#             label=label,
#             shape=shape,
#             emphasis=emphasis,
#         ))
#
#     def add_edge(
#         self,
#         source: str,
#         target: str,
#         label: Optional[str] = None,
#         style: str = "solid",
#     ):
#         """Add an edge between nodes."""
#         self.spec.edges.append(EdgeSpec(
#             source=source,
#             target=target,
#             label=label,
#             style=style,
#         ))
#
#     def set_group(self, group_name: str, node_ids: list):
#         """Define a group of nodes (rendered as subgraph)."""
#         self.spec.layout.groups[group_name] = node_ids
#
#     def emphasize(self, *node_ids: str):
#         """Mark nodes as emphasized (primary styling)."""
#         self.spec.paper.emphasize.extend(node_ids)
#
#     def to_mermaid(self, path: Optional[Union[str, Path]] = None) -> str:
#         """
#         Compile to Mermaid format.
#
#         Parameters
#         ----------
#         path : str or Path, optional
#             If provided, write to file.
#
#         Returns
#         -------
#         str
#             Mermaid source code.
#         """
#         result = compile_to_mermaid(self.spec)
#
#         if path:
#             path = Path(path)
#             with open(path, "w", encoding="utf-8") as f:
#                 f.write(result)
#
#         return result
#
#     def to_graphviz(self, path: Optional[Union[str, Path]] = None) -> str:
#         """
#         Compile to Graphviz DOT format.
#
#         Parameters
#         ----------
#         path : str or Path, optional
#             If provided, write to file.
#
#         Returns
#         -------
#         str
#             Graphviz DOT source code.
#         """
#         result = compile_to_graphviz(self.spec)
#
#         if path:
#             path = Path(path)
#             with open(path, "w", encoding="utf-8") as f:
#                 f.write(result)
#
#         return result
#
#     def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
#         """
#         Export specification as YAML.
#
#         Parameters
#         ----------
#         path : str or Path, optional
#             If provided, write to file.
#
#         Returns
#         -------
#         str
#             YAML specification.
#         """
#         data = {
#             "type": self.spec.type.value,
#             "title": self.spec.title,
#             "paper": {
#                 "column": self.spec.paper.column.value if hasattr(self.spec.paper.column, 'value') else self.spec.paper.column,
#                 "max_width_mm": self.spec.paper.max_width_mm,
#                 "reading_direction": self.spec.paper.reading_direction,
#                 "mode": self.spec.paper.mode.value if hasattr(self.spec.paper.mode, 'value') else self.spec.paper.mode,
#                 "emphasize": self.spec.paper.emphasize,
#             },
#             "layout": {
#                 "groups": self.spec.layout.groups,
#                 "layers": self.spec.layout.layers,
#                 "layer_gap": self.spec.layout.layer_gap.value,
#                 "node_gap": self.spec.layout.node_gap.value,
#             },
#             "nodes": [
#                 {"id": n.id, "label": n.label, "shape": n.shape, "emphasis": n.emphasis}
#                 for n in self.spec.nodes
#             ],
#             "edges": [
#                 {"from": e.source, "to": e.target, "label": e.label, "style": e.style}
#                 for e in self.spec.edges
#             ],
#         }
#
#         result = yaml.dump(data, default_flow_style=False, allow_unicode=True)
#
#         if path:
#             path = Path(path)
#             with open(path, "w", encoding="utf-8") as f:
#                 f.write(result)
#
#         return result
#
#     def split(
#         self,
#         max_nodes: int = 12,
#         strategy: str = "by_groups",
#         keep_hubs: bool = True,
#     ) -> List["Diagram"]:
#         """
#         Split diagram into multiple figures if too large.
#
#         Parameters
#         ----------
#         max_nodes : int
#             Maximum nodes per figure before splitting.
#         strategy : str
#             Split strategy: "by_groups" or "by_articulation".
#         keep_hubs : bool
#             Show hub nodes as ghosts in both parts.
#
#         Returns
#         -------
#         List[Diagram]
#             List of split diagrams (or single diagram if no split needed).
#
#         Examples
#         --------
#         >>> d = Diagram.from_yaml("large_workflow.yaml")
#         >>> parts = d.split(max_nodes=8)
#         >>> for i, part in enumerate(parts):
#         ...     part.to_mermaid(f"workflow_part_{i+1}.mmd")
#         """
#         config = SplitConfig(
#             enabled=True,
#             max_nodes=max_nodes,
#             strategy=SplitStrategy(strategy),
#             keep_hubs=keep_hubs,
#         )
#
#         result = split_diagram(self.spec, config)
#
#         # Wrap each spec in a Diagram object
#         diagrams = []
#         for fig_spec, label in zip(result.figures, result.labels):
#             d = Diagram.__new__(Diagram)
#             d.spec = fig_spec
#             if label:
#                 d.spec.title = f"{self.spec.title} ({label})" if self.spec.title else f"Part {label}"
#             diagrams.append(d)
#
#         return diagrams

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/diagram/_diagram.py
# --------------------------------------------------------------------------------
