#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-15
# Author: ywatanabe / Claude
# File: scitex/diagram/_diagram.py

"""
Main Diagram class - the user-facing API.
"""

from pathlib import Path
from typing import Optional, Union, List
import yaml
import re

from scitex.diagram._schema import DiagramSpec, DiagramType, NodeSpec, EdgeSpec, PaperMode
from scitex.diagram._compile import compile_to_mermaid, compile_to_graphviz
from scitex.diagram._presets import get_preset
from scitex.diagram._split import split_diagram, SplitConfig, SplitStrategy, SplitResult


class Diagram:
    """
    Paper-optimized diagram with semantic specification.

    This class provides the main interface for creating diagrams
    that compile to Mermaid or Graphviz with paper-appropriate
    layout constraints.

    Examples
    --------
    >>> # From YAML spec
    >>> d = Diagram.from_yaml("workflow.diagram.yaml")
    >>> d.to_mermaid("workflow.mmd")

    >>> # From existing Mermaid (parse and enhance)
    >>> d = Diagram.from_mermaid("existing.mmd", diagram_type="workflow")
    >>> d.spec.paper.column = "double"
    >>> d.to_mermaid("enhanced.mmd")

    >>> # Programmatic creation
    >>> d = Diagram(type="pipeline")
    >>> d.add_node("input", "Raw Data")
    >>> d.add_node("process", "Transform", emphasis="primary")
    >>> d.add_node("output", "Results")
    >>> d.add_edge("input", "process")
    >>> d.add_edge("process", "output")
    >>> print(d.to_mermaid())
    """

    def __init__(
        self,
        type: str = "workflow",
        title: str = "",
        column: str = "single",
    ):
        """
        Initialize a new diagram.

        Parameters
        ----------
        type : str
            Diagram type: workflow, decision, pipeline, hierarchy.
        title : str
            Diagram title.
        column : str
            Paper column: single or double.
        """
        self.spec = DiagramSpec(
            type=DiagramType(type),
            title=title,
        )
        self.spec.paper.column = column

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Diagram":
        """
        Load diagram from YAML specification file.

        Parameters
        ----------
        path : str or Path
            Path to YAML file.

        Returns
        -------
        Diagram
            Loaded diagram.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        diagram = cls.__new__(cls)
        diagram.spec = DiagramSpec.from_dict(data)
        return diagram

    @classmethod
    def from_mermaid(
        cls,
        path: Union[str, Path],
        diagram_type: str = "workflow",
    ) -> "Diagram":
        """
        Parse existing Mermaid file and create enhanced Diagram.

        This allows upgrading existing Mermaid files with SciTeX
        paper constraints while preserving the original structure.

        Parameters
        ----------
        path : str or Path
            Path to .mmd file.
        diagram_type : str
            Inferred diagram type.

        Returns
        -------
        Diagram
            Parsed diagram (can be enhanced and re-exported).
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        diagram = cls(type=diagram_type)
        diagram._parse_mermaid(content)
        return diagram

    def _parse_mermaid(self, content: str):
        """Parse Mermaid content to extract structure."""
        # Extract nodes
        # Pattern: ID["Label"] or ID("Label") etc.
        node_pattern = r'(\w+)\s*[\[\(\{<][\"\']?([^"\'\]\)\}>]+)[\"\']?[\]\)\}>]'

        for match in re.finditer(node_pattern, content):
            node_id = match.group(1)
            label = match.group(2).strip()
            # Skip if it looks like a subgraph name
            if node_id.lower() not in ["subgraph", "end", "style", "graph", "direction"]:
                # Check for duplicates
                existing = [n for n in self.spec.nodes if n.id == node_id]
                if not existing:
                    self.spec.nodes.append(NodeSpec(id=node_id, label=label))

        # Extract edges
        # Pattern: A --> B or A -->|label| B
        edge_pattern = r'(\w+)\s*(-->|-.->|-----|---)\s*(?:\|["\']?([^|"\']+)["\']?\|)?\s*(\w+)'

        for match in re.finditer(edge_pattern, content):
            source = match.group(1)
            arrow = match.group(2)
            label = match.group(3)
            target = match.group(4)

            style = "solid"
            if "-.->" in arrow or "-.." in arrow:
                style = "dashed"

            self.spec.edges.append(EdgeSpec(
                source=source,
                target=target,
                label=label.strip() if label else None,
                style=style,
            ))

        # Extract subgraphs as groups
        subgraph_pattern = r'subgraph\s+(\w+)\s*\[?"?([^"\]]*)"?\]?'
        for match in re.finditer(subgraph_pattern, content):
            group_id = match.group(1)
            group_name = match.group(2) or group_id
            self.spec.layout.groups[group_name] = []

    def add_node(
        self,
        id: str,
        label: str,
        shape: str = "box",
        emphasis: str = "normal",
    ):
        """Add a node to the diagram."""
        self.spec.nodes.append(NodeSpec(
            id=id,
            label=label,
            shape=shape,
            emphasis=emphasis,
        ))

    def add_edge(
        self,
        source: str,
        target: str,
        label: Optional[str] = None,
        style: str = "solid",
    ):
        """Add an edge between nodes."""
        self.spec.edges.append(EdgeSpec(
            source=source,
            target=target,
            label=label,
            style=style,
        ))

    def set_group(self, group_name: str, node_ids: list):
        """Define a group of nodes (rendered as subgraph)."""
        self.spec.layout.groups[group_name] = node_ids

    def emphasize(self, *node_ids: str):
        """Mark nodes as emphasized (primary styling)."""
        self.spec.paper.emphasize.extend(node_ids)

    def to_mermaid(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Compile to Mermaid format.

        Parameters
        ----------
        path : str or Path, optional
            If provided, write to file.

        Returns
        -------
        str
            Mermaid source code.
        """
        result = compile_to_mermaid(self.spec)

        if path:
            path = Path(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write(result)

        return result

    def to_graphviz(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Compile to Graphviz DOT format.

        Parameters
        ----------
        path : str or Path, optional
            If provided, write to file.

        Returns
        -------
        str
            Graphviz DOT source code.
        """
        result = compile_to_graphviz(self.spec)

        if path:
            path = Path(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write(result)

        return result

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Export specification as YAML.

        Parameters
        ----------
        path : str or Path, optional
            If provided, write to file.

        Returns
        -------
        str
            YAML specification.
        """
        data = {
            "type": self.spec.type.value,
            "title": self.spec.title,
            "paper": {
                "column": self.spec.paper.column.value if hasattr(self.spec.paper.column, 'value') else self.spec.paper.column,
                "max_width_mm": self.spec.paper.max_width_mm,
                "reading_direction": self.spec.paper.reading_direction,
                "mode": self.spec.paper.mode.value if hasattr(self.spec.paper.mode, 'value') else self.spec.paper.mode,
                "emphasize": self.spec.paper.emphasize,
            },
            "layout": {
                "groups": self.spec.layout.groups,
                "layers": self.spec.layout.layers,
                "layer_gap": self.spec.layout.layer_gap.value,
                "node_gap": self.spec.layout.node_gap.value,
            },
            "nodes": [
                {"id": n.id, "label": n.label, "shape": n.shape, "emphasis": n.emphasis}
                for n in self.spec.nodes
            ],
            "edges": [
                {"from": e.source, "to": e.target, "label": e.label, "style": e.style}
                for e in self.spec.edges
            ],
        }

        result = yaml.dump(data, default_flow_style=False, allow_unicode=True)

        if path:
            path = Path(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write(result)

        return result

    def split(
        self,
        max_nodes: int = 12,
        strategy: str = "by_groups",
        keep_hubs: bool = True,
    ) -> List["Diagram"]:
        """
        Split diagram into multiple figures if too large.

        Parameters
        ----------
        max_nodes : int
            Maximum nodes per figure before splitting.
        strategy : str
            Split strategy: "by_groups" or "by_articulation".
        keep_hubs : bool
            Show hub nodes as ghosts in both parts.

        Returns
        -------
        List[Diagram]
            List of split diagrams (or single diagram if no split needed).

        Examples
        --------
        >>> d = Diagram.from_yaml("large_workflow.yaml")
        >>> parts = d.split(max_nodes=8)
        >>> for i, part in enumerate(parts):
        ...     part.to_mermaid(f"workflow_part_{i+1}.mmd")
        """
        config = SplitConfig(
            enabled=True,
            max_nodes=max_nodes,
            strategy=SplitStrategy(strategy),
            keep_hubs=keep_hubs,
        )

        result = split_diagram(self.spec, config)

        # Wrap each spec in a Diagram object
        diagrams = []
        for fig_spec, label in zip(result.figures, result.labels):
            d = Diagram.__new__(Diagram)
            d.spec = fig_spec
            if label:
                d.spec.title = f"{self.spec.title} ({label})" if self.spec.title else f"Part {label}"
            diagrams.append(d)

        return diagrams
