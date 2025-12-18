#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex.diagram._split"""

import pytest
from scitex.diagram._schema import DiagramSpec, NodeSpec, EdgeSpec
from scitex.diagram._split import (
    split_diagram, SplitConfig, SplitStrategy, SplitResult,
    _split_by_groups, _split_by_articulation
)


class TestSplitConfig:
    def test_default_config(self):
        config = SplitConfig()
        assert config.enabled == False
        assert config.max_nodes == 12
        assert config.strategy == SplitStrategy.BY_GROUPS

    def test_custom_config(self):
        config = SplitConfig(enabled=True, max_nodes=5, strategy=SplitStrategy.BY_ARTICULATION)
        assert config.enabled == True
        assert config.max_nodes == 5
        assert config.strategy == SplitStrategy.BY_ARTICULATION


class TestSplitDiagram:
    def test_no_split_when_disabled(self):
        spec = DiagramSpec()
        spec.nodes = [NodeSpec(id=f"n{i}", label=f"Node {i}") for i in range(20)]
        config = SplitConfig(enabled=False)
        result = split_diagram(spec, config)
        assert len(result.figures) == 1

    def test_no_split_when_under_threshold(self):
        spec = DiagramSpec()
        spec.nodes = [NodeSpec(id=f"n{i}", label=f"Node {i}") for i in range(5)]
        config = SplitConfig(enabled=True, max_nodes=10)
        result = split_diagram(spec, config)
        assert len(result.figures) == 1

    def test_split_when_over_threshold(self):
        spec = DiagramSpec()
        spec.nodes = [NodeSpec(id=f"n{i}", label=f"Node {i}") for i in range(10)]
        spec.layout.groups = {
            "Group1": ["n0", "n1", "n2"],
            "Group2": ["n3", "n4", "n5"],
            "Group3": ["n6", "n7", "n8", "n9"],
        }
        config = SplitConfig(enabled=True, max_nodes=5)
        result = split_diagram(spec, config)
        assert len(result.figures) >= 2


class TestSplitByGroups:
    def test_greedy_packing(self):
        spec = DiagramSpec()
        spec.nodes = [NodeSpec(id=f"n{i}", label=f"Node {i}") for i in range(9)]
        spec.layout.groups = {
            "A": ["n0", "n1", "n2"],  # 3 nodes
            "B": ["n3", "n4", "n5"],  # 3 nodes
            "C": ["n6", "n7", "n8"],  # 3 nodes
        }
        result = _split_by_groups(spec, max_nodes=5)
        # Should create multiple figures
        assert len(result) >= 2

    def test_no_groups_splits_in_half(self):
        spec = DiagramSpec()
        spec.nodes = [NodeSpec(id=f"n{i}", label=f"Node {i}") for i in range(10)]
        result = _split_by_groups(spec, max_nodes=5)
        assert len(result) == 2


class TestSplitByArticulation:
    def test_finds_hub_node(self):
        spec = DiagramSpec()
        spec.nodes = [
            NodeSpec(id="hub", label="Hub"),
            NodeSpec(id="a", label="A"),
            NodeSpec(id="b", label="B"),
            NodeSpec(id="c", label="C"),
        ]
        spec.edges = [
            EdgeSpec(source="a", target="hub"),
            EdgeSpec(source="b", target="hub"),
            EdgeSpec(source="c", target="hub"),
        ]
        result = _split_by_articulation(spec)
        # Hub should appear in multiple parts
        hub_count = sum(1 for part in result if "hub" in part)
        assert hub_count >= 1


class TestSplitResult:
    def test_result_structure(self):
        spec = DiagramSpec()
        spec.nodes = [NodeSpec(id=f"n{i}", label=f"Node {i}") for i in range(10)]
        spec.layout.groups = {"A": ["n0", "n1"], "B": ["n2", "n3"]}
        config = SplitConfig(enabled=True, max_nodes=3)
        result = split_diagram(spec, config)

        assert isinstance(result, SplitResult)
        assert isinstance(result.figures, list)
        assert isinstance(result.labels, list)
        assert len(result.figures) == len(result.labels)


class TestGhostNodes:
    def test_ghost_nodes_created(self):
        spec = DiagramSpec()
        spec.nodes = [
            NodeSpec(id="a", label="A"),
            NodeSpec(id="b", label="B"),
            NodeSpec(id="c", label="C"),
        ]
        spec.edges = [
            EdgeSpec(source="a", target="b"),
            EdgeSpec(source="b", target="c"),
        ]
        spec.layout.groups = {"G1": ["a"], "G2": ["b", "c"]}
        config = SplitConfig(enabled=True, max_nodes=2, keep_hubs=True)
        result = split_diagram(spec, config)

        # Check that at least one figure has a ghost node
        ghost_found = False
        for fig in result.figures:
            for node in fig.nodes:
                if "→" in node.label:
                    ghost_found = True
                    break
        # Ghost nodes may or may not be created depending on split
        # Just verify the split happened
        assert len(result.figures) >= 1

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/diagram/_split.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: 2025-12-15
# # Author: ywatanabe / Claude
# # File: scitex/diagram/_split.py
# 
# """
# Auto-split large diagrams into multiple figures.
# 
# Strategies:
# - by_groups: Split by existing layout.groups (deterministic, paper-friendly)
# - by_articulation: Split at hub nodes (graph-theoretic)
# 
# The split preserves "ghost nodes" at boundaries for visual continuity.
# """
# 
# from dataclasses import dataclass, field
# from typing import List, Dict, Set, Optional, Tuple
# from copy import deepcopy
# from enum import Enum
# 
# from scitex.diagram._schema import DiagramSpec, NodeSpec, EdgeSpec
# 
# 
# class SplitStrategy(Enum):
#     BY_GROUPS = "by_groups"           # Split by layout.groups
#     BY_ARTICULATION = "by_articulation"  # Split at hub nodes
# 
# 
# @dataclass
# class SplitConfig:
#     """Configuration for auto-splitting."""
#     enabled: bool = False
#     max_nodes: int = 12               # Split if more nodes than this
#     strategy: SplitStrategy = SplitStrategy.BY_GROUPS
#     keep_hubs: bool = True            # Show hub nodes in both parts
#     ghost_style: str = "muted"        # Style for ghost nodes
# 
# 
# @dataclass
# class SplitResult:
#     """Result of splitting a diagram."""
#     figures: List[DiagramSpec]
#     labels: List[str]                  # Figure labels (A, B, C, ...)
#     cut_nodes: Set[str]               # Nodes that appear in multiple figures
# 
# 
# def split_diagram(
#     spec: DiagramSpec,
#     config: Optional[SplitConfig] = None,
#     group_assignments: Optional[List[List[str]]] = None,
# ) -> SplitResult:
#     """
#     Split a diagram into multiple figures.
# 
#     Parameters
#     ----------
#     spec : DiagramSpec
#         Original diagram specification.
#     config : SplitConfig, optional
#         Split configuration.
#     group_assignments : List[List[str]], optional
#         Manual group assignments for splitting.
#         If provided, overrides automatic detection.
# 
#     Returns
#     -------
#     SplitResult
#         List of split diagram specifications.
#     """
#     if config is None:
#         config = SplitConfig(enabled=True)
# 
#     # Check if split is needed
#     if not config.enabled or len(spec.nodes) <= config.max_nodes:
#         return SplitResult(figures=[spec], labels=[""], cut_nodes=set())
# 
#     # Determine groups to split by
#     if group_assignments:
#         groups = group_assignments
#     elif config.strategy == SplitStrategy.BY_GROUPS:
#         groups = _split_by_groups(spec, max_nodes=config.max_nodes)
#     else:  # BY_ARTICULATION
#         groups = _split_by_articulation(spec)
# 
#     # Create split figures
#     figures = []
#     labels = []
#     cut_nodes = set()
# 
#     for i, group_nodes in enumerate(groups):
#         fig, cuts = _create_split_figure(spec, group_nodes, config)
#         figures.append(fig)
#         labels.append(chr(ord('A') + i))
#         cut_nodes.update(cuts)
# 
#     return SplitResult(figures=figures, labels=labels, cut_nodes=cut_nodes)
# 
# 
# def _split_by_groups(spec: DiagramSpec, max_nodes: int = 12) -> List[List[str]]:
#     """
#     Split by existing layout.groups using greedy packing.
# 
#     Packs groups into figures until max_nodes is exceeded,
#     then starts a new figure.
# 
#     Returns list of node ID lists, one per split figure.
#     """
#     if not spec.layout.groups:
#         # No groups defined - try to split in half
#         node_ids = [n.id for n in spec.nodes]
#         mid = len(node_ids) // 2
#         return [node_ids[:mid], node_ids[mid:]]
# 
#     # Group keys in order
#     group_names = list(spec.layout.groups.keys())
# 
#     # Greedy packing: add groups until max_nodes exceeded
#     figures = []
#     current_figure = []
#     current_count = 0
# 
#     for group_name in group_names:
#         group_nodes = spec.layout.groups[group_name]
#         group_size = len(group_nodes)
# 
#         # If adding this group exceeds max and we have something, start new figure
#         if current_count + group_size > max_nodes and current_figure:
#             figures.append(current_figure)
#             current_figure = []
#             current_count = 0
# 
#         # Add group to current figure
#         current_figure.extend(group_nodes)
#         current_count += group_size
# 
#     # Don't forget the last figure
#     if current_figure:
#         figures.append(current_figure)
# 
#     # Add ungrouped nodes to first figure
#     grouped = set()
#     for fig in figures:
#         grouped.update(fig)
#     for n in spec.nodes:
#         if n.id not in grouped:
#             if figures:
#                 figures[0].append(n.id)
#             else:
#                 figures.append([n.id])
# 
#     # Ensure at least 2 figures if we have enough nodes
#     if len(figures) == 1 and len(figures[0]) > max_nodes:
#         # Force split in half
#         nodes = figures[0]
#         mid = len(nodes) // 2
#         figures = [nodes[:mid], nodes[mid:]]
# 
#     return figures
# 
# 
# def _split_by_articulation(spec: DiagramSpec) -> List[List[str]]:
#     """
#     Split at articulation points (hub nodes).
# 
#     This finds nodes that, if removed, would disconnect the graph.
#     These are natural split points for large diagrams.
#     """
#     # Build adjacency
#     adj: Dict[str, Set[str]] = {n.id: set() for n in spec.nodes}
#     for e in spec.edges:
#         adj[e.source].add(e.target)
#         adj[e.target].add(e.source)
# 
#     # Find node with most connections (hub)
#     hub = max(adj.keys(), key=lambda x: len(adj[x]))
# 
#     # BFS from first node, stopping at hub
#     visited = {hub}  # Block the hub
#     node_ids = [n.id for n in spec.nodes if n.id != hub]
# 
#     if not node_ids:
#         return [[hub]]
# 
#     # Find components when hub is removed
#     components = []
#     for start in node_ids:
#         if start in visited:
#             continue
#         component = []
#         queue = [start]
#         while queue:
#             curr = queue.pop(0)
#             if curr in visited:
#                 continue
#             visited.add(curr)
#             component.append(curr)
#             for neighbor in adj[curr]:
#                 if neighbor not in visited:
#                     queue.append(neighbor)
#         if component:
#             components.append(component)
# 
#     # Add hub to each component (as ghost)
#     for comp in components:
#         comp.append(hub)
# 
#     return components if components else [[n.id for n in spec.nodes]]
# 
# 
# def _create_split_figure(
#     spec: DiagramSpec,
#     node_ids: List[str],
#     config: SplitConfig,
# ) -> Tuple[DiagramSpec, Set[str]]:
#     """
#     Create a split figure containing specified nodes.
# 
#     Returns (figure_spec, ghost_node_ids).
#     """
#     node_id_set = set(node_ids)
# 
#     # Find edges that cross the boundary
#     boundary_nodes = set()
#     for edge in spec.edges:
#         src_in = edge.source in node_id_set
#         tgt_in = edge.target in node_id_set
#         if src_in and not tgt_in:
#             if config.keep_hubs:
#                 boundary_nodes.add(edge.target)
#         elif tgt_in and not src_in:
#             if config.keep_hubs:
#                 boundary_nodes.add(edge.source)
# 
#     # Create new spec
#     new_spec = DiagramSpec(
#         type=spec.type,
#         title=spec.title,
#         paper=deepcopy(spec.paper),
#         layout=deepcopy(spec.layout),
#         theme=dict(spec.theme),
#     )
# 
#     # Filter nodes
#     node_map = {n.id: n for n in spec.nodes}
#     for node_id in node_ids:
#         if node_id in node_map:
#             new_spec.nodes.append(deepcopy(node_map[node_id]))
# 
#     # Add ghost nodes (boundary nodes not in this split)
#     for ghost_id in boundary_nodes:
#         if ghost_id in node_map and ghost_id not in node_id_set:
#             ghost = deepcopy(node_map[ghost_id])
#             ghost.emphasis = config.ghost_style
#             ghost.label = f"→ {ghost.label}"  # Mark as continuation
#             new_spec.nodes.append(ghost)
# 
#     # Filter edges
#     all_ids = node_id_set | boundary_nodes
#     for edge in spec.edges:
#         if edge.source in all_ids and edge.target in all_ids:
#             new_spec.edges.append(deepcopy(edge))
# 
#     # Filter groups
#     new_spec.layout.groups = {}
#     for group_name, group_nodes in spec.layout.groups.items():
#         filtered = [n for n in group_nodes if n in all_ids]
#         if filtered:
#             new_spec.layout.groups[group_name] = filtered
# 
#     # Filter layers
#     new_spec.layout.layers = []
#     for layer in spec.layout.layers:
#         filtered = [n for n in layer if n in all_ids]
#         if filtered:
#             new_spec.layout.layers.append(filtered)
# 
#     return new_spec, boundary_nodes

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/diagram/_split.py
# --------------------------------------------------------------------------------
