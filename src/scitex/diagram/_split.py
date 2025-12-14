#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-15
# Author: ywatanabe / Claude
# File: scitex/diagram/_split.py

"""
Auto-split large diagrams into multiple figures.

Strategies:
- by_groups: Split by existing layout.groups (deterministic, paper-friendly)
- by_articulation: Split at hub nodes (graph-theoretic)

The split preserves "ghost nodes" at boundaries for visual continuity.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from copy import deepcopy
from enum import Enum

from scitex.diagram._schema import DiagramSpec, NodeSpec, EdgeSpec


class SplitStrategy(Enum):
    BY_GROUPS = "by_groups"           # Split by layout.groups
    BY_ARTICULATION = "by_articulation"  # Split at hub nodes


@dataclass
class SplitConfig:
    """Configuration for auto-splitting."""
    enabled: bool = False
    max_nodes: int = 12               # Split if more nodes than this
    strategy: SplitStrategy = SplitStrategy.BY_GROUPS
    keep_hubs: bool = True            # Show hub nodes in both parts
    ghost_style: str = "muted"        # Style for ghost nodes


@dataclass
class SplitResult:
    """Result of splitting a diagram."""
    figures: List[DiagramSpec]
    labels: List[str]                  # Figure labels (A, B, C, ...)
    cut_nodes: Set[str]               # Nodes that appear in multiple figures


def split_diagram(
    spec: DiagramSpec,
    config: Optional[SplitConfig] = None,
    group_assignments: Optional[List[List[str]]] = None,
) -> SplitResult:
    """
    Split a diagram into multiple figures.

    Parameters
    ----------
    spec : DiagramSpec
        Original diagram specification.
    config : SplitConfig, optional
        Split configuration.
    group_assignments : List[List[str]], optional
        Manual group assignments for splitting.
        If provided, overrides automatic detection.

    Returns
    -------
    SplitResult
        List of split diagram specifications.
    """
    if config is None:
        config = SplitConfig(enabled=True)

    # Check if split is needed
    if not config.enabled or len(spec.nodes) <= config.max_nodes:
        return SplitResult(figures=[spec], labels=[""], cut_nodes=set())

    # Determine groups to split by
    if group_assignments:
        groups = group_assignments
    elif config.strategy == SplitStrategy.BY_GROUPS:
        groups = _split_by_groups(spec, max_nodes=config.max_nodes)
    else:  # BY_ARTICULATION
        groups = _split_by_articulation(spec)

    # Create split figures
    figures = []
    labels = []
    cut_nodes = set()

    for i, group_nodes in enumerate(groups):
        fig, cuts = _create_split_figure(spec, group_nodes, config)
        figures.append(fig)
        labels.append(chr(ord('A') + i))
        cut_nodes.update(cuts)

    return SplitResult(figures=figures, labels=labels, cut_nodes=cut_nodes)


def _split_by_groups(spec: DiagramSpec, max_nodes: int = 12) -> List[List[str]]:
    """
    Split by existing layout.groups using greedy packing.

    Packs groups into figures until max_nodes is exceeded,
    then starts a new figure.

    Returns list of node ID lists, one per split figure.
    """
    if not spec.layout.groups:
        # No groups defined - try to split in half
        node_ids = [n.id for n in spec.nodes]
        mid = len(node_ids) // 2
        return [node_ids[:mid], node_ids[mid:]]

    # Group keys in order
    group_names = list(spec.layout.groups.keys())

    # Greedy packing: add groups until max_nodes exceeded
    figures = []
    current_figure = []
    current_count = 0

    for group_name in group_names:
        group_nodes = spec.layout.groups[group_name]
        group_size = len(group_nodes)

        # If adding this group exceeds max and we have something, start new figure
        if current_count + group_size > max_nodes and current_figure:
            figures.append(current_figure)
            current_figure = []
            current_count = 0

        # Add group to current figure
        current_figure.extend(group_nodes)
        current_count += group_size

    # Don't forget the last figure
    if current_figure:
        figures.append(current_figure)

    # Add ungrouped nodes to first figure
    grouped = set()
    for fig in figures:
        grouped.update(fig)
    for n in spec.nodes:
        if n.id not in grouped:
            if figures:
                figures[0].append(n.id)
            else:
                figures.append([n.id])

    # Ensure at least 2 figures if we have enough nodes
    if len(figures) == 1 and len(figures[0]) > max_nodes:
        # Force split in half
        nodes = figures[0]
        mid = len(nodes) // 2
        figures = [nodes[:mid], nodes[mid:]]

    return figures


def _split_by_articulation(spec: DiagramSpec) -> List[List[str]]:
    """
    Split at articulation points (hub nodes).

    This finds nodes that, if removed, would disconnect the graph.
    These are natural split points for large diagrams.
    """
    # Build adjacency
    adj: Dict[str, Set[str]] = {n.id: set() for n in spec.nodes}
    for e in spec.edges:
        adj[e.source].add(e.target)
        adj[e.target].add(e.source)

    # Find node with most connections (hub)
    hub = max(adj.keys(), key=lambda x: len(adj[x]))

    # BFS from first node, stopping at hub
    visited = {hub}  # Block the hub
    node_ids = [n.id for n in spec.nodes if n.id != hub]

    if not node_ids:
        return [[hub]]

    # Find components when hub is removed
    components = []
    for start in node_ids:
        if start in visited:
            continue
        component = []
        queue = [start]
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            component.append(curr)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    queue.append(neighbor)
        if component:
            components.append(component)

    # Add hub to each component (as ghost)
    for comp in components:
        comp.append(hub)

    return components if components else [[n.id for n in spec.nodes]]


def _create_split_figure(
    spec: DiagramSpec,
    node_ids: List[str],
    config: SplitConfig,
) -> Tuple[DiagramSpec, Set[str]]:
    """
    Create a split figure containing specified nodes.

    Returns (figure_spec, ghost_node_ids).
    """
    node_id_set = set(node_ids)

    # Find edges that cross the boundary
    boundary_nodes = set()
    for edge in spec.edges:
        src_in = edge.source in node_id_set
        tgt_in = edge.target in node_id_set
        if src_in and not tgt_in:
            if config.keep_hubs:
                boundary_nodes.add(edge.target)
        elif tgt_in and not src_in:
            if config.keep_hubs:
                boundary_nodes.add(edge.source)

    # Create new spec
    new_spec = DiagramSpec(
        type=spec.type,
        title=spec.title,
        paper=deepcopy(spec.paper),
        layout=deepcopy(spec.layout),
        theme=dict(spec.theme),
    )

    # Filter nodes
    node_map = {n.id: n for n in spec.nodes}
    for node_id in node_ids:
        if node_id in node_map:
            new_spec.nodes.append(deepcopy(node_map[node_id]))

    # Add ghost nodes (boundary nodes not in this split)
    for ghost_id in boundary_nodes:
        if ghost_id in node_map and ghost_id not in node_id_set:
            ghost = deepcopy(node_map[ghost_id])
            ghost.emphasis = config.ghost_style
            ghost.label = f"→ {ghost.label}"  # Mark as continuation
            new_spec.nodes.append(ghost)

    # Filter edges
    all_ids = node_id_set | boundary_nodes
    for edge in spec.edges:
        if edge.source in all_ids and edge.target in all_ids:
            new_spec.edges.append(deepcopy(edge))

    # Filter groups
    new_spec.layout.groups = {}
    for group_name, group_nodes in spec.layout.groups.items():
        filtered = [n for n in group_nodes if n in all_ids]
        if filtered:
            new_spec.layout.groups[group_name] = filtered

    # Filter layers
    new_spec.layout.layers = []
    for layer in spec.layout.layers:
        filtered = [n for n in layer if n in all_ids]
        if filtered:
            new_spec.layout.layers.append(filtered)

    return new_spec, boundary_nodes
