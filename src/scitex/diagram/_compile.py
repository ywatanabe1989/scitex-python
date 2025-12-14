#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-15
# Author: ywatanabe / Claude
# File: scitex/diagram/_compile.py

"""
Compilers from DiagramSpec to backend formats (Mermaid, Graphviz).

The compiler applies paper constraints to generate backend-specific
layout directives. This is where domain knowledge about "good paper figures"
gets encoded.
"""

import json
from typing import Optional
from scitex.diagram._schema import (
    DiagramSpec, DiagramType, ColumnLayout, SpacingLevel, PaperMode
)
from scitex.diagram._presets import get_preset, DiagramPreset


def compile_to_mermaid(
    spec: DiagramSpec,
    preset: Optional[DiagramPreset] = None
) -> str:
    """
    Compile DiagramSpec to Mermaid format with paper-optimized settings.

    Parameters
    ----------
    spec : DiagramSpec
        The semantic diagram specification.
    preset : DiagramPreset, optional
        Override preset (default: inferred from spec.type).

    Returns
    -------
    str
        Mermaid diagram source code.
    """
    if preset is None:
        preset = get_preset(spec.type.value)

    lines = []

    # Theme initialization
    theme_vars = {**preset.mermaid_theme, **spec.theme}
    theme_json = json.dumps({"theme": "base", "themeVariables": theme_vars})
    lines.append(f"%%{{init: {theme_json}}}%%")

    # Determine direction based on paper constraints
    direction = preset.mermaid_direction
    if spec.paper.reading_direction == "top_to_bottom":
        direction = "TB"
    elif spec.paper.column == ColumnLayout.DOUBLE:
        # Double column prefers vertical to save horizontal space
        direction = "TB"

    lines.append(f"graph {direction}")

    # Build node ID to spec mapping
    node_map = {n.id: n for n in spec.nodes}

    # Generate subgraphs for groups
    indent = "    "
    for group_name, group_nodes in spec.layout.groups.items():
        lines.append(f'{indent}subgraph {_sanitize_id(group_name)}["{group_name}"]')
        for node_id in group_nodes:
            if node_id in node_map:
                node = node_map[node_id]
                lines.append(f"{indent}{indent}{_mermaid_node(node, preset)}")
        lines.append(f"{indent}end")

    # Generate standalone nodes (not in any group)
    grouped_nodes = set()
    for group_nodes in spec.layout.groups.values():
        grouped_nodes.update(group_nodes)

    for node in spec.nodes:
        if node.id not in grouped_nodes:
            lines.append(f"{indent}{_mermaid_node(node, preset)}")

    # Generate edges
    for edge in spec.edges:
        edge_str = _mermaid_edge(edge)
        lines.append(f"{indent}{edge_str}")

    # Generate styles for emphasized nodes
    for node in spec.nodes:
        if node.emphasis != "normal" or node.id in spec.paper.emphasize:
            emphasis = "primary" if node.id in spec.paper.emphasize else node.emphasis
            style = preset.emphasis_styles.get(emphasis, {})
            if style:
                style_parts = [f"{k}:{v}" for k, v in style.items()]
                lines.append(f"{indent}style {_sanitize_id(node.id)} {','.join(style_parts)}")

    return "\n".join(lines)


def compile_to_graphviz(
    spec: DiagramSpec,
    preset: Optional[DiagramPreset] = None
) -> str:
    """
    Compile DiagramSpec to Graphviz DOT format.

    Parameters
    ----------
    spec : DiagramSpec
        The semantic diagram specification.
    preset : DiagramPreset, optional
        Override preset.

    Returns
    -------
    str
        Graphviz DOT source code.
    """
    if preset is None:
        preset = get_preset(spec.type.value)

    is_publication = spec.paper.mode == PaperMode.PUBLICATION
    lines = []

    # Determine direction
    rankdir = preset.graphviz_rankdir
    if spec.paper.reading_direction == "top_to_bottom":
        rankdir = "TB"
    elif spec.paper.column == ColumnLayout.DOUBLE:
        rankdir = "TB"

    # Get spacing - publication mode uses tight spacing
    if is_publication:
        spacing = preset.spacing_map.get("tight", {})
    else:
        spacing = preset.spacing_map.get(spec.layout.layer_gap.value, {})
    ranksep = spacing.get("ranksep", preset.graphviz_ranksep)
    nodesep = spacing.get("nodesep", preset.graphviz_nodesep)

    lines.append("digraph G {")
    lines.append(f"    rankdir={rankdir};")
    lines.append(f"    ranksep={ranksep};")
    lines.append(f"    nodesep={nodesep};")
    lines.append("    splines=ortho;")  # Orthogonal edges for cleaner look
    lines.append('    node [fontname="Helvetica", fontsize=10];')
    lines.append('    edge [fontname="Helvetica", fontsize=9];')
    lines.append("")

    # Node map
    node_map = {n.id: n for n in spec.nodes}

    # Build return edges set for publication mode
    return_edge_set = set()
    for e in spec.paper.return_edges:
        if len(e) >= 2:
            return_edge_set.add((e[0], e[1]))

    # Generate subgraphs (without clusters for tighter layout in publication)
    if is_publication and spec.layout.layers:
        # In publication mode with layers, skip clusters - use rank=same instead
        for node in spec.nodes:
            lines.append(f"    {_graphviz_node(node, preset, spec.paper.emphasize)}")
    else:
        # Draft mode: use clusters for visual grouping
        cluster_idx = 0
        for group_name, group_nodes in spec.layout.groups.items():
            lines.append(f'    subgraph cluster_{cluster_idx} {{')
            lines.append(f'        label="{group_name}";')
            for node_id in group_nodes:
                if node_id in node_map:
                    node = node_map[node_id]
                    lines.append(f"        {_graphviz_node(node, preset, spec.paper.emphasize)}")
            lines.append("    }")
            cluster_idx += 1

        # Standalone nodes
        grouped_nodes = set()
        for group_nodes in spec.layout.groups.values():
            grouped_nodes.update(group_nodes)

        for node in spec.nodes:
            if node.id not in grouped_nodes:
                lines.append(f"    {_graphviz_node(node, preset, spec.paper.emphasize)}")

    lines.append("")

    # Rank constraints from layers (CRITICAL for minimizing whitespace)
    for layer in spec.layout.layers:
        if layer:
            node_ids = "; ".join(_sanitize_id(n) for n in layer)
            lines.append(f"    {{ rank=same; {node_ids}; }}")

    lines.append("")

    # Edges - handle return edges in publication mode
    for edge in spec.edges:
        edge_key = (edge.source, edge.target)
        if is_publication and edge_key in return_edge_set:
            # Make return edges invisible in publication mode
            lines.append(f"    {_graphviz_edge_with_style(edge, invisible=True)}")
        else:
            lines.append(f"    {_graphviz_edge(edge)}")

    lines.append("}")

    return "\n".join(lines)


def _sanitize_id(s: str) -> str:
    """Make string safe for use as node ID."""
    import re
    # Remove or replace problematic characters for Mermaid/Graphviz
    s = re.sub(r'[^\w]', '_', s)  # Replace non-word chars with _
    s = re.sub(r'_+', '_', s)      # Collapse multiple underscores
    s = s.strip('_')               # Remove leading/trailing underscores
    return s or "node"


def _mermaid_node(node, preset: DiagramPreset) -> str:
    """Generate Mermaid node definition."""
    shape_template = preset.mermaid_shapes.get(node.shape, '["__LABEL__"]')
    shape_str = shape_template.replace("__LABEL__", node.label)
    return f"{_sanitize_id(node.id)}{shape_str}"


def _mermaid_edge(edge) -> str:
    """Generate Mermaid edge definition."""
    arrow = "-->" if edge.arrow == "normal" else "---"
    if edge.style == "dashed":
        arrow = "-.->" if edge.arrow == "normal" else "-.-"
    elif edge.style == "dotted":
        arrow = "..>" if edge.arrow == "normal" else "..."

    src = _sanitize_id(edge.source)
    tgt = _sanitize_id(edge.target)

    if edge.label:
        return f'{src} {arrow}|"{edge.label}"| {tgt}'
    return f"{src} {arrow} {tgt}"


def _graphviz_node(node, preset: DiagramPreset, emphasize: list) -> str:
    """Generate Graphviz node definition."""
    shape = preset.graphviz_shapes.get(node.shape, "box")

    # Get emphasis style
    emphasis_key = "primary" if node.id in emphasize else node.emphasis
    style = preset.emphasis_styles.get(emphasis_key, {})

    attrs = [f'label="{node.label}"', f'shape={shape}']

    # Collect style values (filled, rounded, etc.) - combine with comma
    styles = []
    if style.get("fill"):
        attrs.append(f'fillcolor="{style["fill"]}"')
        styles.append("filled")
    if style.get("stroke"):
        attrs.append(f'color="{style["stroke"]}"')
    if node.shape == "rounded":
        styles.append("rounded")

    # Output style once with comma-separated values
    if styles:
        attrs.append(f'style="{",".join(styles)}"')

    return f'{_sanitize_id(node.id)} [{", ".join(attrs)}];'


def _graphviz_edge(edge) -> str:
    """Generate Graphviz edge definition."""
    src = _sanitize_id(edge.source)
    tgt = _sanitize_id(edge.target)

    attrs = []
    if edge.label:
        attrs.append(f'label="{edge.label}"')
    if edge.style == "dashed":
        attrs.append("style=dashed")
    elif edge.style == "dotted":
        attrs.append("style=dotted")
    if edge.arrow == "none":
        attrs.append("arrowhead=none")

    if attrs:
        return f'{src} -> {tgt} [{", ".join(attrs)}];'
    return f"{src} -> {tgt};"


def _graphviz_edge_with_style(edge, invisible: bool = False) -> str:
    """Generate Graphviz edge with optional invisible style."""
    src = _sanitize_id(edge.source)
    tgt = _sanitize_id(edge.target)

    attrs = []
    if invisible:
        attrs.append("style=invis")
        # Invisible edges still constrain layout
        attrs.append("constraint=true")
    else:
        if edge.label:
            attrs.append(f'label="{edge.label}"')
        if edge.style == "dashed":
            attrs.append("style=dashed")
        elif edge.style == "dotted":
            attrs.append("style=dotted")
        if edge.arrow == "none":
            attrs.append("arrowhead=none")

    if attrs:
        return f'{src} -> {tgt} [{", ".join(attrs)}];'
    return f"{src} -> {tgt};"
