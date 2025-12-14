#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-15
# Author: ywatanabe / Claude
# File: scitex/diagram/.dev/test_poc.py

"""
Proof of concept test for SciTeX Diagram.

Tests:
1. Parse existing Mermaid files
2. Export as YAML spec (semantic layer)
3. Re-compile to Mermaid with paper constraints
4. Compile to Graphviz as alternative backend
"""

import sys
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scitex.diagram import Diagram


def test_parse_existing_mermaid():
    """Test parsing existing mermaid files."""
    examples_dir = Path(__file__).parent.parent.parent.parent.parent / "docs/diagram_examples"

    print("=" * 60)
    print("TEST 1: Parse existing Mermaid files")
    print("=" * 60)

    for mmd_file in sorted(examples_dir.glob("*.mmd")):
        print(f"\n--- {mmd_file.name} ---")
        diagram = Diagram.from_mermaid(mmd_file, diagram_type="workflow")

        print(f"  Nodes: {len(diagram.spec.nodes)}")
        for node in diagram.spec.nodes[:5]:
            print(f"    - {node.id}: {node.label[:30]}...")
        if len(diagram.spec.nodes) > 5:
            print(f"    ... and {len(diagram.spec.nodes) - 5} more")

        print(f"  Edges: {len(diagram.spec.edges)}")


def test_yaml_roundtrip():
    """Test YAML export and import."""
    examples_dir = Path(__file__).parent.parent.parent.parent.parent / "docs/diagram_examples"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("TEST 2: YAML roundtrip")
    print("=" * 60)

    # Parse existing mermaid
    mmd_file = examples_dir / "05_workflow.mmd"
    diagram = Diagram.from_mermaid(mmd_file, diagram_type="workflow")

    # Add paper constraints
    diagram.spec.paper.column = "single"
    diagram.spec.paper.reading_direction = "left_to_right"
    diagram.spec.paper.emphasize = ["FIGZ", "EDIT", "AI"]  # Emphasize key nodes

    # Export to YAML
    yaml_path = output_dir / "05_workflow.diagram.yaml"
    yaml_content = diagram.to_yaml(yaml_path)
    print(f"\nExported YAML spec to: {yaml_path}")
    print("--- YAML Preview (first 40 lines) ---")
    print("\n".join(yaml_content.split("\n")[:40]))


def test_compile_with_constraints():
    """Test compilation with paper constraints."""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("TEST 3: Compile with paper constraints")
    print("=" * 60)

    # Create a simple diagram programmatically
    diagram = Diagram(type="workflow", title="Data Processing Pipeline")

    # Add nodes
    diagram.add_node("input", "Raw Data", shape="stadium")
    diagram.add_node("clean", "Preprocessing", emphasis="normal")
    diagram.add_node("transform", "Feature Extraction", emphasis="primary")
    diagram.add_node("model", "ML Model", emphasis="primary")
    diagram.add_node("output", "Results", shape="stadium")

    # Add edges
    diagram.add_edge("input", "clean")
    diagram.add_edge("clean", "transform")
    diagram.add_edge("transform", "model")
    diagram.add_edge("model", "output")

    # Group related nodes
    diagram.set_group("Processing", ["clean", "transform"])
    diagram.set_group("Analysis", ["model"])

    # Set paper constraints
    diagram.spec.paper.column = "double"  # Half-width figure
    diagram.spec.paper.reading_direction = "left_to_right"

    # Compile to Mermaid
    mmd_out = output_dir / "pipeline_double_col.mmd"
    mmd_content = diagram.to_mermaid(mmd_out)
    print(f"\nMermaid output (double column):")
    print("-" * 40)
    print(mmd_content)

    # Compile to Graphviz
    dot_out = output_dir / "pipeline_double_col.dot"
    dot_content = diagram.to_graphviz(dot_out)
    print(f"\nGraphviz output (double column):")
    print("-" * 40)
    print(dot_content)

    # Now try single column (wider)
    diagram.spec.paper.column = "single"
    mmd_single = diagram.to_mermaid(output_dir / "pipeline_single_col.mmd")
    print(f"\nMermaid output (single column - notice direction change):")
    print("-" * 40)
    print(mmd_single)


def test_enhance_existing():
    """Test enhancing existing mermaid with constraints."""
    examples_dir = Path(__file__).parent.parent.parent.parent.parent / "docs/diagram_examples"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("TEST 4: Enhance existing Mermaid")
    print("=" * 60)

    # Load existing
    original = examples_dir / "01_opaque_figures.mmd"
    diagram = Diagram.from_mermaid(original, diagram_type="comparison")

    print(f"\nOriginal: {original}")
    print(f"Parsed {len(diagram.spec.nodes)} nodes, {len(diagram.spec.edges)} edges")

    # Add emphasis to key nodes
    diagram.emphasize("A3", "B3")  # Highlight the key comparison points

    # Re-compile
    enhanced = diagram.to_mermaid(output_dir / "01_opaque_figures_enhanced.mmd")
    print("\nEnhanced Mermaid (with emphasis styles):")
    print("-" * 40)
    print(enhanced)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SciTeX Diagram - Proof of Concept")
    print("=" * 60)

    test_parse_existing_mermaid()
    test_yaml_roundtrip()
    test_compile_with_constraints()
    test_enhance_existing()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
