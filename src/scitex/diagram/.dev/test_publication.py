#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test publication mode vs draft mode."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scitex.diagram import Diagram

OUTPUT_DIR = Path(__file__).parent / "output"


def test_publication_mode():
    """Compare draft vs publication mode."""
    yaml_path = OUTPUT_DIR / "scitex_workflow_publication.diagram.yaml"

    print("=" * 60)
    print("Testing Publication Mode")
    print("=" * 60)

    # Load publication spec
    diagram = Diagram.from_yaml(yaml_path)

    print(f"\nMode: {diagram.spec.paper.mode.value}")
    print(f"Layers defined: {len(diagram.spec.layout.layers)}")
    for i, layer in enumerate(diagram.spec.layout.layers):
        print(f"  Layer {i+1}: {layer}")
    print(f"Return edges to hide: {diagram.spec.paper.return_edges}")

    # Compile to Graphviz (for reference - can be rendered later)
    dot_path = OUTPUT_DIR / "scitex_workflow_publication.dot"
    dot_content = diagram.to_graphviz(dot_path)
    print(f"\n--- Graphviz DOT (Publication) ---")
    print(dot_content)
    print(f"\nGraphviz DOT saved to: {dot_path}")

    # Render to Mermaid
    mmd_path = OUTPUT_DIR / "scitex_workflow_publication.mmd"
    mmd_content = diagram.to_mermaid(mmd_path)
    print(f"\n--- Mermaid (Publication) ---")
    print(mmd_content)

    png_mmd_path = OUTPUT_DIR / "scitex_workflow_publication.png"
    result = subprocess.run(
        [
            "mmdc",
            "-i",
            str(mmd_path),
            "-o",
            str(png_mmd_path),
            "-b",
            "transparent",
            "-w",
            "800",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Mermaid error: {result.stderr}")
    else:
        print(f"Success! Mermaid rendered to: {png_mmd_path}")


def test_draft_mode():
    """Test draft mode for comparison."""
    yaml_path = OUTPUT_DIR / "scitex_workflow.diagram.yaml"

    print("\n" + "=" * 60)
    print("Testing Draft Mode (for comparison)")
    print("=" * 60)

    diagram = Diagram.from_yaml(yaml_path)

    # Force draft mode
    from scitex.diagram._schema import PaperMode

    diagram.spec.paper.mode = PaperMode.DRAFT

    dot_path = OUTPUT_DIR / "scitex_workflow_draft.dot"
    dot_content = diagram.to_graphviz(dot_path)

    # Render
    png_path = OUTPUT_DIR / "scitex_workflow_draft_graphviz.png"
    subprocess.run(
        ["dot", "-Tpng", str(dot_path), "-o", str(png_path)],
        capture_output=True,
    )
    print(f"Draft mode rendered to: {png_path}")


if __name__ == "__main__":
    test_publication_mode()
    test_draft_mode()

    print("\n" + "=" * 60)
    print("Compare the outputs:")
    print("  - scitex_workflow_draft_graphviz.png (with whitespace)")
    print("  - scitex_workflow_publication_graphviz.png (optimized)")
    print("=" * 60)
