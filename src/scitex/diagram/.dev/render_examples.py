#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render example diagrams from YAML specs."""

import sys
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scitex.diagram import Diagram

OUTPUT_DIR = Path(__file__).parent / "output"


def render_from_yaml():
    """Render the YAML-defined workflow."""
    yaml_path = OUTPUT_DIR / "scitex_workflow.diagram.yaml"

    print(f"Loading: {yaml_path}")
    diagram = Diagram.from_yaml(yaml_path)

    # Compile to Mermaid
    mmd_path = OUTPUT_DIR / "scitex_workflow.mmd"
    mmd_content = diagram.to_mermaid(mmd_path)
    print(f"\n--- Mermaid Output ---")
    print(mmd_content)

    # Compile to Graphviz
    dot_path = OUTPUT_DIR / "scitex_workflow.dot"
    dot_content = diagram.to_graphviz(dot_path)
    print(f"\n--- Graphviz Output ---")
    print(dot_content)

    # Render PNG
    png_path = OUTPUT_DIR / "scitex_workflow.png"
    print(f"\nRendering to: {png_path}")
    result = subprocess.run(
        ["mmdc", "-i", str(mmd_path), "-o", str(png_path), "-b", "transparent", "-w", "1200"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"mmdc error: {result.stderr}")
    else:
        print(f"Success! PNG saved to {png_path}")


def render_comparison():
    """Render comparison diagram (A vs B layout)."""
    diagram = Diagram(type="comparison", title="Traditional vs SciTeX")

    # Traditional workflow (A)
    diagram.add_node("a_script", "Python Script", shape="rounded")
    diagram.add_node("a_savefig", "plt.savefig()", shape="box")
    diagram.add_node("a_png", "PNG File", shape="stadium", emphasis="warning")
    diagram.add_node("a_lost", "Data Lost", shape="box", emphasis="warning")

    # SciTeX workflow (B)
    diagram.add_node("b_script", "Python Script", shape="rounded")
    diagram.add_node("b_savefig", "scitex.savefig()", shape="box")
    diagram.add_node("b_bundle", ".figz Bundle", shape="stadium", emphasis="success")
    diagram.add_node("b_preserved", "Data Preserved", shape="box", emphasis="success")

    # Edges
    diagram.add_edge("a_script", "a_savefig")
    diagram.add_edge("a_savefig", "a_png")
    diagram.add_edge("a_png", "a_lost")

    diagram.add_edge("b_script", "b_savefig")
    diagram.add_edge("b_savefig", "b_bundle")
    diagram.add_edge("b_bundle", "b_preserved")

    # Groups
    diagram.set_group("Traditional Workflow", ["a_script", "a_savefig", "a_png", "a_lost"])
    diagram.set_group("SciTeX Workflow", ["b_script", "b_savefig", "b_bundle", "b_preserved"])

    # Paper constraints
    diagram.spec.paper.column = "single"

    # Output
    mmd_path = OUTPUT_DIR / "comparison.mmd"
    mmd_content = diagram.to_mermaid(mmd_path)
    print(f"\n--- Comparison Diagram ---")
    print(mmd_content)

    # Render
    png_path = OUTPUT_DIR / "comparison.png"
    subprocess.run(
        ["mmdc", "-i", str(mmd_path), "-o", str(png_path), "-b", "transparent", "-w", "1000"],
        capture_output=True,
    )
    print(f"Rendered: {png_path}")


def render_decision_tree():
    """Render a decision tree diagram."""
    diagram = Diagram(type="decision", title="Figure Format Decision")

    diagram.add_node("start", "New Figure?", shape="diamond")
    diagram.add_node("multipanel", "Multi-panel?", shape="diamond")
    diagram.add_node("use_figz", "Use .figz", shape="stadium", emphasis="primary")
    diagram.add_node("use_pltz", "Use .pltz", shape="stadium", emphasis="primary")
    diagram.add_node("simple", "Simple plot?", shape="diamond")
    diagram.add_node("use_png", "Use PNG", shape="stadium", emphasis="muted")
    diagram.add_node("use_svg", "Use SVG", shape="stadium", emphasis="success")

    diagram.add_edge("start", "multipanel", label="Yes")
    diagram.add_edge("multipanel", "use_figz", label="Yes")
    diagram.add_edge("multipanel", "use_pltz", label="No")
    diagram.add_edge("start", "simple", label="No")
    diagram.add_edge("simple", "use_png", label="Yes")
    diagram.add_edge("simple", "use_svg", label="No")

    # Decision trees are top-to-bottom
    diagram.spec.paper.reading_direction = "top_to_bottom"

    mmd_path = OUTPUT_DIR / "decision_tree.mmd"
    mmd_content = diagram.to_mermaid(mmd_path)
    print(f"\n--- Decision Tree ---")
    print(mmd_content)

    png_path = OUTPUT_DIR / "decision_tree.png"
    subprocess.run(
        ["mmdc", "-i", str(mmd_path), "-o", str(png_path), "-b", "transparent"],
        capture_output=True,
    )
    print(f"Rendered: {png_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("SciTeX Diagram - Rendering Examples")
    print("=" * 60)

    render_from_yaml()
    render_comparison()
    render_decision_tree()

    print("\n" + "=" * 60)
    print("All examples rendered!")
    print("=" * 60)
