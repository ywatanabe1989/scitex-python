#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test auto-split functionality for large diagrams."""

import sys
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scitex.diagram import Diagram

OUTPUT_DIR = Path(__file__).parent / "output"


def test_split_workflow():
    """Split the workflow diagram into 2 parts."""
    yaml_path = OUTPUT_DIR / "scitex_workflow_publication.diagram.yaml"

    print("=" * 60)
    print("Testing Diagram Split")
    print("=" * 60)

    # Load diagram
    diagram = Diagram.from_yaml(yaml_path)
    print(f"\nOriginal: {len(diagram.spec.nodes)} nodes")

    # Split with low threshold to force split
    parts = diagram.split(max_nodes=5, strategy="by_groups", keep_hubs=True)

    print(f"Split into {len(parts)} parts")

    for i, part in enumerate(parts):
        label = chr(ord('A') + i)
        print(f"\n--- Part {label}: {len(part.spec.nodes)} nodes ---")
        for node in part.spec.nodes:
            ghost = "→" in node.label
            print(f"  {'[ghost] ' if ghost else ''}{node.id}: {node.label}")

        # Render each part
        mmd_path = OUTPUT_DIR / f"workflow_split_{label}.mmd"
        part.to_mermaid(mmd_path)

        png_path = OUTPUT_DIR / f"workflow_split_{label}.png"
        result = subprocess.run(
            ["mmdc", "-i", str(mmd_path), "-o", str(png_path), "-b", "transparent", "-w", "600"],
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"  Rendered: {png_path}")


def test_manual_split():
    """Create a manually optimized 2-figure split."""
    print("\n" + "=" * 60)
    print("Manual 2-Figure Split (Optimized)")
    print("=" * 60)

    # Part A: Creation -> Bundle (with contents)
    part_a = Diagram(type="workflow", title="Figure Creation")
    part_a.add_node("python", "Python", shape="rounded")
    part_a.add_node("savefig", "savefig()", shape="box")
    part_a.add_node("figz", ".figz Bundle", shape="stadium", emphasis="primary")
    part_a.add_node("spec", "spec.json", shape="box")
    part_a.add_node("data", "data.csv", shape="box")
    part_a.add_node("preview", "preview", shape="box", emphasis="muted")

    part_a.add_edge("python", "savefig")
    part_a.add_edge("savefig", "figz")
    part_a.add_edge("figz", "spec", style="dashed")
    part_a.add_edge("figz", "data", style="dashed")
    part_a.add_edge("figz", "preview", style="dashed")

    part_a.set_group("Creation", ["python", "savefig"])
    part_a.set_group("Bundle", ["figz", "spec", "data", "preview"])

    mmd_a = OUTPUT_DIR / "workflow_manual_A.mmd"
    part_a.to_mermaid(mmd_a)
    subprocess.run(["mmdc", "-i", str(mmd_a), "-o", str(OUTPUT_DIR / "workflow_manual_A.png"),
                    "-b", "transparent", "-w", "500"], capture_output=True)
    print(f"Part A rendered: {OUTPUT_DIR / 'workflow_manual_A.png'}")

    # Part B: Bundle -> Editing -> Export
    part_b = Diagram(type="workflow", title="Figure Editing")
    part_b.add_node("figz", ".figz Bundle", shape="stadium", emphasis="primary")
    part_b.add_node("editor", "Editor", shape="rounded", emphasis="primary")
    part_b.add_node("ai", "AI Review", shape="diamond", emphasis="primary")
    part_b.add_node("export", "Export", shape="stadium", emphasis="success")

    part_b.add_edge("figz", "editor")
    part_b.add_edge("editor", "figz", label="changes")
    part_b.add_edge("figz", "ai")
    part_b.add_edge("ai", "figz", label="diffs")
    part_b.add_edge("figz", "export")

    part_b.set_group("Editing", ["editor", "ai"])
    part_b.set_group("Output", ["export"])

    mmd_b = OUTPUT_DIR / "workflow_manual_B.mmd"
    part_b.to_mermaid(mmd_b)
    subprocess.run(["mmdc", "-i", str(mmd_b), "-o", str(OUTPUT_DIR / "workflow_manual_B.png"),
                    "-b", "transparent", "-w", "500"], capture_output=True)
    print(f"Part B rendered: {OUTPUT_DIR / 'workflow_manual_B.png'}")


if __name__ == "__main__":
    test_split_workflow()
    test_manual_split()

    print("\n" + "=" * 60)
    print("Split diagrams created!")
    print("Compare:")
    print("  - workflow_split_A.png / workflow_split_B.png (auto)")
    print("  - workflow_manual_A.png / workflow_manual_B.png (manual)")
    print("=" * 60)
