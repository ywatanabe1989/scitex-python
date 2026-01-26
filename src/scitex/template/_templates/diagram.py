#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/template/_templates/diagram.py
# ----------------------------------------

"""Template for stx.diagram module usage."""

TEMPLATE = {
    "name": "Diagram Module",
    "description": "stx.diagram usage for creating Mermaid and Graphviz diagrams from YAML specs",
    "filename": "diagram_script.py",
    "priority": 7,
    "content": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: {timestamp}
# File: {filepath}

"""
stx.diagram - Scientific Diagrams
=================================

stx.diagram provides:
- YAML-based diagram specifications
- Mermaid and Graphviz output
- Paper-ready presets (workflow, decision, pipeline, scientific)
- Large diagram splitting for multi-column layouts
- PNG/SVG/PDF rendering

Usage Patterns
--------------
1. MCP: diagram_* tools for Claude Code integration
2. YAML spec files for reproducible diagrams
3. Multiple rendering backends (mermaid-cli, graphviz, mermaid.ink)
"""

import scitex as stx

# ============================================================
# Pattern 1: YAML Specification
# ============================================================

"""
YAML Diagram Specification Format:
----------------------------------

# diagram_spec.yaml
metadata:
  title: "Data Processing Pipeline"
  paper_mode: single_column  # or double_column, full_page

preset: workflow  # workflow, decision, pipeline, scientific

nodes:
  - id: input
    label: "Raw Data"
    shape: cylinder
    group: data

  - id: preprocess
    label: "Preprocessing"
    shape: box
    group: processing

edges:
  - from: input
    to: preprocess
    label: "load"

groups:
  - id: data
    label: "Data Layer"
"""

# ============================================================
# Pattern 2: MCP Tools Usage
# ============================================================

"""
MCP Tools for Claude Code:
--------------------------

# Create diagram from YAML file
diagram_create(spec_path="/path/to/diagram.yaml")

# Create from dict specification (see Pattern 5 for dict format)
diagram_create(spec_dict=spec)

# Compile to Mermaid format
diagram_compile_mermaid(spec_path="diagram.yaml", output_path="diagram.mmd")

# Compile to Graphviz DOT
diagram_compile_graphviz(spec_path="diagram.yaml", output_path="diagram.dot")

# Render to image
diagram_render(
    spec_path="diagram.yaml",
    output_path="diagram.png",
    format="png",      # png, svg, pdf
    backend="auto",    # mermaid-cli, graphviz, mermaid.ink, auto
    scale=2.0
)

# List available presets
diagram_list_presets()

# Get preset configuration
diagram_get_preset(preset_name="workflow")

# Get paper layout modes
diagram_get_paper_modes()

# Check rendering backends
diagram_get_backends()
"""

# ============================================================
# Pattern 3: Presets
# ============================================================

"""
Available Presets:
------------------

workflow:
  - Left-to-right flow
  - Rounded boxes
  - Good for process diagrams

decision:
  - Top-down flow
  - Diamond decision nodes
  - Good for flowcharts

pipeline:
  - Left-to-right flow
  - Data cylinders
  - Good for data pipelines

scientific:
  - Top-down flow
  - Clean academic style
  - Good for methods diagrams
"""

# ============================================================
# Pattern 4: Large Diagram Splitting
# ============================================================

"""
Split Large Diagrams:
---------------------
For multi-column layouts, split diagrams into parts:

# Split by groups
diagram_split(
    spec_path="large_diagram.yaml",
    max_nodes_per_part=10,
    strategy="by_groups"
)

# Split by articulation points
diagram_split(
    spec_path="large_diagram.yaml",
    strategy="by_articulation"
)
"""

# ============================================================
# Pattern 5: Python API Usage
# ============================================================

def python_api_example():
    """Direct Python API for diagrams."""
    from scitex.diagram import (
        create_diagram,
        compile_mermaid,
        compile_graphviz,
        render_diagram,
    )

    # Define specification as dict
    # Keys: metadata, preset, nodes, edges
    spec = dict(
        metadata=dict(title="Simple Pipeline"),
        preset="pipeline",
        nodes=[
            dict(id="a", label="Input"),
            dict(id="b", label="Process"),
            dict(id="c", label="Output"),
        ],
        edges=[
            dict(from_="a", to="b"),
            dict(from_="b", to="c"),
        ],
    )

    # Create (returns mermaid and graphviz strings)
    result = create_diagram(spec_dict=spec)
    print("Mermaid:")
    print(result["mermaid"])

    # Compile to file
    compile_mermaid(spec_dict=spec, output_path="pipeline.mmd")

    # Render to PNG
    render_diagram(
        spec_dict=spec,
        output_path="pipeline.png",
        format="png",
        scale=2.0
    )

# ============================================================
# Pattern 6: With @stx.session
# ============================================================

@stx.session
def main(
    CONFIG=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Create research methodology diagram."""
    from scitex.diagram import render_diagram

    # Define methodology using dict() constructor
    spec = dict(
        metadata=dict(
            title="Research Methodology",
            paper_mode="single_column"
        ),
        preset="scientific",
        nodes=[
            dict(id="data", label="Data Collection", shape="cylinder"),
            dict(id="preprocess", label="Preprocessing"),
            dict(id="features", label="Feature Extraction"),
            dict(id="model", label="Model Training"),
            dict(id="eval", label="Evaluation"),
            dict(id="results", label="Results", shape="cylinder"),
        ],
        edges=[
            dict(from_="data", to="preprocess"),
            dict(from_="preprocess", to="features"),
            dict(from_="features", to="model"),
            dict(from_="model", to="eval"),
            dict(from_="eval", to="results"),
        ],
    )

    # Render diagram
    output_path = CONFIG.SDIR_OUT / "methodology.png"
    render_diagram(
        spec_dict=spec,
        output_path=str(output_path),
        format="png",
        scale=2.0
    )

    logger.info("Diagram saved to output directory")
    return 0

# ============================================================
# MCP Tools Reference
# ============================================================

"""
MCP Tools:
----------

Creation:
- diagram_create(spec_dict, spec_path)
- diagram_compile_mermaid(spec_dict, spec_path, output_path)
- diagram_compile_graphviz(spec_dict, spec_path, output_path)

Rendering:
- diagram_render(spec_dict, spec_path, output_path, format, backend, scale)
- diagram_get_backends()

Presets & Layout:
- diagram_list_presets()
- diagram_get_preset(preset_name)
- diagram_get_paper_modes()

Advanced:
- diagram_split(spec_path, max_nodes_per_part, strategy)
"""

if __name__ == "__main__":
    main()
''',
}

__all__ = ["TEMPLATE"]

# EOF
