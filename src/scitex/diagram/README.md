# SciTeX Diagram

Paper-optimized diagram generation with semantic constraints.

## Overview

SciTeX Diagram provides a **semantic layer** above Mermaid/Graphviz that understands paper constraints:
- Column width (single/double)
- Reading direction
- Node emphasis for scientific communication
- Automatic splitting of large diagrams

**Key insight**: LLMs are good at generating *constraints*, not pixel layouts. SciTeX Diagram defines "what this diagram means for a paper" and compiles that to backend-specific layout directives.

## Architecture

```
scitex-diagram.yaml   ← Semantic layer (human/LLM readable)
        ↓
    Compiler (applies paper constraints)
        ↓
workflow.mmd / workflow.dot  ← Backend output
```

## Quick Start

```python
from scitex.diagram import Diagram

# Create programmatically
d = Diagram(type="workflow", title="Data Pipeline")
d.add_node("input", "Raw Data", shape="stadium")
d.add_node("process", "Transform", emphasis="primary")
d.add_node("output", "Results", shape="stadium")
d.add_edge("input", "process")
d.add_edge("process", "output")

# Export
d.to_mermaid("pipeline.mmd")
d.to_graphviz("pipeline.dot")
```

## From YAML Specification

```yaml
# workflow.diagram.yaml
type: workflow
title: SciTeX Figure Lifecycle

paper:
  column: single
  mode: publication      # draft | publication
  emphasize: [figure_bundle, editor]
  return_edges:          # Hide in publication mode
    - [editor, figure_bundle]

layout:
  layer_gap: tight
  layers:                # rank=same constraints
    - [python, savefig]
    - [figure_bundle]
    - [editor, ai_review]

nodes:
  - id: python
    label: Python
    shape: rounded
  - id: figure_bundle
    label: .figure Bundle
    shape: stadium
    emphasis: primary

edges:
  - from: python
    to: savefig
  - from: savefig
    to: figure_bundle
```

```python
d = Diagram.from_yaml("workflow.diagram.yaml")
d.to_mermaid("workflow.mmd")
```

## Paper Modes

### Draft Mode (default)
- Full arrows and labels
- Medium spacing
- All edges visible

### Publication Mode
- Tight spacing (`ranksep=0.3, nodesep=0.2`)
- Return edges hidden (invisible but constrain layout)
- No clusters in Graphviz (uses `rank=same` only)

```yaml
paper:
  mode: publication
  return_edges:
    - [editor, figure_bundle]  # Will be invisible
```

## Auto-Split Large Diagrams

```python
d = Diagram.from_yaml("large_workflow.yaml")

# Split if > 8 nodes per figure
parts = d.split(max_nodes=8, strategy="by_groups")

for i, part in enumerate(parts):
    part.to_mermaid(f"fig_{chr(65+i)}.mmd")  # fig_A.mmd, fig_B.mmd
```

### Split Strategies

| Strategy | Description |
|----------|-------------|
| `by_groups` | Split by layout.groups (deterministic, paper-friendly) |
| `by_articulation` | Split at hub nodes (graph-theoretic) |

Ghost nodes are automatically added at boundaries with `→` prefix.

## Diagram Types

| Type | Direction | Use Case |
|------|-----------|----------|
| `workflow` | LR | Sequential processes |
| `decision` | TB | Decision trees |
| `pipeline` | LR | Data pipelines with stages |
| `hierarchy` | TB | Tree structures |
| `comparison` | LR | Side-by-side comparison |

## Node Shapes

| Shape | Mermaid | Use Case |
|-------|---------|----------|
| `box` | `["label"]` | Default |
| `rounded` | `("label")` | Processes |
| `stadium` | `(["label"])` | Start/End |
| `diamond` | `{"label"}` | Decisions |
| `circle` | `(("label"))` | Events |

## Emphasis Levels

| Level | Color | Use Case |
|-------|-------|----------|
| `normal` | Dark | Default |
| `primary` | Blue | Key nodes |
| `success` | Green | Positive outcomes |
| `warning` | Red | Negative outcomes |
| `muted` | Gray | Secondary/derived |

## Graphviz Output

For tightest layouts, use Graphviz:

```bash
# Render DOT to PNG
dot -Tpng workflow.dot -o workflow.png

# Render DOT to SVG (vector)
dot -Tsvg workflow.dot -o workflow.svg
```

Note: Mermaid doesn't support `rank=same` constraints, so Graphviz produces more compact output.

## API Reference

### Diagram Class

```python
Diagram(type="workflow", title="", column="single")
Diagram.from_yaml(path)
Diagram.from_mermaid(path, diagram_type="workflow")

diagram.add_node(id, label, shape="box", emphasis="normal")
diagram.add_edge(source, target, label=None, style="solid")
diagram.set_group(group_name, node_ids)
diagram.emphasize(*node_ids)

diagram.to_mermaid(path=None) -> str
diagram.to_graphviz(path=None) -> str
diagram.to_yaml(path=None) -> str
diagram.split(max_nodes=12, strategy="by_groups") -> List[Diagram]
```

### Schema Classes

```python
DiagramSpec      # Complete specification
PaperConstraints # column, mode, emphasize, return_edges
LayoutHints      # groups, layers, layer_gap, node_gap
NodeSpec         # id, label, shape, emphasis
EdgeSpec         # source, target, label, style
```
