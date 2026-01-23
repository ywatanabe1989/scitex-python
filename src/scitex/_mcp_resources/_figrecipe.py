#!/usr/bin/env python3
# Timestamp: 2026-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_resources/_figrecipe.py
"""Figrecipe integration documentation for stx.plt MCP tools."""

from __future__ import annotations

__all__ = ["register_figrecipe_resources"]

FIGRECIPE_INTEGRATION = """\
# stx.plt - Powered by FigRecipe
=================================

stx.plt provides publication-ready figures with **automatic reproducibility**.
It's built on FigRecipe, which records all matplotlib calls to YAML recipes.

## Using stx.plt in @stx.session
```python
@stx.session
def main(plt=stx.INJECTED, COLORS=stx.INJECTED):
    fig, ax = stx.plt.subplots()

    # Use stx_ prefixed methods for auto CSV export
    ax.stx_line(x, y, color=COLORS.blue, label="data")
    ax.set_xyt("X", "Y", "Title")

    # Save creates: plot.png + plot.csv (data)
    stx.io.save(fig, "plot.png", symlink_to="./data")
    fig.close()
    return 0
```

## Output Files from stx.io.save(fig, ...)
```
script_out/
├── plot.png           # Image file
├── plot.csv           # Extracted plot data (auto-generated)
└── plot.yaml          # FigRecipe recipe (if using fr.save directly)
```

## Tracked Methods (stx_ prefix)
These methods track data for automatic CSV export:
- ax.stx_line(x, y)
- ax.stx_scatter(x, y)
- ax.stx_bar(x, height)
- ax.stx_errorbar(x, y, yerr)
- ax.stx_hist(data, bins)
- ax.stx_boxplot(data)
- ax.stx_violinplot(data)
- ax.stx_imshow(matrix)

## MCP Declarative Spec (via plt_plot tool)

### RECOMMENDED: CSV Column Input
Use CSV files - enables code to write data, MCP to visualize:
```yaml
plots:
  - type: scatter
    data_file: results.csv   # CSV from your analysis
    x: time                   # Column name (string)
    y: measurement            # Column name
    color: blue
```

**Workflow**: Python writes CSV → MCP reads columns → Creates figure

### Alternative: Inline Data (simple cases only)
```yaml
plots:
  - type: line
    x: [1, 2, 3, 4, 5]    # Inline array (less flexible)
    y: [1, 4, 9, 16, 25]
    color: blue
```

### Full Example
```yaml
figure:
  width_mm: 85          # Nature single-column width
  height_mm: 60
plots:
  - type: scatter
    data_file: experiment.csv
    x: time_hours
    y: concentration_mm
    color: blue
    label: "Data"
xlabel: "Time (h)"
ylabel: "Concentration (mM)"
title: "Enzyme Kinetics"
legend: true
stat_annotations:
  - x1: 1
    x2: 3
    p_value: 0.01
    style: stars
```

## Statistical Annotations
```python
# Method 1: Via stx.plt (Python API)
ax.add_stat_annotation(x1=0, x2=1, p_value=0.01, style="stars")

# Method 2: Via MCP spec (declarative)
stat_annotations:
  - x1: 0
    x2: 1
    p_value: 0.01  # Converts to ** automatically
```

## Color Palette (COLORS=stx.INJECTED)
```python
COLORS.blue, COLORS.red, COLORS.green, COLORS.orange
COLORS.purple, COLORS.navy, COLORS.pink, COLORS.brown
```

## For Detailed FigRecipe Documentation
See figrecipe MCP resources directly:
- figrecipe://cheatsheet - Quick reference
- figrecipe://api/core - Full API documentation
- figrecipe://mcp-spec - Declarative spec format

## Supported Plot Types
line, scatter, bar, barh, hist, boxplot, violinplot, imshow, heatmap,
errorbar, fill_between, contour, contourf, pie, stem
"""


def register_figrecipe_resources(mcp) -> None:
    """Register figrecipe integration resource."""

    @mcp.resource("scitex://plt-figrecipe")
    def figrecipe_integration() -> str:
        """stx.plt integration with FigRecipe for reproducible figures."""
        return FIGRECIPE_INTEGRATION


# EOF
