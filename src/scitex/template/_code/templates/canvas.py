#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/template/_templates/canvas.py
# ----------------------------------------

"""Template for stx.canvas figure composition module usage."""

TEMPLATE = {
    "name": "Canvas Module",
    "description": "stx.canvas usage for composing multi-panel figures with panel labels",
    "filename": "canvas_script.py",
    "priority": 8,
    "content": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: {timestamp}
# File: {filepath}

"""
stx.canvas - Figure Composition
===============================

stx.canvas provides:
- Multi-panel figure composition
- Automatic panel labels (A, B, C, ...)
- Millimeter-based positioning
- Export to PNG, PDF, SVG

For single figures: use stx.plt
For composed figures: use stx.canvas

Usage Patterns
--------------
1. MCP: canvas_* tools for Claude Code integration
2. Python API: stx.canvas.* functions
3. Common for: paper figures with multiple panels
"""

import scitex as stx

# ============================================================
# Pattern 1: Basic Composition (via MCP)
# ============================================================

"""
MCP Tools for Claude Code:
--------------------------

# Create a new canvas
canvas_create_canvas(
    parent_dir="/path/to/project",
    canvas_name="figure1",
    width_mm=180,    # Single column width
    height_mm=120
)

# Add panels from images
canvas_add_panel(
    parent_dir="/path/to/project",
    canvas_name="figure1",
    panel_name="panel_a",
    source="/path/to/plot1.png",
    x_mm=0,
    y_mm=0,
    width_mm=85,
    height_mm=60,
    label="A"
)

canvas_add_panel(
    parent_dir="/path/to/project",
    canvas_name="figure1",
    panel_name="panel_b",
    source="/path/to/plot2.png",
    x_mm=90,
    y_mm=0,
    width_mm=85,
    height_mm=60,
    label="B"
)

# List panels
canvas_list_panels(
    parent_dir="/path/to/project",
    canvas_name="figure1"
)

# Export canvas
canvas_export_canvas(
    parent_dir="/path/to/project",
    canvas_name="figure1",
    format="png",    # png, pdf, svg
    dpi=300
)
"""

# ============================================================
# Pattern 2: Common Paper Figure Layouts
# ============================================================

"""
Journal-Specific Widths:
------------------------
Single column: 85-90 mm (Nature, Science)
1.5 column:    114 mm
Double column: 170-180 mm
Full page:     210 mm (A4)

Common Layouts:
---------------

# 2-panel horizontal (A | B)
Panel A: x=0, y=0, w=85, h=60
Panel B: x=90, y=0, w=85, h=60
Canvas:  w=180, h=60

# 2-panel vertical (A over B)
Panel A: x=0, y=0, w=85, h=55
Panel B: x=0, y=60, w=85, h=55
Canvas:  w=85, h=120

# 4-panel grid (2x2)
Panel A: x=0, y=0, w=85, h=55
Panel B: x=90, y=0, w=85, h=55
Panel C: x=0, y=60, w=85, h=55
Panel D: x=90, y=60, w=85, h=55
Canvas:  w=180, h=120
"""

# ============================================================
# Pattern 3: Python API Usage
# ============================================================

def python_api_example():
    """Direct Python API for canvas."""
    from scitex.canvas import (
        create_canvas,
        add_panel,
        list_panels,
        export_canvas,
        remove_panel,
    )

    # Create canvas
    create_canvas(
        parent_dir="./figures",
        canvas_name="fig1",
        width_mm=180,
        height_mm=120
    )

    # Add panels
    add_panel(
        parent_dir="./figures",
        canvas_name="fig1",
        panel_name="scatter",
        source="./plots/scatter.png",
        x_mm=0, y_mm=0,
        width_mm=85, height_mm=55,
        label="A"
    )

    add_panel(
        parent_dir="./figures",
        canvas_name="fig1",
        panel_name="histogram",
        source="./plots/histogram.png",
        x_mm=90, y_mm=0,
        width_mm=85, height_mm=55,
        label="B"
    )

    # List current panels
    panels = list_panels(
        parent_dir="./figures",
        canvas_name="fig1"
    )
    print(f"Panels: {{panels}}")

    # Export
    export_canvas(
        parent_dir="./figures",
        canvas_name="fig1",
        format="png",
        dpi=300
    )

# ============================================================
# Pattern 4: With @stx.session
# ============================================================

@stx.session
def main(
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Create composed figure for paper."""
    import numpy as np
    from scitex.canvas import (
        create_canvas,
        add_panel,
        export_canvas,
    )

    # Generate individual plots first
    plots_dir = CONFIG.SDIR_OUT / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Plot 1: Scatter
    fig, ax = plt.subplots(figsize=(4, 3))
    x = np.random.randn(100)
    y = 2*x + np.random.randn(100)*0.5
    ax.scatter(x, y, alpha=0.6)
    ax.set_xyt("X", "Y", "Correlation")
    stx.io.save(fig, plots_dir / "scatter.png")

    # Plot 2: Histogram
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(np.random.randn(500), bins=30, edgecolor='black')
    ax.set_xyt("Value", "Count", "Distribution")
    stx.io.save(fig, plots_dir / "histogram.png")

    # Plot 3: Time series
    fig, ax = plt.subplots(figsize=(4, 3))
    t = np.linspace(0, 10, 200)
    ax.plot(t, np.sin(t) + np.random.randn(200)*0.1)
    ax.set_xyt("Time (s)", "Amplitude", "Signal")
    stx.io.save(fig, plots_dir / "timeseries.png")

    # Create canvas and compose
    canvas_dir = CONFIG.SDIR_OUT / "canvas"
    canvas_dir.mkdir(exist_ok=True)

    create_canvas(
        parent_dir=str(canvas_dir),
        canvas_name="figure1",
        width_mm=180,
        height_mm=120
    )

    # Add panels
    add_panel(
        parent_dir=str(canvas_dir),
        canvas_name="figure1",
        panel_name="scatter",
        source=str(plots_dir / "scatter.png"),
        x_mm=0, y_mm=0,
        width_mm=85, height_mm=55,
        label="A"
    )

    add_panel(
        parent_dir=str(canvas_dir),
        canvas_name="figure1",
        panel_name="histogram",
        source=str(plots_dir / "histogram.png"),
        x_mm=90, y_mm=0,
        width_mm=85, height_mm=55,
        label="B"
    )

    add_panel(
        parent_dir=str(canvas_dir),
        canvas_name="figure1",
        panel_name="timeseries",
        source=str(plots_dir / "timeseries.png"),
        x_mm=0, y_mm=60,
        width_mm=180, height_mm=55,
        label="C"
    )

    # Export final figure
    export_canvas(
        parent_dir=str(canvas_dir),
        canvas_name="figure1",
        format="png",
        dpi=300
    )

    logger.info(f"Figure saved to {{canvas_dir}}/figure1.png")
    return 0

# ============================================================
# MCP Tools Reference
# ============================================================

"""
MCP Tools:
----------

Canvas Management:
- canvas_create_canvas(parent_dir, canvas_name, width_mm, height_mm)
- canvas_list_canvases(parent_dir)
- canvas_canvas_exists(parent_dir, canvas_name)

Panel Operations:
- canvas_add_panel(parent_dir, canvas_name, panel_name, source, x_mm, y_mm, width_mm, height_mm, label)
- canvas_list_panels(parent_dir, canvas_name)
- canvas_remove_panel(parent_dir, canvas_name, panel_name)

Export:
- canvas_export_canvas(parent_dir, canvas_name, format, dpi, output_path)
"""

if __name__ == "__main__":
    main()
''',
}

__all__ = ["TEMPLATE"]

# EOF
