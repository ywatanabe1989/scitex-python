<!-- ---
!-- Timestamp: 2025-11-22 10:37:48
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/vis/README.md
!-- --- -->

# scitex.vis - Structured Visualization for Publication Figures

**JSON-based figure specifications for reproducible, publication-quality visualizations**

---

## Overview

`scitex.vis` is the visualization pillar of the SciTeX ecosystem, providing a structured approach to creating publication-quality figures through JSON specifications. It bridges the gap between data and final figures by treating visualization as structured data.

### The SciTeX Ecosystem

```
scitex/
├── scholar/    → Literature & metadata management
├── writer/     → Document generation (LaTeX/manuscripts)
└── vis/        → Visualization & figures (this module)
```

### Why scitex.vis?

Traditional plotting workflows have several pain points:
- **Not version-controllable**: Figures are binary blobs
- **Not reproducible**: Code scattered across notebooks, hard to recreate exact output
- **Not collaborative**: Hard to edit someone else's plotting code
- **Not structured**: No standard way to represent figure specifications

`scitex.vis` solves these by:
- ✅ **JSON-based**: Figures as structured, version-controllable text
- ✅ **Reproducible**: One JSON → One figure, guaranteed
- ✅ **Collaborative**: Easy to edit, review, and share specifications
- ✅ **Structured**: Standard schema for all figure types
- ✅ **Publication-ready**: Built-in templates for Nature, Science, etc.

---

## Quick Start

### Basic Usage

```python
import scitex as stx
import numpy as np

# 1. Create figure using publication template
fig_json = stx.vis.get_template("nature_single", height_mm=100)

# 2. Define your plot
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

fig_json["axes"] = [{
    "row": 0,
    "col": 0,
    "xlabel": "Time (s)",
    "ylabel": "Amplitude",
    "title": "Sine Wave",
    "plots": [{
        "plot_type": "line",
        "data": {"x": x.tolist(), "y": y.tolist()},
        "color": "blue",
        "linewidth": 2,
        "label": "sin(x)"
    }],
    "grid": True,
    "legend": True
}]

# 3. Save the specification
stx.vis.save_figure_json(fig_json, "figure.json")

# 4. Render to matplotlib
fig, axes = stx.vis.build_figure_from_json(fig_json)

# 5. Export to image
stx.vis.export_figure(fig_json, "figure.png", dpi=300)
```

### Project-Based Workflow

```python
# Save to project structure: project/scitex/vis/figs/fig-001.json
stx.vis.save_figure_json_to_project(
    project_dir="/path/to/project",
    figure_id="fig-001",
    fig_json=fig_json
)

# Load from project
fig_json = stx.vis.load_figure_json_from_project(
    project_dir="/path/to/project",
    figure_id="fig-001"
)

# Export directly
stx.vis.export_figure(fig_json, "output/fig-001.png", dpi=300)
```

---

## Architecture

```
scitex.vis/
├── model/              # JSON data models
│   ├── FigureModel     # Top-level figure specification
│   ├── AxesModel       # Subplot configuration
│   ├── PlotModel       # Individual plot (line, scatter, etc.)
│   ├── GuideModel      # Reference lines, spans
│   └── AnnotationModel # Text, arrows, shapes
│
├── backend/            # Rendering engine
│   ├── parser.py       # JSON → Python objects
│   ├── render.py       # Objects → matplotlib figures
│   └── export.py       # Export to PNG/PDF/SVG
│
├── io/                 # Load/save operations
│   ├── load.py         # Load figure JSONs
│   └── save.py         # Save figure JSONs
│
└── utils/              # Utilities
    ├── validate.py     # JSON validation
    └── defaults.py     # Publication templates
```

---

## Publication Templates

### Available Templates

```python
# List all templates
templates = stx.vis.list_templates()
# ['nature_single', 'nature_double', 'science_single', 'a4', 'square', 'presentation']

# Get template dimensions
for name in templates:
    template = stx.vis.get_template(name)
    print(f"{name}: {template['width_mm']} × {template['height_mm']} mm")
```

### Template Dimensions

| Template | Width (mm) | Use Case |
|----------|------------|----------|
| `nature_single` | 89 | Nature single column |
| `nature_double` | 183 | Nature double column |
| `science_single` | 84 | Science single column |
| `a4` | 180 | A4 document figures |
| `square` | 120 | Square aspect ratio |
| `presentation` | 254 | Presentation slides (16:9) |

### Creating Custom Templates

```python
# Start with a template
fig_json = stx.vis.get_template("nature_single")

# Customize dimensions
fig_json["height_mm"] = 120
fig_json["nrows"] = 2
fig_json["ncols"] = 1

# Or create from scratch
fig_json = {
    "width_mm": 180,
    "height_mm": 120,
    "nrows": 1,
    "ncols": 1,
    "dpi": 300,
    "axes": []
}
```

---

## Figure JSON Schema

### Minimal Example

```json
{
  "width_mm": 89,
  "height_mm": 100,
  "nrows": 1,
  "ncols": 1,
  "axes": [
    {
      "row": 0,
      "col": 0,
      "xlabel": "X",
      "ylabel": "Y",
      "plots": [
        {
          "plot_type": "line",
          "data": {"x": [0, 1, 2], "y": [0, 1, 4]},
          "color": "blue"
        }
      ]
    }
  ]
}
```

### Complete Example

```json
{
  "width_mm": 183,
  "height_mm": 120,
  "nrows": 1,
  "ncols": 2,
  "dpi": 300,
  "facecolor": "white",
  "suptitle": "Figure 1: Example Results",
  "suptitle_fontsize": 12,
  "axes": [
    {
      "row": 0,
      "col": 0,
      "xlabel": "Time (s)",
      "ylabel": "Amplitude",
      "title": "Experimental Data",
      "xlim": [0, 10],
      "ylim": [-1, 1],
      "grid": true,
      "legend": true,
      "plots": [
        {
          "plot_type": "line",
          "data": {"x": [...], "y": [...]},
          "color": "blue",
          "linewidth": 2,
          "label": "Measurement"
        },
        {
          "plot_type": "scatter",
          "data": {"x": [...], "y": [...]},
          "color": "red",
          "alpha": 0.6,
          "label": "Control"
        }
      ],
      "guides": [
        {
          "guide_type": "axhline",
          "y": 0,
          "color": "gray",
          "linestyle": "--"
        }
      ],
      "annotations": [
        {
          "annotation_type": "text",
          "text": "Peak",
          "x": 5,
          "y": 0.8,
          "fontsize": 10
        }
      ]
    }
  ]
}
```

---

## Plot Types

### Supported Plot Types

| Plot Type | Description | Required Data Fields |
|-----------|-------------|---------------------|
| `line` | Line plot | `x`, `y` |
| `scatter` | Scatter plot | `x`, `y` |
| `errorbar` | Error bars | `x`, `y`, optional `xerr`, `yerr` |
| `bar` | Bar chart | `x`, `height` (or `y`) |
| `barh` | Horizontal bar chart | `y`, `width` (or `x`) |
| `hist` | Histogram | `x`, optional `bins` |
| `fill_between` | Filled area | `x`, `y1`, `y2` |
| `heatmap` | Heatmap | `z` (or `img`) |
| `imshow` | Image display | `img` (or `z`) |
| `contour` | Contour lines | `x`, `y`, `z` |
| `contourf` | Filled contours | `x`, `y`, `z` |

### Examples

#### Line Plot

```python
{
  "plot_type": "line",
  "data": {"x": [0, 1, 2, 3], "y": [0, 1, 4, 9]},
  "color": "blue",
  "linewidth": 2,
  "linestyle": "-",
  "marker": "o",
  "label": "Quadratic"
}
```

#### Scatter Plot

```python
{
  "plot_type": "scatter",
  "data": {"x": [...], "y": [...]},
  "color": "red",
  "alpha": 0.6,
  "markersize": 50,
  "label": "Data points"
}
```

#### Error Bar Plot

```python
{
  "plot_type": "errorbar",
  "data": {"x": [...], "y": [...]},
  "yerr": [...],  # Can be scalar or array
  "capsize": 5,
  "color": "green",
  "label": "Mean ± SD"
}
```

#### Heatmap

```python
{
  "plot_type": "heatmap",
  "data": {"z": [[...], [...], [...]]},  # 2D array
  "cmap": "viridis",
  "vmin": 0,
  "vmax": 1
}
```

---

## Multi-Subplot Figures

### Creating Subplots

```python
# 2×2 subplot layout
fig_json = stx.vis.get_template("nature_double", height_mm=160)
fig_json["nrows"] = 2
fig_json["ncols"] = 2

# Define each subplot
fig_json["axes"] = [
    # Top-left (row=0, col=0)
    {
        "row": 0,
        "col": 0,
        "title": "Plot A",
        "plots": [...]
    },
    # Top-right (row=0, col=1)
    {
        "row": 0,
        "col": 1,
        "title": "Plot B",
        "plots": [...]
    },
    # Bottom-left (row=1, col=0)
    {
        "row": 1,
        "col": 0,
        "title": "Plot C",
        "plots": [...]
    },
    # Bottom-right (row=1, col=1)
    {
        "row": 1,
        "col": 1,
        "title": "Plot D",
        "plots": [...]
    }
]
```

---

## Guides and Annotations

### Reference Lines and Spans

```python
# Horizontal line
{
  "guide_type": "axhline",
  "y": 0,
  "color": "gray",
  "linestyle": "--",
  "linewidth": 1
}

# Vertical line
{
  "guide_type": "axvline",
  "x": 5,
  "color": "red",
  "alpha": 0.5
}

# Horizontal span (shaded region)
{
  "guide_type": "axhspan",
  "ymin": -0.5,
  "ymax": 0.5,
  "color": "yellow",
  "alpha": 0.3
}

# Vertical span
{
  "guide_type": "axvspan",
  "xmin": 2,
  "xmax": 4,
  "color": "blue",
  "alpha": 0.2
}
```

### Text Annotations

```python
# Simple text
{
  "annotation_type": "text",
  "text": "Important point",
  "x": 5,
  "y": 10,
  "fontsize": 12,
  "color": "red",
  "ha": "center",  # horizontal alignment
  "va": "bottom"   # vertical alignment
}

# Annotate with arrow
{
  "annotation_type": "annotate",
  "text": "Peak value",
  "x": 5,           # Point to annotate
  "y": 10,
  "xytext": [6, 12], # Text position
  "arrowprops": {
    "arrowstyle": "->",
    "color": "black",
    "lw": 1.5
  }
}
```

---

## Advanced Usage

### Using Models Directly

```python
from scitex.vis.model import FigureModel, AxesModel, PlotModel

# Create models programmatically
fig_model = FigureModel(
    width_mm=stx.vis.NATURE_SINGLE_COLUMN_MM,
    height_mm=100,
    nrows=1,
    ncols=1
)

axes_model = AxesModel(
    row=0,
    col=0,
    xlabel="Time",
    ylabel="Signal"
)

plot_model = PlotModel(
    plot_type="line",
    data={"x": [0, 1, 2], "y": [0, 1, 4]},
    color="blue"
)

# Build hierarchy
axes_model.add_plot(plot_model.to_dict())
fig_model.add_axes(axes_model.to_dict())

# Validate
fig_model.validate()

# Convert to JSON and render
fig_json = fig_model.to_dict()
fig, axes = stx.vis.build_figure_from_json(fig_json)
```

### Validation

```python
from scitex.vis.backend import validate_figure_json

try:
    validate_figure_json(fig_json)
    print("✓ Valid figure JSON")
except ValueError as e:
    print(f"✗ Invalid: {e}")
```

### Export Multiple Formats

```python
# Export to PNG, PDF, and SVG simultaneously
paths = stx.vis.backend.export_multiple_formats(
    fig_json=fig_json,
    output_dir="output",
    base_name="figure-01",
    formats=["png", "pdf", "svg"],
    dpi=300,
    auto_crop=True
)

# Returns: {'png': Path(...), 'pdf': Path(...), 'svg': Path(...)}
```

---

## Integration with scitex.plt

`scitex.vis` uses `scitex.plt` as its rendering backend, which means:

1. **All figures maintain mm-exact dimensions**
2. **Automatic metadata embedding** (creation time, dimensions, etc.)
3. **Consistent styling** across all figures
4. **High-quality output** optimized for publications

```python
# This renders through scitex.plt
fig, axes = stx.vis.build_figure_from_json(fig_json)

# You get a FigWrapper with all scitex.plt features
print(type(fig))  # <class 'scitex.plt._subplots._FigWrapper.FigWrapper'>

# Axes are AxisWrapper objects
print(type(axes[0]))  # <class 'scitex.plt._subplots._AxesWrapper.AxisWrapper'>
```

---

## API Reference

### Top-Level Functions

```python
# Templates
stx.vis.get_template(name, **kwargs) -> Dict
stx.vis.list_templates() -> List[str]

# Build & Export
stx.vis.build_figure_from_json(fig_json) -> (fig, axes)
stx.vis.export_figure(fig_json, output_path, fmt=None, dpi=300, **kwargs) -> Path
stx.vis.export_figure_from_file(json_path, output_path, **kwargs) -> Path

# I/O
stx.vis.load_figure_json(path, validate=True) -> Dict
stx.vis.save_figure_json(fig_json, path) -> Path
stx.vis.load_figure_json_from_project(project_dir, figure_id) -> Dict
stx.vis.save_figure_json_to_project(project_dir, figure_id, fig_json) -> Path

# Models
stx.vis.FigureModel
stx.vis.AxesModel
stx.vis.PlotModel
stx.vis.GuideModel
stx.vis.AnnotationModel

# Constants
stx.vis.NATURE_SINGLE_COLUMN_MM  # 89
stx.vis.NATURE_DOUBLE_COLUMN_MM  # 183
```

### Backend Module

```python
from scitex.vis import backend

# Parsing
backend.parse_figure_json(fig_json) -> FigureModel
backend.validate_figure_json(fig_json) -> bool

# Rendering
backend.render_figure(fig_model) -> (fig, axes)
backend.build_figure_from_json(fig_json) -> (fig, axes)

# Export
backend.export_figure(fig_json, output_path, **kwargs) -> Path
backend.export_multiple_formats(fig_json, output_dir, base_name, formats) -> Dict
```

### IO Module

```python
from scitex.vis import io

# Load
io.load_figure_json(path) -> Dict
io.load_figure_json_from_project(project_dir, figure_id) -> Dict
io.load_figure_model(path) -> FigureModel
io.list_figures_in_project(project_dir) -> List[str]

# Save
io.save_figure_json(fig_json, path) -> Path
io.save_figure_json_to_project(project_dir, figure_id, fig_json) -> Path
io.save_figure_model(fig_model, path) -> Path
```

### Utils Module

```python
from scitex.vis import utils

# Templates
utils.get_nature_single_column(height_mm=89, nrows=1, ncols=1) -> Dict
utils.get_nature_double_column(height_mm=120, nrows=1, ncols=1) -> Dict
utils.get_science_single_column(height_mm=84, nrows=1, ncols=1) -> Dict
utils.get_a4_figure(width_mm=180, height_mm=120) -> Dict
utils.get_square_figure(size_mm=120) -> Dict
utils.get_presentation_slide(aspect_ratio="16:9") -> Dict

# Validation
utils.validate_json_structure(fig_json) -> bool
utils.validate_plot_data(plot_data) -> bool
utils.check_schema_version(fig_json) -> str

# Constants
utils.NATURE_SINGLE_COLUMN_MM
utils.NATURE_DOUBLE_COLUMN_MM
utils.SCIENCE_SINGLE_COLUMN_MM
utils.A4_WIDTH_MM
utils.A4_HEIGHT_MM
```

---

## Project Structure

When using project-based workflows, figures are organized as:

```
project/
└── scitex/
    └── vis/
        ├── figs/           # Figure JSON specifications
        │   ├── fig-001.json
        │   ├── fig-002.json
        │   └── ...
        └── export/         # Exported images (optional)
            ├── fig-001.png
            ├── fig-002.pdf
            └── ...
```

This structure:
- ✅ **Version controllable**: JSON files track with your code
- ✅ **Organized**: Clear separation of specs and outputs
- ✅ **Collaborative**: Easy to review and edit
- ✅ **Reproducible**: One JSON → One figure

---

## Comparison with Other Tools

| Feature | scitex.vis | Matplotlib | Plotly | Seaborn |
|---------|-----------|-----------|--------|---------|
| JSON-based specs | ✅ | ❌ | ✅ | ❌ |
| Version controllable | ✅ | ❌ | Partial | ❌ |
| Publication templates | ✅ | ❌ | ❌ | ❌ |
| mm-exact dimensions | ✅ | ❌ | ❌ | ❌ |
| Metadata embedding | ✅ | ❌ | ❌ | ❌ |
| Project structure | ✅ | ❌ | ❌ | ❌ |
| Backend flexibility | ✅ | N/A | ❌ | Depends on mpl |

---

## Future Enhancements

Planned features for future versions:

- [ ] **Interactive editing**: Web UI for figure JSON editing
- [ ] **AI figure generation**: LLM-based figure creation from descriptions
- [ ] **Figure diff/merge**: Git-friendly figure comparison
- [ ] **Style inheritance**: Template inheritance and composition
- [ ] **Figure collections**: Multi-figure documents
- [ ] **Backend plugins**: Support for alternative renderers
- [ ] **Automatic layouts**: Smart subplot arrangement
- [ ] **3D plots**: Support for 3D visualizations

---

## Contributing

When contributing to `scitex.vis`:

1. **Models**: Extend models in `model/` for new features
2. **Renderers**: Add plot types in `backend/render.py`
3. **Templates**: Add publication formats in `utils/defaults.py`
4. **Validation**: Update validators in `utils/validate.py`

---

## License

Part of the SciTeX project. See main LICENSE file.

---

## Related Documentation

- [scitex.plt](../plt/README.md) - Plotting backend
- [scitex.io](../io/README.md) - I/O operations
- [scitex.scholar](../scholar/README.md) - Literature management
- [scitex.writer](../writer/README.md) - Document generation

---

## Examples

See `examples/demo_vis_module.py` for comprehensive examples including:
- Basic figure creation
- Multi-subplot layouts
- Project workflows
- Model validation
- Template usage

Run the demo:
```bash
python examples/demo_vis_module.py
```

---

**Made with ❤️ by the SciTeX team**

*Completing the SciTeX ecosystem: scholar → writer → vis*

<!-- EOF -->