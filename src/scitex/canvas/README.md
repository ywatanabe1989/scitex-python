<!-- ---
!-- Timestamp: 2025-12-08
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/vis/README.md
!-- --- -->

# scitex.vis - Canvas-Based Figure Composition

**Compose publication-quality figures from multiple panels**

Schema Version: 2.0.0

---

## Overview

`scitex.vis` provides canvas-based composition of publication figures. A **canvas** represents a complete paper figure (e.g., "Figure 1") that can contain multiple **panels** (A, B, C...).

### Terminology

| Term | Meaning | Example |
|------|---------|---------|
| **Canvas** | Paper figure workspace | "Figure 1" in a publication |
| **Panel** | Single component on canvas | Panel A, B, C... |
| **Figure** | Reserved for matplotlib's `fig` object | `stx.plt` output |

### The SciTeX Ecosystem

```
scitex/
├── scholar/    → Literature & metadata management
├── writer/     → Document generation (LaTeX/manuscripts)
├── plt/        → Plotting (matplotlib wrapper, outputs PNG+JSON+CSV)
└── vis/        → Canvas composition (this module)
```

---

## Quick Start

### Create and Compose a Canvas

```python
import scitex as stx

# 1. Create a canvas
stx.vis.ensure_canvas_directory(
    project_dir="/path/to/project",
    canvas_name="fig1_results"
)

# 2. Add panels from stx.plt outputs
stx.vis.add_panel_from_scitex(
    project_dir="/path/to/project",
    canvas_name="fig1_results",
    panel_name="panel_a",
    source_png="./output/timeseries.png",
    panel_properties={
        "position": {"x_mm": 10, "y_mm": 10},
        "size": {"width_mm": 80, "height_mm": 60},
        "label": {"text": "A"}
    }
)

# 3. Add an image panel
stx.vis.add_panel_from_image(
    project_dir="/path/to/project",
    canvas_name="fig1_results",
    panel_name="panel_b",
    source_image="./external/diagram.png",
    panel_properties={
        "position": {"x_mm": 100, "y_mm": 10},
        "size": {"width_mm": 70, "height_mm": 60},
        "label": {"text": "B"}
    }
)

# 4. Export composed canvas
stx.vis.export_canvas_to_file(
    project_dir="/path/to/project",
    canvas_name="fig1_results",
    output_format="png"
)
```

---

## Directory Structure

Canvas directories use `.canvas` extension for portability and distinguishability:

```
project/scitex/vis/canvases/
└── fig1_results.canvas/          # .canvas extension for bundle
    ├── canvas.json               # Layout, panels, composition
    ├── panels/
    │   ├── panel_a/              # type: scitex (full stx.plt output)
    │   │   ├── panel.json
    │   │   ├── panel.csv
    │   │   └── panel.png
    │   └── panel_b/              # type: image (static)
    │       └── panel.png
    └── exports/
        ├── canvas.png            # Final composed output
        ├── canvas.pdf
        └── canvas.svg
```

The `.canvas` extension makes directories self-documenting, portable, and detectable by `scitex.io`.

---

## Panel Types

| Type | Contents | Editable | Re-renderable |
|------|----------|----------|---------------|
| `scitex` | PNG + JSON + CSV | Full (data, style) | Yes |
| `image` | PNG/JPG/SVG only | Position/size/transform | No |

### Panel Properties

```python
panel_properties = {
    # Position and size (required)
    "position": {"x_mm": 10, "y_mm": 10},
    "size": {"width_mm": 70, "height_mm": 50},

    # Transform
    "z_index": 0,           # Stacking order
    "rotation_deg": 0,      # Rotation (clockwise)
    "opacity": 1.0,         # 0.0 - 1.0
    "flip_h": False,        # Horizontal flip
    "flip_v": False,        # Vertical flip
    "visible": True,

    # Clip (crop)
    "clip": {
        "enabled": False,
        "x_mm": 0, "y_mm": 0,
        "width_mm": None, "height_mm": None
    },

    # Label (A, B, C...)
    "label": {
        "text": "A",
        "position": "top-left",
        "fontsize": 12,
        "fontweight": "bold"
    },

    # Border
    "border": {
        "visible": False,
        "color": "#000000",
        "width_mm": 0.2
    }
}
```

---

## canvas.json Schema

```json
{
  "schema_version": "2.0.0",
  "canvas_name": "fig1_results",

  "size": {
    "width_mm": 180,
    "height_mm": 240
  },

  "background": {
    "color": "#ffffff",
    "grid": false
  },

  "panels": [
    {
      "name": "panel_a",
      "type": "scitex",
      "position": {"x_mm": 10, "y_mm": 10},
      "size": {"width_mm": 80, "height_mm": 60},
      "z_index": 0,
      "label": {"text": "A", "position": "top-left"}
    },
    {
      "name": "panel_b",
      "type": "image",
      "source": "panel.png",
      "position": {"x_mm": 100, "y_mm": 10},
      "size": {"width_mm": 70, "height_mm": 60},
      "label": {"text": "B"}
    }
  ],

  "annotations": [
    {"type": "text", "content": "p < 0.05", "position": {"x_mm": 50, "y_mm": 80}}
  ],

  "data_files": [
    {"path": "panels/panel_a/panel.csv", "hash": "sha256:abc123..."}
  ],

  "metadata": {
    "created_at": "2025-12-08T12:00:00Z",
    "updated_at": "2025-12-08T15:30:00Z"
  }
}
```

---

## API Reference

### Directory Operations

```python
# Create canvas directory structure
canvas_dir = stx.vis.ensure_canvas_directory(project_dir, canvas_name)

# Get canvas path
path = stx.vis.get_canvas_directory_path(project_dir, canvas_name)

# List all canvases
canvases = stx.vis.list_canvas_directories(project_dir)

# Check existence
exists = stx.vis.canvas_directory_exists(project_dir, canvas_name)

# Delete canvas
deleted = stx.vis.delete_canvas_directory(project_dir, canvas_name)
```

### Canvas Operations

```python
# Save canvas.json
stx.vis.save_canvas_json(project_dir, canvas_name, canvas_json)

# Load canvas.json (with hash verification)
canvas_json = stx.vis.load_canvas_json(project_dir, canvas_name)

# Partial update
stx.vis.update_canvas_json(project_dir, canvas_name, {"size": {"width_mm": 200}})

# Get schema version
version = stx.vis.get_canvas_schema_version(project_dir, canvas_name)
```

### Panel Operations

```python
# Add panel from stx.plt output
stx.vis.add_panel_from_scitex(
    project_dir, canvas_name, panel_name,
    source_png="plot.png",
    panel_properties={...}
)

# Add panel from image
stx.vis.add_panel_from_image(
    project_dir, canvas_name, panel_name,
    source_image="image.png",
    panel_properties={...}
)

# Update panel properties
stx.vis.update_panel(project_dir, canvas_name, panel_name, {"opacity": 0.8})

# Remove panel
stx.vis.remove_panel(project_dir, canvas_name, panel_name)

# List panels
panels = stx.vis.list_panels(project_dir, canvas_name)

# Get single panel
panel = stx.vis.get_panel(project_dir, canvas_name, panel_name)

# Reorder panels (z-index)
stx.vis.reorder_panels(project_dir, canvas_name, ["panel_b", "panel_a"])
```

### Data Operations (Hash Verification)

```python
# Compute file hash
hash_str = stx.vis.compute_file_hash(filepath)  # "sha256:abc123..."

# Verify hash
is_valid = stx.vis.verify_data_hash(filepath, expected_hash)

# Verify all data files in canvas
results = stx.vis.verify_all_data_hashes(project_dir, canvas_name)
# {"panels/panel_a/panel.csv": True, ...}
```

### Export Operations

```python
# Export to single format
export_path = stx.vis.export_canvas_to_file(
    project_dir, canvas_name,
    output_format="png",  # png, pdf, svg
    dpi=300
)

# Export to multiple formats
paths = stx.vis.export_canvas_to_multiple_formats(
    project_dir, canvas_name,
    formats=["png", "pdf", "svg"]
)

# List existing exports
exports = stx.vis.list_canvas_exports(project_dir, canvas_name)
```

---

## Integration with stx.plt

`stx.plt` outputs (PNG + JSON + CSV) can be directly added as panels:

```python
# 1. Create figures with stx.plt
fig, ax = stx.plt.subplots()
ax.plot(x, y)
stx.io.save(fig, "./output/timeseries.png")
# Creates: timeseries.png, timeseries.json, timeseries.csv

# 2. Add to canvas as panel
stx.vis.add_panel_from_scitex(
    project_dir, canvas_name, "panel_a",
    source_png="./output/timeseries.png"  # Auto-finds .json and .csv
)
```

---

## Examples

See `examples/vis/` for complete examples:

- `demo_canvas.py` - Canvas composition workflow
- `demo_vis_editor.py` - Interactive editor usage
- `demo_vis_module.py` - Legacy JSON-based workflow

---

## Related Documentation

- [CANVAS_ARCHITECTURE.md](./docs/CANVAS_ARCHITECTURE.md) - Full architecture details
- [scitex.plt](../plt/README.md) - Plotting backend
- [FIGURE_ARCHITECTURE.md](../plt/docs/FIGURE_ARCHITECTURE.md) - Figure output format

---

## Integration with stx.io

Canvas directories (`.canvas`) are first-class citizens in the scitex I/O system:

```python
# Save canvas to a portable directory
stx.io.save(canvas_dict, "/path/to/fig1_results.canvas")

# Load canvas from directory
canvas = stx.io.load("/path/to/fig1_results.canvas")

# Access canvas properties
print(canvas["canvas_name"])
print(canvas["panels"])
```

---

## Constants

```python
stx.vis.SCHEMA_VERSION          # "2.0.0"
stx.vis.CANVAS_EXTENSION        # ".canvas"
stx.vis.NATURE_SINGLE_COLUMN_MM # 89
stx.vis.NATURE_DOUBLE_COLUMN_MM # 183
```

<!-- EOF -->
