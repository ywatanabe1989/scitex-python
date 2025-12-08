<!-- ---
!-- Timestamp: 2025-12-08 16:05:39
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/vis/docs/CANVAS_ARCHITECTURE.md
!-- --- -->

# Canvas Architecture for scitex.vis

## Terminology

| Term       | Meaning                                     | Example                     |
|------------|---------------------------------------------|-----------------------------|
| **Canvas** | A paper figure workspace managed by `/vis/` | "Figure 1" in a publication |
| **Panel**  | A single component placed on canvas         | Panel A, B, C...            |
| **Figure** | Reserved for matplotlib's `fig` object      | `stx.plt` output            |

## Directory Structure

Canvas directories use `.canvas` extension and are standalone (no nested structure required):

```
project/
├── fig1_neural_results.canvas/       # Standalone .canvas directory
│   ├── canvas.json                   # Layout, panels, composition settings
│   ├── canvas.png                    # Composed output (auto-generated)
│   ├── canvas.pdf
│   ├── canvas.svg
│   └── panels/
│       ├── panel_a/                  # type: scitex (symlinks by default)
│       │   ├── panel.json -> ../../plots/plot.json
│       │   ├── panel.csv -> ../../plots/plot.csv
│       │   └── panel.png -> ../../plots/plot.png
│       └── panel_b/                  # type: image
│           └── panel.png -> ../../images/diagram.png
├── fig2_supplementary.canvas/        # Another standalone canvas
│   └── ...
└── plots/                            # Source files (symlink targets)
    ├── plot.png
    ├── plot.json
    └── plot.csv
```

Panel files use symlinks by default (`bundle=False`). Use `bundle=True` to copy files for portable archives.

The `.canvas` extension:
- Makes directories self-documenting (clearly a canvas bundle)
- Enables portability (can be moved/copied as a unit)
- Is detectable by `scitex.io` for automated handling

## Panel Types

| Type     | Contents         | Editable                | Re-renderable |
|----------|------------------|-------------------------|---------------|
| `scitex` | JSON + CSV + PNG | Full (data, style)      | Yes           |
| `image`  | PNG/JPG/SVG only | Position/size/transform | No            |

## canvas.json Schema

```json
{
  "schema_version": "2.0.0",
  "canvas_name": "fig1_neural_results",

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
      "size": {"width_mm": 70, "height_mm": 50},
      "z_index": 0,
      "rotation_deg": 0,
      "clip": {
        "enabled": false,
        "x_mm": 0,
        "y_mm": 0,
        "width_mm": null,
        "height_mm": null
      },
      "opacity": 1.0,
      "flip_h": false,
      "flip_v": false,
      "visible": true,
      "label": {
        "text": "A",
        "position": "top-left",
        "fontsize": 12,
        "fontweight": "bold"
      },
      "border": {
        "visible": false,
        "color": "#000000",
        "width_mm": 0.2
      }
    },
    {
      "name": "panel_b",
      "type": "image",
      "source": "panel.png",
      "position": {"x_mm": 100, "y_mm": 10},
      "size": {"width_mm": 70, "height_mm": 50},
      "z_index": 1,
      "rotation_deg": 0,
      "clip": {"enabled": false},
      "opacity": 1.0,
      "flip_h": false,
      "flip_v": false,
      "visible": true,
      "label": {"text": "B", "position": "top-left"}
    }
  ],

  "annotations": [
    {
      "type": "text",
      "content": "p < 0.05",
      "position": {"x_mm": 50, "y_mm": 60},
      "fontsize": 8
    },
    {
      "type": "arrow",
      "start": {"x_mm": 30, "y_mm": 70},
      "end": {"x_mm": 50, "y_mm": 55}
    },
    {
      "type": "bracket",
      "start": {"x_mm": 10, "y_mm": 80},
      "end": {"x_mm": 80, "y_mm": 80}
    }
  ],

  "title": {
    "text": "",
    "position": {"x_mm": 90, "y_mm": 5},
    "fontsize": 14
  },

  "data_files": [
    {
      "path": "panels/panel_a/panel.csv",
      "hash": "sha256:abc123..."
    }
  ],

  "metadata": {
    "created_at": "2025-12-08T12:00:00Z",
    "updated_at": "2025-12-08T15:30:00Z",
    "author": "user",
    "description": "Neural activity results for Figure 1"
  },

  "manual_overrides": {}
}
```

## Panel Properties

### Transform Properties

| Property       | Type                  | Default  | Purpose                            |
|----------------|-----------------------|----------|------------------------------------|
| `position`     | {x_mm, y_mm}          | required | Position on canvas                 |
| `size`         | {width_mm, height_mm} | required | Panel dimensions                   |
| `z_index`      | int                   | 0        | Stacking order (higher = on top)   |
| `rotation_deg` | float                 | 0        | Rotation around center (clockwise) |
| `opacity`      | float                 | 1.0      | Transparency (0.0 - 1.0)           |
| `flip_h`       | bool                  | false    | Horizontal flip                    |
| `flip_v`       | bool                  | false    | Vertical flip                      |
| `visible`      | bool                  | true     | Show/hide panel                    |

### Clip Properties

| Property         | Type  | Default | Purpose                          |
|------------------|-------|---------|----------------------------------|
| `clip.enabled`   | bool  | false   | Enable cropping                  |
| `clip.x_mm`      | float | 0       | Crop start X (from panel origin) |
| `clip.y_mm`      | float | 0       | Crop start Y                     |
| `clip.width_mm`  | float | null    | Crop width (null = to edge)      |
| `clip.height_mm` | float | null    | Crop height (null = to edge)     |

### Label Properties

| Property           | Type   | Default    | Purpose                    |
|--------------------|--------|------------|----------------------------|
| `label.text`       | string | ""         | Label text (A, B, C...)    |
| `label.position`   | string | "top-left" | Position relative to panel |
| `label.fontsize`   | int    | 12         | Font size in points        |
| `label.fontweight` | string | "bold"     | Font weight                |

### Border Properties

| Property          | Type   | Default   | Purpose          |
|-------------------|--------|-----------|------------------|
| `border.visible`  | bool   | false     | Show border      |
| `border.color`    | string | "#000000" | Border color     |
| `border.width_mm` | float  | 0.2       | Border thickness |

## Annotation Types

| Type        | Properties                         | Purpose          |
|-------------|------------------------------------|------------------|
| `text`      | content, position, fontsize, color | Text overlay     |
| `arrow`     | start, end, color, width           | Arrow annotation |
| `bracket`   | start, end, color, width           | Bracket/brace    |
| `line`      | start, end, color, width, style    | Line annotation  |
| `rectangle` | position, size, color, fill        | Rectangle shape  |

## Core API (scitex.vis)

### Primary API (minimal, reusable, flexible)

```python
# Canvas operations
stx.vis.create_canvas(parent_dir, canvas_name) -> Path
stx.vis.get_canvas_path(parent_dir, canvas_name) -> Path
stx.vis.canvas_exists(parent_dir, canvas_name) -> bool
stx.vis.list_canvases(parent_dir) -> List[str]
stx.vis.delete_canvas(parent_dir, canvas_name) -> bool

# Panel operations (symlinks by default, bundle=True for copies)
stx.vis.add_panel(parent_dir, canvas_name, panel_name, source,
                  position=(x, y), size=(w, h), label="A",
                  bundle=False, **kwargs) -> Path
stx.vis.update_panel(parent_dir, canvas_name, panel_name, updates) -> Dict
stx.vis.remove_panel(parent_dir, canvas_name, panel_name) -> bool
stx.vis.list_panels(parent_dir, canvas_name) -> List[Dict]

# Export (auto-called by stx.io.save)
stx.vis.export_canvas(parent_dir, canvas_name, output_format="png") -> Path

# Data integrity
stx.vis.verify_data(parent_dir, canvas_name) -> Dict[str, bool]

# Editor
stx.vis.edit(json_path)
```

### Advanced API (via submodules)

```python
# stx.vis.io - Low-level I/O operations
# stx.vis.model - Data models (FigureModel, AxesModel, etc.)
# stx.vis.backend - JSON → matplotlib rendering
# stx.vis.utils - Templates and validation
```

## Django Integration (scitex-cloud)

Django views are thin wrappers:

```python
# apps/vis_app/views.py
import scitex as stx

def list_canvases(request, project_id):
    project_path = get_project_path(project_id)
    canvases = stx.vis.io.list_canvas_directories(project_path)
    return JsonResponse({"canvases": canvases})

def get_canvas(request, project_id, canvas_name):
    project_path = get_project_path(project_id)
    canvas_json = stx.vis.io.load_canvas_json(project_path, canvas_name)
    return JsonResponse(canvas_json)
```

## Integration with stx.io

Canvas directories are first-class citizens in the scitex I/O system:

```python
import scitex as stx

# Save canvas to a portable directory
stx.io.save(canvas_dict, "/path/to/fig1_results.canvas")

# Load canvas from directory
canvas = stx.io.load("/path/to/fig1_results.canvas")
# Returns dict with canvas.json content + '_canvas_dir' reference

# Load canvas with panel images in memory
canvas = stx.io.load("/path/to/fig1_results.canvas", load_panels=True)
# Each panel dict now has '_image' key with numpy array
```

## Summary

| Aspect           | Decision                                                   |
|------------------|------------------------------------------------------------|
| Directory path   | `{parent_dir}/{canvas_name}.canvas/`                       |
| Directory ext    | `.canvas` extension for portability and distinguishability |
| Naming           | Descriptive names (e.g., `fig1_neural_results`)            |
| Panel types      | `scitex` (full) or `image` (static)                        |
| Data integrity   | SHA256 hash verification                                   |
| Export structure | Flat (`exports/canvas.png`, `exports/canvas.pdf`)          |
| Manual overrides | Stored in `canvas.json` under `manual_overrides` key       |
| I/O integration  | `stx.io.save/load` supports `.canvas` directories          |

<!-- EOF -->