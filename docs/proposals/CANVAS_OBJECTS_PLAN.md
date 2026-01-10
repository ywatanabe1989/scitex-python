<!-- ---
!-- Timestamp: 2025-12-17 10:52:05
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/docs/CANVAS_OBJECTS_PLAN.md
!-- --- -->

# Plan: Custom Objects Handling in scitex.fig

## Goal

Enable users to add custom objects (images, figures, shapes, text, etc.) to figure compositions in a unified way that works seamlessly across:
- Local Python API (`stx.fig.Figure`)
- Local GUI editors (DearPyGui, Qt, etc.)
- Django web app (`/vis/` module)

---

## Current Structure (As Implemented)

### Project Layout

```
proj-root/scitex/vis/
├── ai/prompts/           # AI prompts for vis module
├── figures/              # User-created figures (.figz.d bundles)
└── gallery/              # Pre-built plot templates (.pltz.d bundles)
    ├── line/
    ├── scatter/
    ├── categorical/
    ├── distribution/
    ├── statistical/
    ├── grid/
    ├── area/
    ├── contour/
    ├── vector/
    └── special/
```

### Bundle Types (Current)

| Type   | Extension | Schema                     | Purpose                            |
|--------|-----------|----------------------------|------------------------------------|
| Plot   | `.pltz.d` | `scitex.plt.spec v1.0.0`   | Single plot (data + axes + traces) |
| Figure | `.figz.d` | `scitex.fig.figure v1.0.0` | Multi-panel composition            |

### .figz.d Structure (Current)

```
figure.figz.d/
├── spec.json              # Figure specification (semantic)
├── style.json             # Figure style (appearance)
├── {basename}.json        # Combined spec+style (legacy compat)
├── exports/
│   ├── figure.png
│   ├── figure.svg
│   └── figure_overview.png
├── cache/
│   ├── geometry_px.json
│   └── render_manifest.json
├── A.pltz.d/              # Panel bundles (at root level)
├── B.pltz.d/
└── README.md
```

### .pltz.d Structure (Current)

```
plot.pltz.d/
├── spec.json              # Plot specification (data, axes, traces)
├── style.json             # Plot style (colors, fonts, theme)
├── plot.csv               # Data file
├── exports/
│   ├── plot.png
│   ├── plot.svg
│   └── plot_hitmap.png
├── cache/
│   └── geometry_px.json
└── README.md
```

---

## Proposed Enhancement: Custom Objects

### Problem

Current `.figz.d` only supports `.pltz.d` panels. Users need to add:
- External images (photos, diagrams, logos)
- Shapes (rectangles, lines, arrows, brackets)
- Text annotations (labels, captions, equations)
- Other vector graphics (SVG)

### Solution: Unified Objects Model

Extend `.figz.d` to handle any object type uniformly.

---

## Workflow Overview

```
User Input                    Processing                     Storage
─────────────────────────────────────────────────────────────────────────

  Drag & Drop    ──┐                                         .figz.d/
  File Picker    ──┼──▶  Unified Import   ──▶  Normalize  ──▶  assets/
  API Call       ──┘     Handler               to PNG/SVG       ├── img_001.png
                                                                ├── logo_002.svg
                              │                                 └── photo_003.png
                              ▼
                         Register in                        *.pltz.d/
                         spec.json                            ├── A.pltz.d/
                              │                               └── B.pltz.d/
                              ▼
                         objects: [
                           {id, type, asset_ref, position, size, style}
                         ]
```

---

## 1. Asset Import & Normalization

### Supported Input Types → Normalized Storage

| Input Type       | Extension                            | Normalized To   | Editable |
|------------------|--------------------------------------|-----------------|----------|
| Raster Image     | .png, .jpg, .jpeg, .gif, .bmp, .tiff | PNG             | No       |
| Vector Image     | .svg                                 | SVG (keep)      | No       |
| PDF Figure       | .pdf                                 | SVG or PNG      | No       |
| SciTeX Plot      | .pltz.d                              | Keep as .pltz.d | Yes      |
| Custom Plot JSON | .json                                | .pltz.d bundle  | Yes      |

### Why Normalize?

1. **Consistency**: Renderer only handles PNG/SVG/PLTZ
2. **Web compatibility**: Django serves PNG/SVG directly
3. **Quality preservation**: SVG for vector, PNG for raster
4. **Editability**: .pltz.d preserved for re-editing

---

## 2. Proposed Storage Structure

### Enhanced .figz.d Structure

```
figure.figz.d/
├── spec.json              # Figure specification (semantic)
├── style.json             # Figure style (appearance)
├── {basename}.json        # Combined (legacy compat)
├── exports/
│   ├── figure.png
│   ├── figure.svg
│   └── figure_overview.png
├── cache/
│   ├── geometry_px.json
│   └── render_manifest.json
├── assets/                # NEW: Imported custom objects
│   ├── img_001.png        # Normalized raster
│   ├── logo_002.svg       # Vector graphic
│   └── photo_003.png      # User photo
├── A.pltz.d/              # Panel bundles (existing)
├── B.pltz.d/
└── README.md
```

### Asset Naming Convention

```
{type}_{sequence:03d}.{ext}

Examples:
- img_001.png      (imported image)
- logo_002.svg     (vector graphic)
- shape_003.svg    (vector shape)
```

---

## 3. Schema Update: spec.json v1.1.0

### Current Schema (v1.0.0)

```json
{
  "schema": {"name": "scitex.fig.figure", "version": "1.0.0"},
  "figure": {"id": "figure", "title": "", "caption": ""},
  "panels": [
    {"id": "A", "plot": "A.pltz.d", "position": {...}, "size": {...}}
  ],
  "notations": [...]
}
```

### Proposed Schema (v1.1.0)

```json
{
  "schema": {"name": "scitex.fig.figure", "version": "1.1.0"},
  "figure": {"id": "figure", "title": "", "caption": ""},

  "objects": [
    {
      "id": "panel_A",
      "type": "figure",
      "asset": "A.pltz.d",
      "position": {"x_mm": 10, "y_mm": 10},
      "size": {"width_mm": 80, "height_mm": 60},
      "z_index": 0,
      "visible": true,
      "locked": false,
      "label": {"text": "A", "position": "top-left"}
    },
    {
      "id": "img_001",
      "type": "image",
      "asset": "assets/img_001.png",
      "position": {"x_mm": 100, "y_mm": 10},
      "size": {"width_mm": 70, "height_mm": 50},
      "z_index": 1,
      "style": {"opacity": 1.0, "border": null}
    },
    {
      "id": "text_001",
      "type": "text",
      "content": "Scale bar: 100 um",
      "position": {"x_mm": 50, "y_mm": 100},
      "style": {"fontsize": 10, "color": "#000000"}
    }
  ],

  "panels": [],
  "notations": []
}
```

### Object Types

| Type       | Description                              | Has Asset |
|------------|------------------------------------------|-----------|
| `figure`   | .pltz.d plot bundle (editable)           | Yes       |
| `image`    | PNG/SVG image (non-editable)             | Yes       |
| `shape`    | Rectangle, ellipse, line, arrow, bracket | No        |
| `text`     | Text label/annotation                    | No        |
| `symbol`   | Scientific marker (*, **, dagger)        | No        |
| `equation` | LaTeX equation                           | No        |

### Migration Strategy

```python
def migrate_v1_0_to_v1_1(spec: dict) -> dict:
    """Migrate panels/notations to unified objects list."""
    objects = []

    # Convert panels → figure objects
    for panel in spec.get("panels", []):
        objects.append({
            "id": f"panel_{panel['id']}",
            "type": "figure",
            "asset": panel["plot"],
            "position": panel.get("position", {}),
            "size": panel.get("size", {}),
            "z_index": len(objects),
            "label": {"text": panel.get("label", panel["id"])}
        })

    # Convert notations → text/shape objects
    for notation in spec.get("notations", []):
        objects.append(convert_notation(notation))

    spec["schema"]["version"] = "1.1.0"
    spec["objects"] = objects
    return spec
```

---

## 4. Import Workflow

### Python API

```python
import scitex as stx

figure = stx.fig.Figure("fig1", width_mm=180, height_mm=120)

# Add any image - automatically normalized
figure.add_object("photo.jpg", position=(10, 10), size=(50, 40))
figure.add_object("diagram.pdf", position=(70, 10), size=(50, 40))
figure.add_object("plot.pltz.d", position=(10, 60), size=(80, 50), label="A")

# Add shapes/text directly
figure.add_text("Figure 1", position=(90, 5), fontsize=14)
figure.add_shape("rectangle", position=(5, 5), size=(170, 110), stroke="#000")

stx.save(figure, "output/fig1.figz.d")
```

### Django Web App

```
┌─────────────────────────────────────────────────────────┐
│  /vis/                                                  │
├────────────┬────────────────────────────┬───────────────┤
│ File Tree  │      Figure Editor         │   Properties  │
│            │  ┌─────────────────────┐   │               │
│ figures/   │  │ Drop files here     │   │   Position    │
│  └ fig1    │  │ or click to browse  │   │   x: [___] mm │
│ gallery/   │  └─────────────────────┘   │   y: [___] mm │
│  └ line/   │                            │               │
│            │  ← Figure renders here     │   Size        │
│            │                            │   w: [___] mm │
│            │                            │   h: [___] mm │
└────────────┴────────────────────────────┴───────────────┘

Drag & Drop → AJAX upload → normalize → store in assets/ → update spec.json
```

---

## 5. Implementation Modules

### Core (scitex-code)

| File                                | Purpose                      |
|-------------------------------------|------------------------------|
| `src/scitex/fig/model/_objects.py`  | Object dataclasses           |
| `src/scitex/fig/io/_assets.py`      | Asset import/normalize/store |
| `src/scitex/fig/io/_migrate.py`     | Schema migration v1.0→v1.1   |
| `src/scitex/fig/io/_bundle.py`      | Extended bundle I/O          |
| `src/scitex/fig/backend/_render.py` | Object rendering             |

### Django (scitex-cloud)

| File                                           | Purpose                  |
|------------------------------------------------|--------------------------|
| `apps/vis_app/views/api/assets.py`             | Asset upload endpoint    |
| `apps/vis_app/services/figz_service.py`        | Extended figz service    |
| `static/vis_app/ts/vis-editor/AssetManager.ts` | Frontend upload handling |

---

## 6. Key Design Decisions

### Q1: Where are assets stored?

**Answer**: Inside `.figz.d/assets/` directory
- Self-contained bundle (portable)
- Easy backup/version control
- Works offline
- Django serves from project storage

### Q2: How to handle large files?

**Answer**:
- Resize images > 4096px to max 4096px
- Compress PNG with optimization
- Warn user if file > 10MB
- Option to link external (advanced users)

### Q3: How to sync between local and Django?

**Answer**:
- `.figz.d` bundle is the source of truth
- Django reads/writes same structure
- `stx.save()` and Django both use same `scitex.fig.io` module
- No separate Django storage format

### Q4: Backward compatibility?

**Answer**:
- v1.0.0 figz bundles load normally
- On first save, migrate to v1.1.0
- Keep `panels`/`notations` in JSON for old readers
- Warning log for deprecated access patterns

---

## 7. Phase Implementation Order

| Phase | Scope                       | Files                       |
|-------|-----------------------------|-----------------------------|
| **1** | Object model + Asset import | `_objects.py`, `_assets.py` |
| **2** | Bundle I/O extension        | `_bundle.py`                |
| **3** | Rendering support           | `_render.py`                |
| **4** | Schema migration            | `_migrate.py`               |
| **5** | Django integration          | `vis_app/`                  |

---

## 8. Example: Full Workflow

```python
# User has: photo.jpg, data_plot.pltz.d, logo.svg

import scitex as stx

# Create figure
figure = stx.fig.Figure("publication_fig", width_mm=180, height_mm=120)

# Add objects - all handled uniformly
figure.add_object("./photo.jpg", position=(10, 10), size=(50, 50), label="A")
figure.add_object("./data_plot.pltz.d", position=(70, 10), size=(100, 50), label="B")
figure.add_object("./logo.svg", position=(150, 100), size=(20, 10))

# Add annotation
figure.add_text("*p < 0.05", position=(120, 65), fontsize=8)

# Save - creates self-contained bundle
stx.save(figure, "output/publication_fig.figz.d")

# Result:
# output/publication_fig.figz.d/
#   ├── spec.json
#   ├── style.json
#   ├── assets/
#   │   ├── img_001.png      (normalized from photo.jpg)
#   │   └── img_002.svg      (copied logo)
#   ├── data_plot.pltz.d/    (copied plot bundle)
#   ├── exports/
#   │   ├── publication_fig.png
#   │   └── publication_fig.svg
#   └── README.md
```

---

## Summary

1. **Unified object model** - All items (plots, images, shapes, text) are "objects"
2. **Asset normalization** - Import anything, store as PNG/SVG/PLTZ
3. **Self-contained bundles** - `.figz.d/assets/` holds custom objects
4. **Same code path** - Python API, GUI, Django all use `scitex.fig`
5. **Schema evolution** - v1.0 → v1.1 with migration support

<!-- EOF -->