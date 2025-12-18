<!-- ---
!-- Timestamp: 2025-12-19
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/README.md
!-- --- -->

# scitex.fig - Unified Figure Bundle Module

**Compose publication-quality figures using `.stx` bundles with the Unified Element API**

Schema Version: 2.0.0

---

## Core Concept: Everything is an Element

No special "panel" terminology. All content types use the **same unified API**:

```python
from scitex.fig import Figz

figz = Figz.create("figure.stx", "My Figure")

# All element types use add_element()
figz.add_element("A", "plot", pltz_bytes, {"x_mm": 10, "y_mm": 10})
figz.add_element("title", "text", "Figure Title", {"x_mm": 85, "y_mm": 5})
figz.add_element("arrow", "shape", {"shape_type": "arrow", ...})
figz.add_element("inset", "figure", child_figz_bytes, {"x_mm": 100, "y_mm": 60})

figz.save()
```

---

## Bundle Formats

Two storage formats are supported:

| Format | Extension | Description | Use Case |
|--------|-----------|-------------|----------|
| ZIP archive | `.stx` | Compressed single file | Storage, transfer, distribution |
| Directory | `.stx.d` | Uncompressed directory | Editing, development, debugging |

```python
# Create ZIP format (default)
figz = Figz.create("figure.stx", "My Figure")

# Create directory format
figz = Figz.create("figure.stx.d", "My Figure")

# Convert between formats
packed = figz.pack()      # .stx.d -> .stx
unpacked = figz.unpack()  # .stx -> .stx.d

# Check format
figz.is_directory  # True for .stx.d
```

---

## Coordinate System

**Origin `(0,0)` at top-left.** All positions in millimeters, relative to parent.

```
(0,0) ──────────────────────────────► x_mm
  │
  │   ┌─────────────────────────────────────┐
  │   │  Figure Canvas (170mm × 120mm)      │
  │   │                                     │
  │   │   Element A at (10, 10)             │
  │   │   ┌────────────────┐                │
  │   │   │ (0,0) local    │                │
  │   │   │   annotation   │                │
  │   │   │   at (5, 3)    │  ← absolute: (15, 13)
  │   │   └────────────────┘                │
  │   │                                     │
  │   └─────────────────────────────────────┘
  ▼
y_mm
```

**Key principle**: Child positions are LOCAL to parent. Moving a parent moves all children.

```python
from scitex.fig import to_absolute

parent_pos = {"x_mm": 10, "y_mm": 10}
child_local = {"x_mm": 5, "y_mm": 3}
child_absolute = to_absolute(child_local, parent_pos)
# Result: {"x_mm": 15.0, "y_mm": 13.0}
```

---

## Element Types

| Type | Content | Description |
|------|---------|-------------|
| `plot` | bytes (.pltz) | Plot bundle |
| `figure` | bytes (.stx) | Nested figure (self-recursive) |
| `text` | str or dict | Text annotation |
| `shape` | dict | Arrow, bracket, line |
| `image` | bytes | Raster image |
| `stats` | bytes (.statsz) | Statistics bundle |

---

## Quick Start

### Create a Figure

```python
from scitex.fig import Figz

# Create bundle
figz = Figz.create(
    "output.stx",
    "Figure 1",
    size_mm={"width": 170, "height": 120}
)

# Add text
figz.add_element("title", "text", "My Results", {"x_mm": 85, "y_mm": 5})

# Add plot (assuming you have pltz_bytes)
with open("plot_a.pltz", "rb") as f:
    pltz_bytes = f.read()
figz.add_element("A", "plot", pltz_bytes, {"x_mm": 10, "y_mm": 20}, {"width_mm": 70, "height_mm": 50})

# Add annotation arrow
figz.add_element("arrow1", "shape", {
    "shape_type": "arrow",
    "start": {"x_mm": 50, "y_mm": 40},
    "end": {"x_mm": 70, "y_mm": 30}
})

figz.save()
```

### Load and Modify

```python
figz = Figz("existing.stx")

# Query elements
print(figz.elements)
print(figz.list_element_ids("plot"))

# Modify
figz.update_element_position("A", x_mm=20, y_mm=30)
figz.remove_element("old_annotation")

figz.save()
```

---

## API Reference

### Creation

```python
# Create new bundle
figz = Figz.create(path, name, size_mm=None, bundle_type="figure")

# Load existing
figz = Figz("path/to/bundle.stx")
```

### Element Operations

```python
# Add element
figz.add_element(
    element_id,      # Unique ID: "A", "title", "arrow_1"
    element_type,    # "plot", "figure", "text", "shape", "image", "stats"
    content,         # bytes, str, or dict (type-dependent)
    position,        # {"x_mm": float, "y_mm": float}
    size,            # {"width_mm": float, "height_mm": float}
    **kwargs         # Additional properties
)

# Query
elem = figz.get_element(id)
content = figz.get_element_content(id)  # For embedded bundles
ids = figz.list_element_ids()           # All IDs
ids = figz.list_element_ids("plot")     # Filter by type

# Modify
figz.update_element_position(id, x_mm, y_mm)
figz.update_element_size(id, width_mm, height_mm)
figz.remove_element(id)

# Save
figz.save()
```

### Layout Utilities

```python
from scitex.fig import (
    to_absolute,
    to_relative,
    normalize_position,
    normalize_size,
    element_bounds,
    auto_layout_grid,
)

# Coordinate transforms
abs_pos = to_absolute(local_pos, parent_pos)
local_pos = to_relative(abs_pos, parent_pos)

# Normalize formats
pos = normalize_position({"x": 10, "y": 20})  # → {"x_mm": 10.0, "y_mm": 20.0}

# Auto-layout
layouts = auto_layout_grid(4, {"width_mm": 170, "height_mm": 120})
for pos, size in layouts:
    print(pos, size)
```

### Properties

```python
figz.bundle_id     # UUID
figz.bundle_type   # "figure", "plot", etc.
figz.elements      # List of element specs
figz.size_mm       # {"width": 170, "height": 120}
figz.constraints   # {"allow_children": True, "max_depth": 3}
figz.spec          # Full spec dict
figz.style         # Style dict
```

---

## Self-Recursive Structure

Figures can contain figures. Depth controlled by `TYPE_DEFAULTS`:

```python
from scitex.io.bundle import TYPE_DEFAULTS

# TYPE_DEFAULTS = {
#     "figure": {"allow_children": True, "max_depth": 3},
#     "plot": {"allow_children": True, "max_depth": 2},
#     "text": {"allow_children": False, "max_depth": 1},
#     ...
# }
```

### Nested Figure Example

```python
# Create child figure
child = Figz.create("/tmp/inset.stx", "Inset")
child.add_element("detail", "text", "Zoomed view", {"x_mm": 5, "y_mm": 5})
child.save()

# Add to parent
with open("/tmp/inset.stx", "rb") as f:
    child_bytes = f.read()

parent = Figz.create("main.stx", "Main Figure")
parent.add_element("inset", "figure", child_bytes, {"x_mm": 100, "y_mm": 60})
parent.save()
```

---

## Module Structure

```
scitex/fig/
├── README.md        # This file
├── __init__.py      # Main exports
├── _bundle.py       # Figz class (Unified Element API)
├── layout.py        # Coordinate system utilities
├── io/              # Load/save operations
├── model/           # Data models
├── backend/         # Rendering backends
├── utils/           # Validation, defaults
└── editor/          # Interactive editing
```

---

## Migration from Legacy API

The old "panel" API still works but emits deprecation warnings:

```python
# Old (deprecated) → New (recommended)
figz.add_panel("A", bytes)        → figz.add_element("A", "plot", bytes)
figz.panels                       → figz.elements
figz.get_panel("A")               → figz.get_element("A")
figz.get_panel_pltz("A")          → figz.get_element_content("A")
figz.list_panel_ids()             → figz.list_element_ids("plot")
```

---

## Examples

See `examples/fig/` for complete examples:

- `unified_element_api.py` - Demonstrates the unified element API

---

## Related Documentation

- [layout.py](./layout.py) - Coordinate system utilities (inline docs)
- [scitex.io.bundle](../io/bundle/README.md) - Bundle I/O operations
- [TYPE_DEFAULTS](../io/bundle/_types.py) - Element type constraints

<!-- EOF -->
