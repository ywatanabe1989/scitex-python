# SciTeX Vis Capabilities Specification v0.1

## Purpose

This document defines **capability-based design** for SciTeX Vis.

Element behavior, Inspector UI, and grouping rules are determined by **capabilities**, not by plot types.

---

## Core Principles

1. Element *types* (line, scatter, bar...) are **NOT** first-class UI logic
2. All behavior is expressed via **capabilities**
3. Inspector and grouping logic depend only on capability intersections
4. New plot types are introduced by **combining existing capabilities**

---

## Core Capabilities (MVP / v1)

### Selectable

**Purpose:** Identify elements via mouse interaction

**Properties:**
- `element_id: str`
- `zorder: int` (tie-break when equal)
- `pick_mask: hitmap | geometry`

**Operations:**
- `hit_test(x_px, y_px) -> element_id | None`
- `hover_enter / hover_leave`
- `select / deselect`

**Constraints:**
- Overlap resolved by zorder
- v1: Return top-most element only (no cycling)

---

### Visible

**Purpose:** Control visibility and editability

**Properties:**
- `visible: bool`
- `locked: bool`

**Operations:**
- `toggle_visible()`
- `toggle_lock()`

**Constraints:**
- Locked elements cannot be transformed/styled

---

### Positionable

**Purpose:** Control element placement (mm-based)

**Properties:**
- `position: { x_mm: float, y_mm: float }`
- `anchor: str` ("center" | "top-left" | "baseline-left")
- `space: str` ("figure" | "axes")

**Operations:**
- Drag move
- Numeric input
- Axis-lock (Shift key)

**Constraints:**
- v1: "figure" space only
- v2: "axes" space with transform

---

### Sizable

**Purpose:** Control element dimensions

**Properties:**
- `size: { w_mm: float, h_mm: float }`
- `aspect_lock: bool`

**Operations:**
- Resize handles
- Numeric input

**Constraints:**
- Text elements don't have Sizable (no natural size)

---

### Stylable

**Purpose:** Shared visual styling

**Properties (MVP):**
- `style: { color, alpha, linewidth }`

**Operations:**
- `set_color()`, `set_alpha()`, `set_linewidth()`

**Constraints:**
- Application target varies (line=stroke, bar=edge/face)
- Sub-capabilities handle specifics (HasFill, HasMarkers)

---

## Data Capability (v1 read-only)

### HasDataMapping

**Purpose:** Track data provenance and binding

**Properties:**
- `data_ref: { source_id, table_id?, x_key?, y_key?, row_ids? }`
- `mapping: { x, y, ... }`

**Operations:**
- `inspect()` (read-only in v1)
- Later: relink / filter / rebind

**Constraints:**
- Vis is about reference and binding, not data editing

---

## Extended Capabilities (v2+)

### HasTextContent

**Properties:**
- `text: { value, font_size_pt, family?, weight?, ha?, va?, rotation_deg? }`

**Operations:**
- Edit text, set font size, align

---

### HasMarkers

**Properties:**
- `marker: { shape, size_pt, edgecolor?, facecolor? }`

**Operations:**
- Set marker shape/size/colors

---

### HasFill

**Properties:**
- `fill: { facecolor, edgecolor, hatch?, alpha? }`

**Operations:**
- Set face/edge, hatch

---

### HasColormap

**Properties:**
- `cmap: { name, vmin?, vmax?, norm? }`
- `colorbar: { show, location?, ticks? }`

**Operations:**
- Set vmin/vmax, change cmap

---

### HasErrorBars

**Properties:**
- `err: { xerr?, yerr?, capsize?, errcolor? }`

---

### HasPathGeometry

**Properties:**
- `geometry_ref` (path/segments reference)

**Operations:**
- Hit-test by distance-to-segment
- Node edit (later)

---

## Grouping Rules (Capability-based)

### Group Creation Conditions

```python
# All must have Selectable and Visible
GroupMove_allowed = all(e.has("Positionable") for e in elements)
GroupStyle_allowed = all(e.has("Stylable") for e in elements) and style_keys_overlap
GroupResize_allowed = all(e.has("Sizable") for e in elements)
```

### Mixed Selection Rule

When selected elements have mixed capabilities:
- Inspector shows **only shared capabilities**
- Properties show **only common keys**
- Mixed values display as "mixed"

---

## Inspector Rendering Rules

Inspector sections appear **only if all selected elements share the capability**.

**Display Order:**
1. Identity (id/type display is OK)
2. Visibility (Visible)
3. Position (Positionable)
4. Size (Sizable)
5. Style (Stylable + HasFill/HasMarkers/HasText...)
6. Data (HasDataMapping)
7. Advanced (HasColormap/HasErrorBars/...)

---

## MVP Capability Set (v1)

**Required for v1:**
- Selectable
- Visible
- Positionable
- Stylable (color/alpha/linewidth only)
- HasDataMapping (read-only)

> This enables: 選べる → 動かせる → 保存できる → 再現できる

---

## Implementation Notes

```python
# Element exposes capabilities
class Element:
    capabilities: Set[str]

    def has(self, cap: str) -> bool:
        return cap in self.capabilities

# Each capability defines:
class Capability:
    def get_props(element) -> dict
    def apply_patch(element, patch: dict) -> None

# UI checks capability, not type
if element.has("HasTextContent"):
    show_text_inspector()
```

---

## JSON Schema

```json
{
  "element_id": "e12",
  "type": "scatter",
  "capabilities": ["Selectable", "Visible", "Positionable", "Stylable", "HasDataMapping", "HasMarkers"],
  "position": {"x_mm": 12.3, "y_mm": 45.6},
  "style": {"color": "#333", "alpha": 0.8},
  "marker": {"shape": "o", "size_pt": 6}
}
```

> New types can be added without breaking schema

---

## Version

- scitex.vis.capabilities: v0.1
- Last updated: 2025-12-13
