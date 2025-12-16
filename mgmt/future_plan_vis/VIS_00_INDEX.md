<!-- ---
!-- Timestamp: 2025-12-13 04:14:26
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/mgmt/vis/VIS_00_INDEX.md
!-- --- -->

# SciTeX Vis Architecture Index

## Document Overview

| Doc                                                           | Purpose                              | Status |
|---------------------------------------------------------------|--------------------------------------|--------|
| [VIS_01_CAPABILITIES](./VIS_01_CAPABILITIES.md)               | Capability-based design (UI/editing) | v0.1   |
| [VIS_02_PLOT_CAPABILITY_MAP](./VIS_02_PLOT_CAPABILITY_MAP.md) | Plot type × capability matrix        | v0.1   |
| [VIS_03_HITMAP_REGISTRY](./VIS_03_HITMAP_REGISTRY.md)         | Hit detection & grouping registry    | v0.1   |
| [VIS_04_MVP_ROADMAP](./VIS_04_MVP_ROADMAP.md)                 | Implementation phases                | v0.1   |

---

## Core Principle

> **Two separate systems with different purposes:**

```
┌─────────────────────────────────────────────────────────┐
│                    SciTeX Vis                           │
├─────────────────────────┬───────────────────────────────┤
│   Capability System     │   PlotTypeRegistry (Hitmap)   │
│   (VIS_01, VIS_02)      │   (VIS_03)                    │
├─────────────────────────┼───────────────────────────────┤
│ "何が編集できるか"         │ "どう当てるか"                  │
│ "What can be edited"    │ "How to hit-test"             │
├─────────────────────────┼───────────────────────────────┤
│ → Inspector UI          │ → Click detection             │
│ → Property editing      │ → Element grouping            │
│ → Group operations      │ → Hitmap generation           │
├─────────────────────────┼───────────────────────────────┤
│ Priority: HIGH (v1)     │ Priority: Medium (v1 simple)  │
└─────────────────────────┴───────────────────────────────┘
```

---

## Key Design Rules

### DO (Capability-based)
```python
# Inspector shows sections based on capabilities
if element.has("HasMarkers"):
    show_marker_section()
if element.has("HasFill"):
    show_fill_section()
```

### DON'T (Type-based)
```python
# This leads to maintenance hell
if element.type == "line":
    ...
elif element.type == "scatter":
    ...
```

---

## Coverage Tiers

| Tier | Coverage | Plot Types                                | Target            |
|------|----------|-------------------------------------------|-------------------|
| v1   | 80%      | line, scatter, bar, text, image           | MVP               |
| v2   | 90%      | + errorbar, fill_between, boxplot, violin | Publication-ready |
| v3   | 95%      | + contour, quiver, streamplot, hexbin     | Power user        |

---

## Quick Reference

### MVP Capabilities (v1)
- Selectable
- Visible
- Positionable
- Stylable (color/alpha/linewidth)
- HasDataMapping (read-only)

### MVP Hit Strategy (v1)
- All types: `bbox_only` (fast, "roughly works")
- Skip pixel-perfect until v2

---

## Related Files

- Implementation: `src/scitex/plt/utils/_hitmap.py`
- Future: `src/scitex/plt/utils/hitmap/` (split modules)
- Future: `src/scitex/vis/capabilities/` (capability system)

---

*Last updated: 2025-12-13*

<!-- EOF -->