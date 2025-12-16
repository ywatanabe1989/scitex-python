# SciTeX Vis MVP Roadmap v0.1

## Philosophy

> **"ショボくても一通り動く"** (Even if basic, make it work end-to-end)
>
> **"やらないことを決めたプロダクトは、必ず前に進む"**
> (A product that decides what NOT to do always moves forward)

---

## Success Criteria

> **"matplotlib で作った図を、壊さずに・微調整できる"**
>
> When this works, SciTeX Vis becomes "usable".

---

## v1 MVP: Must Have (これがないと意味がない)

### 1. Select Elements
- [x] Click to select 1 element
- [ ] Highlight on selection (border or glow)
- [ ] Inspector shows "what is this"

> 選べない Vis は存在価値がない

### 2. Move Elements (mm-based)
- [ ] Drag to move
- [ ] Shift key for axis-lock (x/y)
- [ ] Numeric input (x_mm, y_mm)

> Snap/align is v2

### 3. Edit Axes
- [ ] xlim / ylim
- [ ] linear / log scale
- [ ] tick count (auto or fixed)

> Without axis editing, doesn't feel like "beyond matplotlib"

### 4. Minimal Styling
- [ ] color
- [ ] linewidth / size
- [ ] alpha

> font / colormap / theme is v2

### 5. Save/Load (SciTeX-style)
- [ ] .pltz / .figz save
- [ ] JSON generated
- [ ] Reload works

> Export is PNG, but **structure save is the real feature**

---

## v1 MVP: Plot Types (80% coverage)

| Type | Status | Notes |
|------|--------|-------|
| line | [ ] | |
| scatter | [ ] | |
| bar | [ ] | |
| text | [ ] | |
| image (imshow) | [ ] | |

> violin / contour / fill_between = v2

---

## v1 MVP: Capabilities

| Capability | Status | Notes |
|------------|--------|-------|
| Selectable | [ ] | |
| Visible | [ ] | |
| Positionable | [ ] | |
| Stylable | [ ] | color/alpha/linewidth only |
| HasDataMapping | [ ] | read-only |

---

## v1: Explicitly NOT Doing (捨てる)

### Multi-select
- [ ] ~~Box select~~
- [ ] ~~Group move~~

→ v2

### Perfect Hit-test
- [ ] ~~All artist types~~
- [ ] ~~Overlap cycling~~

→ v1 is "だいたい当たる" (roughly works)

### Auto Layout
- [ ] ~~Align~~
- [ ] ~~Distribute~~
- [ ] ~~Snap~~

→ Manual move is fine

### Advanced Annotation
- [ ] ~~Bracket~~
- [ ] ~~Stats auto-annotation~~

→ Text only is enough

---

## Implementation Order (超現実的)

```
1. JSON → Canvas render        (already exists)
2. Hitmap for element_id       (bbox_only, fast)
3. Selection state             (single element)
4. Drag → mm update → re-render
5. Inspector → JSON edit
6. Save → Reload
```

---

## v2: Publication-Ready (90% coverage)

### Additional Plot Types
- [ ] errorbar
- [ ] fill_between
- [ ] boxplot
- [ ] violin
- [ ] step / stem

### Additional Capabilities
- [ ] HasErrorBars
- [ ] HasPathGeometry (path-based hit-test)
- [ ] Extended HasFill (hatch, edge)

### Additional Features
- [ ] Multi-select (box select)
- [ ] Group operations
- [ ] Snap to grid/axis
- [ ] Undo/Redo

---

## v3: Power User (95% coverage)

### Additional Plot Types
- [ ] contour / contourf
- [ ] quiver / streamplot
- [ ] hexbin
- [ ] pie (optional)
- [ ] polar

### Additional Capabilities
- [ ] Advanced HasColormap
- [ ] Multi-space Positionable
- [ ] Complex grouping

### Additional Features
- [ ] Overlap cycling
- [ ] Pixel-perfect hit-test
- [ ] Advanced annotation (bracket, stats)

---

## Immediate Actions

### Quick Fix: Slow Heatmaps (30 min)
```python
# Skip pixel-perfect for complex types
SKIP_TYPES = {"mesh", "image", "contour", "fill"}
```

### Next: Capability System Skeleton (2-4 hours)
```python
class Element:
    capabilities: Set[str]

    def has(self, cap: str) -> bool:
        return cap in self.capabilities
```

### Then: Inspector Based on Capabilities
```python
if element.has("HasMarkers"):
    show_marker_section()
```

---

## File Checklist

| File | Purpose | Priority |
|------|---------|----------|
| `VIS_01_CAPABILITIES.md` | Capability definitions | Done |
| `VIS_02_PLOT_CAPABILITY_MAP.md` | Coverage tiers | Done |
| `VIS_03_HITMAP_REGISTRY.md` | Hit detection specs | Done |
| `VIS_04_MVP_ROADMAP.md` | This file | Done |

---

## Metrics

### v1 Success
- [ ] Can select line/scatter/bar/text/image
- [ ] Can move elements by drag
- [ ] Can edit xlim/ylim
- [ ] Can change color/linewidth
- [ ] Can save and reload

### v2 Success
- [ ] 90% of scientific figures can be edited
- [ ] Multi-select works
- [ ] Undo/Redo works

---

## Version

- scitex.vis.roadmap: v0.1
- Last updated: 2025-12-13
