# SciTeX Vis - Plot Types × Capabilities Map v0.1

This document maps **plot types → required capabilities** and defines **coverage tiers (80/90/95%)**.

---

## Capability Legend

### Core (MVP)
- **Selectable** - Click/hover detection
- **Visible** - Show/hide, lock
- **Positionable** - x_mm, y_mm placement
- **Stylable** - color, alpha, linewidth

### Layout/Geometry
- **Sizable** - width, height
- **HasPathGeometry** - Path-based hit testing

### Data/Semantics
- **HasDataMapping** - Data provenance tracking
- **HasErrorBars** - xerr, yerr
- **HasColormap** - cmap, vmin, vmax

### Appearance Extensions
- **HasMarkers** - marker shape, size
- **HasFill** - facecolor, edgecolor, hatch
- **HasTextContent** - text value, font

---

## Plot Type × Capability Table

| Plot Type | Required Capabilities |
|-----------|----------------------|
| **Line** | Selectable, Visible, Stylable, HasDataMapping, HasPathGeometry |
| **Scatter** | Selectable, Visible, Stylable, HasDataMapping, HasMarkers |
| **Bar** | Selectable, Visible, Positionable, Sizable, Stylable, HasFill, HasDataMapping |
| **Histogram** | Selectable, Visible, HasDataMapping, HasFill |
| **Errorbar** | Line + HasErrorBars |
| **Area/Fill_between** | Selectable, Visible, HasDataMapping, HasFill |
| **Step** | Line |
| **Stem** | Line + HasMarkers |
| **Boxplot** | Selectable, Visible, HasDataMapping, HasFill |
| **Violin** | Selectable, Visible, HasDataMapping, HasFill |
| **Image (imshow)** | Selectable, Visible, Sizable, HasColormap |
| **Heatmap** | Image + HasDataMapping |
| **Contour** | Selectable, Visible, HasColormap |
| **Contourf** | Contour + HasFill |
| **Text** | Selectable, Visible, Positionable, HasTextContent |
| **Annotation** | Text |
| **Arrow** | Selectable, Visible, Positionable |
| **Patch (rect/circle)** | Selectable, Visible, Positionable, Sizable, HasFill |
| **Legend** | Selectable, Visible, Positionable |
| **Colorbar** | Selectable, Visible, Positionable, HasColormap |

---

## Coverage Definition

> **Coverage** = Percentage of figures in real scientific papers that can be faithfully recreated and edited.

---

## 80% Coverage (MVP / v1 Target)

### Supported Plot Types
- Line
- Scatter
- Bar
- Histogram
- Text / Annotation
- Image (imshow)
- Heatmap (basic)

### Required Capabilities
- Selectable
- Visible
- Positionable
- Stylable
- HasDataMapping
- HasMarkers
- HasFill
- Sizable
- HasColormap (basic)

### NOT in v1
- Error bars
- Contour / Violin
- Advanced grouping semantics

> Covers **most neuroscience, ML, bio, physics figures**

---

## 90% Coverage (Publication-Ready / v2)

### Additional Plot Types
- Errorbar
- Area / Fill_between
- Boxplot
- Violin
- Step / Stem
- Patch shapes

### Additional Capabilities
- HasErrorBars
- HasPathGeometry (robust hit-test)
- Extended HasFill (hatch, edge control)

> Suitable for *Nature / Neuron / PNAS* level figures

---

## 95% Coverage (Power User / v3)

### Additional Plot Types
- Contour / Contourf
- Quiver / Vector field
- Streamplot
- Hexbin
- Pie (optional, low scientific value)
- Polar plots

### Additional Capabilities
- Advanced HasColormap (norm, ticks)
- Multi-space Positionable (axes vs figure)
- Complex grouping semantics
- Axis-linked capabilities (shared transforms)

> Mostly needed by niche subfields. High cost, diminishing returns.

---

## Strategic Recommendation

| Phase | Coverage | Strategy |
|-------|----------|----------|
| v1 | 80% | **Ship early** - minimal viable |
| v2 | 90% | **Publication tool** - serious usage |
| v3 | 95% | **Optional** - implement on demand |

> **Capability-first design ensures each added capability unlocks multiple plot types.**

---

## Capability → Plot Type Unlock Matrix

Adding a capability unlocks these plot types:

| Capability Added | Unlocks |
|-----------------|---------|
| HasMarkers | scatter, stem |
| HasFill | bar, histogram, boxplot, violin, fill_between, patch |
| HasErrorBars | errorbar |
| HasColormap | image, heatmap, contour |
| HasTextContent | text, annotation |
| HasPathGeometry | precise line selection |

---

## Version

- scitex.vis.plot-capability-map: v0.1
- Last updated: 2025-12-13
