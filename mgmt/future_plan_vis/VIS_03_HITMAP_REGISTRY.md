# SciTeX Hitmap Registry Specification v0.1

## Purpose

This document defines the **PlotTypeRegistry** for hitmap generation.

> **Separate from Capability System:**
> - Hitmap Registry: "どう当てるか" (how to hit-test)
> - Capability System: "何が編集できるか" (what can be edited)

---

## Core Design

### PlotTypeConfig

```python
@dataclass
class PlotTypeConfig:
    # Identification
    name: str                           # "line", "scatter", "histogram"
    artist_classes: Tuple[Type, ...]    # Matplotlib classes to match

    # Hit detection strategy
    strategy: Literal["pixel_perfect", "path_based", "bbox_only", "group_as_one"]
    selection_scope: Literal["artist", "group", "none"]

    # Disambiguation (higher wins)
    priority: int = 50
    match_predicate: Optional[Callable[[Artist], bool]] = None

    # Grouping behavior
    grouping: Optional[str] = None      # "histogram", "bar_series", "contour", etc.

    # Performance hint
    complexity: Literal["simple", "medium", "complex"] = "simple"

    # Path sampling (for path_based strategy)
    path_sampling: Optional[int] = None
```

---

## Hit Detection Strategies

### 1. bbox_only (Fastest)

- Simple rectangular bounding box
- O(n) lookup
- Good for: bars, histograms, heatmaps, simple shapes
- Less precise for diagonal/curved elements

```python
def hit_test_bbox(x, y, element):
    bbox = element.bbox_px
    return bbox.x0 <= x <= bbox.x1 and bbox.y0 <= y <= bbox.y1
```

### 2. path_based (Balanced)

- Sampled path points with distance calculation
- O(n × m) where m = points per element
- Good for: lines, scatter plots
- Works at any zoom level

```python
def hit_test_path(x, y, element, threshold=5):
    for px, py in element.path_px:
        if distance(x, y, px, py) < threshold:
            return True
    return False
```

### 3. pixel_perfect (Most Accurate)

- Hitmap PNG with RGB-encoded element IDs
- O(1) lookup
- Good for: complex shapes, overlapping elements
- Trade-off: Extra file, regenerate on resize

```python
def hit_test_pixel(x, y, hitmap):
    rgb = hitmap[y, x]
    element_id = rgb_to_id(rgb)
    return element_id
```

### 4. group_as_one (Composite)

- Multiple artists treated as single element
- Combined bounding box
- Good for: errorbar, boxplot containers

---

## Plot Type Registry (v1)

### MVP Types (80% coverage)

| Type | Strategy | Scope | Priority | Grouping | Complexity |
|------|----------|-------|----------|----------|------------|
| **line** | bbox_only | artist | 50 | - | simple |
| **scatter** | bbox_only | artist | 50 | - | simple |
| **bar** | bbox_only | artist | 50 | - | simple |
| **histogram** | bbox_only | group | 60 | histogram | simple |
| **text** | bbox_only | artist | 50 | - | simple |
| **image** | bbox_only | artist | 50 | - | simple |
| **heatmap** | bbox_only | artist | 50 | - | simple |

### v2 Types (90% coverage)

| Type | Strategy | Scope | Priority | Grouping | Complexity |
|------|----------|-------|----------|----------|------------|
| **errorbar** | group_as_one | group | 55 | errorbar | medium |
| **fill_between** | bbox_only | artist | 50 | - | medium |
| **boxplot** | group_as_one | group | 55 | boxplot | medium |
| **violin** | bbox_only | group | 55 | violin | medium |
| **bar_series** | bbox_only | group | 55 | bar_series | simple |

### v3 Types (95% coverage)

| Type | Strategy | Scope | Priority | Grouping | Complexity |
|------|----------|-------|----------|----------|------------|
| **contour** | bbox_only | group | 55 | contour | complex |
| **contourf** | bbox_only | group | 55 | contour | complex |
| **quiver** | pixel_perfect | artist | 50 | - | complex |
| **pie** | pixel_perfect | artist | 50 | pie | simple |

---

## MVP Strategy: "だいたい当たる" (Roughly Works)

For v1, **all types use `bbox_only`**:

```python
# v1 implementation - simple and fast
MVP_STRATEGY = "bbox_only"

def get_hit_strategy(artist_type: str) -> str:
    # v1: Everything uses bbox
    return "bbox_only"
```

Benefits:
- Fast hitmap generation (no pixel-perfect rendering)
- No slow heatmap/contour rendering
- Simple implementation
- "Good enough" for 80% use cases

---

## Disambiguation with Priority

When multiple types match the same artist:

```python
# Example: BarContainer matches both histogram and bar_series
histogram_config = PlotTypeConfig(
    name="histogram",
    priority=60,  # Higher priority
    match_predicate=detect_histogram_pattern,
)

bar_series_config = PlotTypeConfig(
    name="bar_series",
    priority=55,  # Lower priority
    match_predicate=detect_bar_series_pattern,
)
```

### Match Predicates

```python
def detect_histogram_pattern(container) -> bool:
    """Bars are adjacent (no gaps) = histogram."""
    patches = container.patches
    if len(patches) < 2:
        return False

    gaps = []
    for i in range(len(patches) - 1):
        x1 = patches[i].get_x() + patches[i].get_width()
        x2 = patches[i + 1].get_x()
        gaps.append(x2 - x1)

    avg_width = sum(p.get_width() for p in patches) / len(patches)
    return all(abs(g) < avg_width * 0.1 for g in gaps)


def detect_bar_series_pattern(axis) -> bool:
    """Multiple BarContainers = grouped bar chart."""
    containers = [c for c in axis.containers if 'BarContainer' in type(c).__name__]
    return len(containers) > 1
```

---

## Selection Scope

### artist
- Individual element selected
- Each Line2D, Rectangle, etc. is independent

### group
- Logical group selected as one unit
- All histogram bars, all errorbar components

### none
- Element is not selectable (axes elements, background)

---

## Implementation Phases

### Phase 1: MVP (v1)
```python
# Simple bbox-only for all types
PLOT_TYPE_REGISTRY = {
    "line": PlotTypeConfig(name="line", strategy="bbox_only", ...),
    "scatter": PlotTypeConfig(name="scatter", strategy="bbox_only", ...),
    "bar": PlotTypeConfig(name="bar", strategy="bbox_only", ...),
    # ... all bbox_only
}
```

### Phase 2: Add Grouping (v1.5)
```python
# Add logical grouping detection
PLOT_TYPE_REGISTRY["histogram"] = PlotTypeConfig(
    name="histogram",
    strategy="bbox_only",
    selection_scope="group",
    grouping="histogram",
    match_predicate=detect_histogram_pattern,
)
```

### Phase 3: Advanced Strategies (v2)
```python
# Upgrade specific types to path_based or pixel_perfect
PLOT_TYPE_REGISTRY["line"] = PlotTypeConfig(
    name="line",
    strategy="path_based",
    path_sampling=50,
)
```

---

## File Structure (Future)

```
src/scitex/plt/utils/hitmap/
├── __init__.py           # Public API
├── _registry.py          # PlotTypeConfig, PLOT_TYPE_REGISTRY
├── _detection.py         # get_all_artists, type detection
├── _grouping.py          # Logical group detection
├── _strategies.py        # bbox_only, path_based, pixel_perfect
├── _rendering.py         # apply_hitmap_colors
└── _export.py            # generate_hitmap
```

All files < 256 lines.

---

## Quick Fix for Slow Heatmaps (Immediate)

```python
# In current _hitmap.py
SKIP_PIXEL_PERFECT = {"mesh", "image", "contour", "fill"}

def apply_hitmap_colors(fig, ...):
    for artist, ax_idx, artist_type in artists:
        if artist_type in SKIP_PIXEL_PERFECT:
            # Use bbox only, skip expensive rendering
            color_map[id]["strategy"] = "bbox_only"
            continue
        # Normal processing...
```

---

## Version

- scitex.hitmap.registry: v0.1
- Last updated: 2025-12-13
