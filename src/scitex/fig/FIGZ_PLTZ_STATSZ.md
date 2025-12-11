<!-- ---
!-- Timestamp: 2025-12-12 09:46:19
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/FIGZ_PLTZ_STATSZ.md
!-- --- -->

# SciTeX Figure/Plot/Stats Bundle Specification (FIGZ / PLTZ / STATSZ)

## Terminology

```
Figure (.figz) = Publication Figure (e.g., "Figure 1")
└── Panel(s) (A, B, C...)
    └── Plot(s) (.pltz) ← contains plot spec + data + annotations
```

- **Figure** = A complete publication figure consisting of one or more panels (Figure 1, Figure 2...).
- **Panel** = A labeled region (A, B, C) within a figure.
- **Plot** = A `.pltz` bundle containing plot specification, data, annotations, and exports.

---

# Rules

1. **A matplotlib figure never splits axes across Panels.**
   One `.pltz` → one Panel. If you need multiple subplots, they stay within the same Panel.

2. **A Panel can have multiple Plots.**
   e.g., a line plot (`.pltz`) + overlay plot.

3. **Data is immutable.**
   `plot.csv` inside `.pltz` is never modified after creation.

---

# File Type Overview

SciTeX uses **bundle formats** for structured reproducibility:

| Bundle        | Meaning                    | Contents                       | Editable |
|---------------|----------------------------|--------------------------------|----------|
| **`.figz`**   | Publication Figure Bundle  | figure.json + exports + plots  | Yes      |
| **`.pltz`**   | Reproducible Plot Bundle   | plot.json + plot.csv + exports | Yes      |
| **`.statsz`** | Statistical Results Bundle | stats.json + test outputs      | Yes      |

## Bundle Naming Convention

To avoid filesystem conflicts between ZIP archives and directories:

- **Working directory**: `Figure1.figz.d/` (directory with `.d` suffix)
- **Packed archive**: `Figure1.figz` (ZIP file without `.d`)

## Bundle Detection Logic

SciTeX detects bundle type automatically:

```python
def is_bundle(path: str) -> bool:
    """Check if path is a SciTeX bundle (directory or ZIP)."""
    p = Path(path)

    # Directory bundle: ends with .figz.d, .pltz.d, .statsz.d
    if p.is_dir() and p.suffix == ".d":
        stem = p.stem  # e.g., "Figure1.figz"
        return stem.endswith((".figz", ".pltz", ".statsz"))

    # ZIP bundle: ends with .figz, .pltz, .statsz
    if p.is_file() and p.suffix in (".figz", ".pltz", ".statsz"):
        return True

    return False

def load_bundle(path: str) -> Bundle:
    """Load bundle from directory or ZIP transparently."""
    p = Path(path)

    if p.is_dir():
        return Bundle.from_directory(p)
    elif p.is_file():
        # Unpack to temp dir, load, return
        return Bundle.from_zip(p)
```

## API Usage

```python
import scitex as stx
import scitex.plt as splt

# Create a plot
fig, ax = splt.subplots(axes_width_mm=40, axes_height_mm=28)
ax.plot([1, 2, 3], [1, 4, 9])

# Save as image (unified interface)
stx.io.save(fig, "output.png")             # PNG
stx.io.save(fig, "output.jpg")             # JPEG
stx.io.save(fig, "output.svg")             # SVG
stx.io.save(fig, "output.pdf")             # PDF

# Save as .pltz bundle (reproducible plot with data + spec)
splt.save_pltz(fig, "output.pltz.d")       # directory bundle
splt.save_pltz(fig, "output.pltz", as_zip=True)  # ZIP bundle

# Load .pltz bundle
fig, ax, data = splt.load_pltz("output.pltz.d")

# Pack/unpack utilities:
stx.io.pack("Figure1.figz.d/")             # → Figure1.figz (ZIP)
stx.io.unpack("Figure1.figz")              # → Figure1.figz.d/ (dir)
```

---

# Bundle Structure

## `.figz` — Publication Figure Bundle

```
Figure1.figz.d/
 ├── figure.json         # specification (includes panel placements)
 ├── figure.png          # raster export
 ├── figure.svg          # vector export
 ├── figure.pdf          # publication export
 ├── timecourse.pltz.d/
 └── barplot.pltz.d/
```

Contents:

- Layout
- Panel arrangement + captions
- Figure-wide styles
- Notations (arrows, callouts)
- References to `.pltz` bundles

---

## `.pltz` — Reproducible Plot Bundle

```
timecourse.pltz.d/
  ├── plot.json        # specification (includes annotations)
  ├── plot.csv         # raw data (immutable)
  ├── plot.png         # raster export
  ├── plot.svg         # vector export
  ├── plot.pdf         # publication export
  └── meta.json        # optional provenance metadata
```

Required:

- `plot.json`
- `plot.csv`
- `plot.png`

Optional:

- `svg`, `pdf`, `meta.json`

---

## `.statsz` — Statistical Results Bundle

```
comparison.statsz.d/
  ├── stats.json      # effect sizes, CI, p-values
  ├── tests.csv       # raw test summaries
  ├── bootstrap.csv   # optional bootstrap results
  └── meta.json       # optional provenance
```

Purpose:

- Reuse stats across multiple `.pltz` bundles
- Store expensive test results (bootstrap/permutation)
- Ensure full reproducibility

---

# Core Schemas (Updated)

## Figure Spec (`figure.json` inside `.figz`)

```json
{
  "schema": { "name": "scitex.fig.figure", "version": "1.1.0" },
  "figure": {
    "id": "fig1_neural_response",
    "title": "Neural Response Patterns",
    "caption": "Comparison of firing rates.",
    "styles": {
      "size": {"width_mm": 180, "height_mm": 120},
      "background": "#ffffff"
    }
  },
  "panels": [
    {
      "id": "A",
      "label": "A",
      "caption": "Time series",
      "plot": "timecourse.pltz.d",
      "position": {"x_mm": 5, "y_mm": 5},
      "size": {"width_mm": 80, "height_mm": 50}
    },
    {
      "id": "B",
      "label": "B",
      "caption": "Bar comparison",
      "plot": "barplot.pltz.d",
      "position": {"x_mm": 90, "y_mm": 5},
      "size": {"width_mm": 80, "height_mm": 50}
    }
  ],
  "notations": [
    {
      "type": "arrow",
      "from_panel": "A",
      "to_panel": "B",
      "style": {"color": "#333", "width_mm": 0.3}
    }
  ]
}
```

---

## Plot Spec (`plot.json` inside `.pltz`)

```json
{
  "schema": { "name": "scitex.plt.plot", "version": "1.0.0" },
  "backend": "mpl",
  "data": {
    "source": "plot.csv",
    "path": "plot.csv",
    "hash": "sha256:abc123...",
    "columns": ["cos(x)_x", "cos(x)_y", "sin(x)_x", "sin(x)_y"]
  },

  "size": {
    "width_inch": 3.15,
    "height_inch": 2.68,
    "width_mm": 80.0,
    "height_mm": 68.0,
    "width_px": 944,
    "height_px": 803,
    "dpi": 300,
    "crop_margin_mm": 1.0
  },

  "axes": [
    {
      "xlabel": "x (radians)",
      "ylabel": "y",
      "title": "Sine Wave",
      "xlim": [-0.31, 6.60],
      "ylim": [-1.10, 1.10],
      "plot_type": "line",
      "bbox": {
        "x0": 0.25, "y0": 0.2941, "x1": 0.75, "y1": 0.7059,
        "width": 0.5, "height": 0.4118
      },
      "bbox_mm": {
        "x0": 20.0, "y0": 20.0, "x1": 60.0, "y1": 48.0,
        "width": 40.0, "height": 28.0
      },
      "bbox_px": {
        "x0": 236, "y0": 236, "x1": 708, "y1": 566,
        "width": 472, "height": 330
      },
      "axes_width_mm": 40.0,
      "axes_height_mm": 28.0,
      "lines": [
        {
          "label": "cos(x)",
          "x_col": "cos(x)_x",
          "y_col": "cos(x)_y",
          "color": "#1f77b4",
          "linewidth": 1.0
        },
        {
          "label": "sin(x)",
          "x_col": "sin(x)_x",
          "y_col": "sin(x)_y",
          "color": "#ff7f0e",
          "linewidth": 1.0
        }
      ]
    }
  ],

  "theme": {
    "mode": "light",
    "colors": {
      "background": "transparent",
      "axes_bg": "transparent",
      "text": "black",
      "spine": "black",
      "tick": "black"
    }
  },

  "stats": {
    "source": "comparison.statsz",
    "elements": [
      {
        "type": "bracket",
        "groups": ["Control", "Treatment"],
        "p_value": 0.003,
        "display": "**"
      }
    ]
  },

  "hit_regions": {
    "figure_px": {"x0": 0, "y0": 0, "x1": 944, "y1": 803},
    "axes": [
      {
        "index": 0,
        "bbox_px": {"x0": 236, "y0": 236, "x1": 708, "y1": 566, "width": 472, "height": 330},
        "title_px": {"x0": 350, "y0": 180, "x1": 594, "y1": 210},
        "xlabel_px": {"x0": 350, "y0": 590, "x1": 594, "y1": 620},
        "ylabel_px": {"x0": 160, "y0": 350, "x1": 190, "y1": 450},
        "legend_px": {"x0": 600, "y0": 250, "x1": 700, "y1": 300},
        "xaxis_px": {"x0": 236, "y0": 566, "x1": 708, "y1": 590},
        "yaxis_px": {"x0": 210, "y0": 236, "x1": 236, "y1": 566},
        "artists": [
          {
            "type": "line",
            "id": "line_0",
            "label": "cos(x)",
            "bbox_px": {"x0": 236, "y0": 236, "x1": 708, "y1": 566},
            "data_ref": {"x_col": "cos(x)_x", "y_col": "cos(x)_y"}
          },
          {
            "type": "line",
            "id": "line_1",
            "label": "sin(x)",
            "bbox_px": {"x0": 236, "y0": 300, "x1": 708, "y1": 500},
            "data_ref": {"x_col": "sin(x)_x", "y_col": "sin(x)_y"}
          }
        ]
      }
    ]
  }
}
```

### Key Fields

**Data fields:**
- **`data.source`**: Filename of data source (always "plot.csv")
- **`data.path`**: Path to CSV file within the bundle (always "plot.csv")
- **`data.hash`**: SHA256 hash of CSV content for integrity verification
- **`data.columns`**: List of column names in the CSV

**Size fields:**
- **`size.width_mm`, `size.height_mm`**: Final cropped figure dimensions
- **`size.width_px`, `size.height_px`**: Figure dimensions in pixels
- **`size.crop_margin_mm`**: Margin preserved during auto-crop (default: 1.0mm)
- **`size.dpi`**: Resolution for raster output (default: 300)

**Axes fields:**
- **`axes[].axes_width_mm`, `axes[].axes_height_mm`**: Individual axis dimensions (default: 40mm × 28mm)
- **`axes[].bbox`**: Normalized bounding box (0-1 figure coordinates)
- **`axes[].bbox_mm`**: Bounding box in millimeters
- **`axes[].bbox_px`**: Bounding box in pixels (for hit detection and snapping)
- **`axes[].lines[]`**: Array of line specifications with `label`, `x_col`, `y_col`, `color`, `linewidth`

### Hit Regions (for Interactive Selection)

The `hit_regions` field provides hierarchical bounding boxes for interactive hit testing:

**Structure:**
```
hit_regions
├── figure_px          # Full figure bounds
└── axes[]             # Per-axes hit regions
    ├── index          # Axes index
    ├── bbox_px        # Plot area bounds
    ├── title_px       # Title text bounds (null if no title)
    ├── xlabel_px      # X-axis label bounds
    ├── ylabel_px      # Y-axis label bounds
    ├── legend_px      # Legend bounds (null if no legend)
    ├── xaxis_px       # X-axis ticks/spine region
    ├── yaxis_px       # Y-axis ticks/spine region
    └── artists[]      # Plot elements (lines, scatter, bars, etc.)
        ├── type       # "line", "scatter", "bar", "fill", "image", etc.
        ├── id         # Unique identifier (e.g., "line_0")
        ├── label      # Display label from legend
        ├── bbox_px    # Bounding box of this artist
        └── data_ref   # Reference to CSV columns {x_col, y_col}
```

**Artist types:**
| Type | Description | bbox_px meaning |
|------|-------------|-----------------|
| `line` | Line plot | Bounding box of all points |
| `scatter` | Scatter plot (group) | Bounding box of all markers |
| `bar` | Bar chart (group) | Bounding box of all bars |
| `fill` | Filled area/polygon | Bounding box of fill region |
| `image` | Heatmap/image | Same as axes bbox |
| `errorbar` | Error bars | Bounding box including error bars |
| `contour` | Contour plot | Same as axes bbox |

### Hit Detection Strategy Candidates

SciTeX supports multiple hit detection strategies. Choose based on your needs:

#### Strategy 1: Bounding Box Only (Simplest)

**How it works:** Each element has a rectangular bounding box. Check if click point is inside.

**Pros:**
- Simple implementation
- Small JSON size
- Fast O(n) lookup

**Cons:**
- Cannot distinguish overlapping elements
- Imprecise for diagonal lines, curves, irregular shapes

**JSON format:**
```json
"artists": [
  {"type": "line", "id": "line_0", "bbox_px": {"x0": 100, "y0": 100, "x1": 500, "y1": 300}}
]
```

**Best for:** Simple figures with non-overlapping elements

---

#### Strategy 2: Normalized Path Data (Reshape-Independent)

**How it works:** Store sampled path points in normalized coordinates (0-1 relative to axes). Client transforms to pixels using axes bbox.

**Pros:**
- Works at any figure size/zoom
- Precise for lines, scatter, polygons
- Moderate JSON size (~20-50 points per element)

**Cons:**
- Client must compute distance to path
- O(n × m) where n=elements, m=points per element
- Complex for some shapes (contours)

**JSON format:**
```json
"artists": [
  {
    "type": "line",
    "id": "line_0",
    "bbox_norm": {"x0": 0, "y0": 0.1, "x1": 1.0, "y1": 0.9},
    "path_norm": [[0, 0.5], [0.25, 0.8], [0.5, 0.3], [0.75, 0.7], [1.0, 0.5]],
    "data_ref": {"x_col": "cos(x)_x", "y_col": "cos(x)_y"}
  }
]
```

**Client-side transform:**
```javascript
function normToPixel(normX, normY, axesBbox) {
  return [
    normX * axesBbox.width + axesBbox.x0,
    (1 - normY) * axesBbox.height + axesBbox.y0  // y flipped
  ];
}
```

**Best for:** Line plots, scatter plots, bar charts with moderate complexity

---

#### Strategy 3: Hit Map Image (Pixel-Perfect)

**How it works:** Generate a hidden PNG where each element is rendered with a unique ID color. On click, read pixel color to identify element.

**Pros:**
- Pixel-perfect accuracy for ANY shape
- O(1) lookup time (just array index)
- Works for complex shapes (contours, overlapping elements)

**Cons:**
- Extra file in bundle (~50-200KB)
- Must regenerate if figure resized
- Slightly slower initial load

**Bundle structure:**
```
plot.pltz.d/
  ├── plot.png           # visible image
  ├── plot_hitmap.png    # hidden hit map (same size)
  └── plot.json
```

**JSON format:**
```json
"hit_regions": {
  "hit_map": "plot_hitmap.png",
  "color_map": {
    "#010000": {"type": "axes", "index": 0},
    "#020000": {"type": "title", "axes_index": 0},
    "#030000": {"type": "line", "id": "line_0", "label": "cos(x)"},
    "#040000": {"type": "line", "id": "line_1", "label": "sin(x)"},
    "#050000": {"type": "contour", "id": "contour_0"}
  }
}
```

**Client-side lookup:**
```javascript
function getHitElement(x, y, hitMapImageData, colorMap) {
  const idx = (y * width + x) * 4;
  const r = hitMapImageData[idx];
  const g = hitMapImageData[idx + 1];
  const b = hitMapImageData[idx + 2];
  const hex = `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
  return colorMap[hex];
}
```

**Best for:** Complex figures with overlapping elements, contours, heatmaps

---

#### Strategy 4: Hybrid (Recommended)

**How it works:** Combine strategies - use bbox for quick rejection, path_norm for common shapes, hit_map for complex figures.

**JSON format:**
```json
"hit_regions": {
  "strategy": "hybrid",
  "hit_map": "plot_hitmap.png",  // optional, for complex figures
  "color_map": {...},            // if hit_map present
  "axes": [
    {
      "bbox_px": {...},
      "title_px": {...},
      "artists": [
        {
          "type": "line",
          "selectable": true,
          "bbox_norm": {...},    // quick rejection
          "path_norm": [...],    // fallback if no hit_map
          "data_ref": {...}
        },
        {
          "type": "contour",
          "selectable": "hit_map",  // requires hit_map
          "bbox_norm": {...}
        }
      ]
    }
  ]
}
```

**`selectable` values:**
- `true` - path-based selection available
- `"bbox"` - bounding box only
- `"hit_map"` - requires hit map for precise selection
- `false` - not selectable

**Client algorithm:**
1. Quick rejection: check `bbox_norm`
2. If `hit_map` exists and loaded → use it
3. Else if `path_norm` exists → compute distance
4. Else → use bbox only

---

#### Strategy Comparison

| Strategy  | Accuracy      | Speed (click)  | JSON Size  | Reshape | Complexity Support |
|-----------|---------------|----------------|------------|---------|--------------------|
| Bbox only | Low           | O(n) fast      | Small      | ✅      | Low                |
| Path norm | High          | O(n×m)         | Medium     | ✅      | Medium             |
| Hit map   | Pixel-perfect | O(1)           | +PNG file  | ❌      | High               |
| Hybrid    | High          | O(1) or O(n×m) | Medium+PNG | Partial | High               |

---

### Hit Map Generation Methods (Experimental Results)

We conducted experiments comparing different methods for generating hit maps. Results are from a complex figure with ~25 artists (lines, scatter, fills, bars, patches, error bars).

#### Method Comparison

| Method | Generation Time | Query Time | Accuracy | Notes |
|--------|-----------------|------------|----------|-------|
| **ID Colors (Single Render)** | **89ms** | O(1) pixel lookup | Pixel-perfect | **RECOMMENDED** |
| Sequential Per-Element | 2968ms | O(1) pixel lookup | Pixel-perfect | 33x slower than ID Colors |
| MPL contains() Full Map | 50466ms | N/A | High | Impractical for full maps |
| Hybrid BBox + contains() | 9032ms | ~0.27ms/query | High | Better for sparse queries |
| Export Path Data | ~192ms | ~0.01ms (client) | High | Best for reshape support |

#### Recommended Methods

**#1 ID Colors Hit Map (Best for Fixed-Size Display)**

Assign unique RGB color to each element, render once, decode by pixel color.

```python
# Assign ID colors (R channel encodes element ID)
for i, artist in enumerate(artists):
    hex_color = f"#{(i+1)*10:02x}0000"  # e.g., #0a0000, #140000, ...
    artist.set_color(hex_color)
    artist.set_antialiased(False)

# Hide non-artist elements
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
# ... hide ticks, labels, legend, spines

# Single render
fig.canvas.draw()
hitmap = np.array(fig.canvas.buffer_rgba())[:, :, 0]  # Red channel = ID
```

- **Generation**: ~89ms (33x faster than sequential)
- **Query**: O(1) - direct pixel lookup
- **Accuracy**: Pixel-perfect
- **Limitation**: Must regenerate if figure resized

---

**#2 Export Path Data (Best for Resizable Web Editors)**

Extract geometry data at save time, do hit testing client-side in JavaScript.

```json
{
  "artists": [
    {
      "id": 0,
      "type": "Line2D",
      "bbox_px": {"x0": 128, "y0": 133, "x1": 692, "y1": 438},
      "path_px": [[128, 285], [134, 295], ...],
      "linewidth": 2.0
    },
    {
      "id": 1,
      "type": "PathCollection",
      "points_px": [[0, 0.5], [0.63, 1.09], ...],
      "sizes": [100]
    },
    {
      "id": 2,
      "type": "BarContainer",
      "bars_data": [
        {"x": 0.85, "y": 0, "width": 0.3, "height": 0.5},
        ...
      ]
    }
  ]
}
```

- **Extraction**: ~192ms one-time
- **Query**: ~0.01ms client-side
- **Reshape**: ✅ Supported (transform coordinates)
- **Best for**: Web editors with zoom/resize

---

**#3 Hybrid BBox + contains() (Best for Server-Side/On-Demand)**

Pre-filter by bounding box, then use Matplotlib's native `contains()` for precision.

```python
# Pre-compute bboxes (once)
bboxes = [artist.get_window_extent(renderer) for artist in artists]

# On click query
candidates = [i for i, bbox in enumerate(bboxes) if bbox.contains(x, y)]
for i in candidates:
    hit, props = artists[i].contains(mouse_event)
    if hit:
        return artists[i]
```

- **Pre-compute**: ~0ms (just bbox extraction)
- **Query**: ~0.27ms per query (70% fewer contains() calls)
- **Reshape**: ✅ Supported (transforms auto-update)
- **Best for**: Interactive Python apps, server-side queries

---

#### Performance Summary

```
Complex Figure (25 artists, 600×500px):

Hit Map Generation:
  ID Colors:     89ms    ← FASTEST (single render)
  Sequential:    2968ms  (33x slower)

On-Demand Query:
  Pixel lookup:  <0.001ms  (requires pre-generated hit map)
  Hybrid:        0.27ms    (bbox filter + contains())
  Path export:   0.01ms    (client-side JavaScript)
```

#### Decision Tree

```
Need hit testing?
  │
  ├─→ Fixed size display (no resize)?
  │     └─→ Use ID Colors Hit Map (89ms gen, O(1) query)
  │
  ├─→ Web editor with resize/zoom?
  │     └─→ Use Export Path Data (192ms gen, client-side)
  │
  └─→ Server-side or Python interactive?
        └─→ Use Hybrid BBox + contains() (0ms gen, 0.27ms query)
```

#### Extractable Artist Types

| Artist Type                   | bbox | path_norm | hit_map |
|-------------------------------|------|-----------|---------|
| Line2D (plot)                 | ✅   | ✅        | ✅      |
| PathCollection (scatter)      | ✅   | ✅        | ✅      |
| Rectangle (bar, hist)         | ✅   | ✅        | ✅      |
| Polygon (fill)                | ✅   | ✅        | ✅      |
| PolyCollection (fill_between) | ✅   | ✅        | ✅      |
| QuadMesh (pcolormesh)         | ✅   | ❌        | ✅      |
| AxesImage (imshow)            | ✅   | ❌        | ✅      |
| ContourSet (contour)          | ✅   | ❌        | ✅      |
| Text                          | ✅   | ❌        | ✅      |

### Theme Support

SciTeX supports dark/light themes for eye-friendly visualization:

```json
{
  "theme": {
    "mode": "dark",
    "colors": {
      "background": "transparent",
      "axes_bg": "#1a1a2e",
      "text": "#e8e8e8",
      "spine": "#4a4a5a",
      "tick": "#e8e8e8"
    }
  }
}
```

---

## Stats Spec (`stats.json` inside `.statsz`)

```json
{
  "schema": { "name": "scitex.stats.stats", "version": "1.0.0" },
  "comparisons": [
    {
      "name": "Control vs Treatment",
      "method": "t-test",
      "p_value": 0.003,
      "effect_size": 1.21,
      "ci95": [0.5, 1.8],
      "formatted": "**"
    }
  ],
  "metadata": {
    "n": 16,
    "bootstrap_iters": 5000,
    "seed": 42
  }
}
```

---

# Style Cascading

```
figure.styles
   ↓ override
panel.styles
   ↓ override
plot.styles
```

---

# Reproducibility Layers

| Layer  | Bundle    | Editable | Purpose                    |
|--------|-----------|----------|----------------------------|
| Data   | `.pltz`   | No       | Immutable CSV              |
| Plot   | `.pltz`   | Yes      | Regenerate plot            |
| Stats  | `.statsz` | Yes      | Reproducible statistics    |
| Figure | `.figz`   | Yes      | Layout, styling, notations |

---

# Summary

SciTeX supports three structured, ZIP-based scientific bundles:

### **`.figz`** — Publication figure

Panels → layout → notations → exports

### **`.pltz`** — Reproducible plot

Data → spec → render

### **`.statsz`** — Statistical results

Effect sizes, CI, p-values, bootstrap logs

Each bundle:

- can be a **directory** (`.d` suffix) or a **ZIP archive**
- is transparent and editable
- is designed for long-term scientific reproducibility
- avoids vendor lock-in

---

# END OF SPEC

<!-- EOF -->