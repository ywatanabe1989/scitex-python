<!-- ---
!-- Timestamp: 2025-12-08 16:05:55
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/docs/FIGURE_ARCHITECTURE.md
!-- --- -->

# Figure Architecture for scitex.plt

## Terminology

| Term       | Meaning                      | In Code                        |
|------------|------------------------------|--------------------------------|
| **Figure** | A matplotlib figure object   | `fig, ax = stx.plt.subplots()` |
| **Axes**   | A single subplot/axes        | `ax.plot(x, y)`                |
| **Panel**  | Used in `scitex.vis` context | See CANVAS_ARCHITECTURE.md     |

## Output Format

`stx.plt` outputs **3 files per figure**:

```
output_dir/
├── 01_plot.png      # Rendered image
├── 01_plot.json     # Metadata (dimensions, axes, traces, styles)
└── 01_plot.csv      # Raw data (columns referenced in JSON)
```

### Save Patterns

**Flat (default):**
```python
fig.savefig("./01_plot.png")
# Creates: 01_plot.png, 01_plot.json, 01_plot.csv
```

**Organized by extension:**
```python
fig.savefig("./png/01_plot.png")
# Creates: png/01_plot.png, json/01_plot.json, csv/01_plot.csv
```

## JSON Schema (panel.json)

The schema is organized into clear, non-overlapping sections:

| Section   | Purpose                                                    |
|-----------|------------------------------------------------------------|
| `runtime` | Software versions and creation metadata                    |
| `figure`  | Figure-level properties (size, dpi, mode)                  |
| `axes`    | Per-axes properties nested under `ax_00`, `ax_01`, etc.    |
| `style`   | Hierarchical styling (axes, ticks, traces, markers, fonts, padding) |
| `plot`    | Plot content (title, type, traces, legend)                 |
| `data`    | CSV linkage (path, hash, column names)                     |

```json
{
  "scitex_schema": "scitex.plt.figure",
  "scitex_schema_version": "0.1.0",
  "figure_uuid": "a3b8f2c1-7d4e-4a9b-b5c6-8e2f1a9d0c3b",

  "runtime": {
    "scitex_version": "2.6.0",
    "matplotlib_version": "3.10.3",
    "created_at": "2025-12-09T15:40:58.453762",
    "created_with": "scitex.plt.subplots"
  },

  "figure": {
    "size_mm": [80.00, 68.00],
    "size_inch": [3.150, 2.677],
    "size_px": [944, 803],
    "dpi": 300,
    "mode": "publication"
  },

  "axes": {
    "ax_00": {
      "size_mm": [40.00, 28.00],
      "size_inch": [1.575, 1.102],
      "size_px": [472, 330],
      "position_ratio": [0.250, 0.294, 0.500, 0.412],
      "position_in_grid": [0, 0],
      "margins_mm": {
        "left": 20.00, "bottom": 20.00, "right": 20.00, "top": 20.00
      },
      "margins_inch": {
        "left": 0.787, "bottom": 0.787, "right": 0.787, "top": 0.787
      },
      "bbox_mm": {
        "x_left": 20.00, "x_right": 60.00, "y_top": 20.00, "y_bottom": 48.00,
        "width": 40.00, "height": 28.00
      },
      "bbox_inch": {
        "x_left": 0.787, "x_right": 2.362, "y_top": 0.787, "y_bottom": 1.890,
        "width": 1.575, "height": 1.102
      },
      "bbox_px": {
        "x_left": 236, "x_right": 708, "y_top": 236, "y_bottom": 566,
        "width": 472, "height": 330
      },
      "x_axis_bottom": {
        "label": "Time",
        "unit": "s",
        "scale": "linear",
        "lim": [-0.31, 6.60],
        "n_ticks": 4
      },
      "y_axis_left": {
        "label": "Amplitude",
        "unit": "a.u.",
        "scale": "linear",
        "lim": [-1.10, 1.10],
        "n_ticks": 4
      }
    }
  },

  "style": {
    "axes": {
      "thickness_mm": 0.20
    },
    "ticks": {
      "length_mm": 0.80,
      "thickness_mm": 0.20,
      "n_ticks": 4
    },
    "traces": {
      "thickness_mm": 0.20
    },
    "markers": {
      "size_mm": 0.80
    },
    "fonts": {
      "axis_size_pt": 7.0,
      "tick_size_pt": 7.0,
      "title_size_pt": 8.0,
      "legend_size_pt": 6.0,
      "family_requested": "Arial",
      "family_actual": "Arial"
    },
    "padding": {
      "label_pt": 0.5,
      "tick_pt": 2.0,
      "title_pt": 1.0
    }
  },

  "plot": {
    "title": "ax.plot(x, y)",
    "type": "line",
    "method": "plot",
    "traces": [
      {
        "id": "sine",
        "label": "sin(x)",
        "color": "#0000ff",
        "linestyle": "-",
        "linewidth": 0.57,
        "csv_columns": {
          "x": "ax_00_plot_0_plot_x",
          "y": "ax_00_plot_0_plot_y"
        }
      },
      {
        "id": "cosine",
        "label": "cos(x)",
        "color": "#ff0000",
        "linestyle": "--",
        "linewidth": 0.57,
        "csv_columns": {
          "x": "ax_00_plot_1_plot_x",
          "y": "ax_00_plot_1_plot_y"
        }
      }
    ],
    "legend": {
      "visible": true,
      "loc": 0,
      "frameon": false,
      "labels": ["sin(x)", "cos(x)"]
    }
  },

  "data": {
    "csv_path": "01_plot.csv",
    "csv_hash": "b6e0de1a9755c201",
    "columns": [
      {"id": "sine", "method": "plot", "columns": ["ax_00_plot_0_plot_x", "ax_00_plot_0_plot_y"]},
      {"id": "cosine", "method": "plot", "columns": ["ax_00_plot_1_plot_x", "ax_00_plot_1_plot_y"]}
    ],
    "columns_actual": [
      "ax_00_plot_0_plot_x", "ax_00_plot_0_plot_y",
      "ax_00_plot_1_plot_x", "ax_00_plot_1_plot_y"
    ]
  }
}
```

## CSV Format

Column naming convention: `ax_{ax_idx}_{trace_id}_{method}_{dim}`

```csv
ax_00_sine_plot_x,ax_00_sine_plot_y,ax_00_cosine_plot_x,ax_00_cosine_plot_y
0.0,0.0,0.0,1.0
0.063,0.063,0.063,0.998
0.127,0.127,0.127,0.992
...
```

## Key Features

### Publication Mode

```python
fig, ax = stx.plt.subplots(
    fig_mm={"width": 80, "height": 68},
    axes_mm={"width": 40, "height": 28},
    mode="publication"
)
```

- All dimensions in **millimeters** for publication standards
- Consistent styling across figures
- Automatic metadata embedding

### Automatic Export

On `fig.savefig()`:
1. PNG/PDF/SVG rendered
2. JSON metadata exported
3. CSV data exported

### Trace Tracking

All plotting calls are tracked:

```python
ax.plot(x, y, id="sine", label="sin(x)")  # Tracked
ax.scatter(x, y, id="points")              # Tracked
ax.stx_line(x, y, id="trace")              # Tracked
```

## Supported Plot Methods

### Standard Matplotlib
- `plot`, `scatter`, `bar`, `barh`
- `hist`, `hist2d`, `hexbin`
- `boxplot`, `violinplot`
- `fill_between`, `fill_betweenx`
- `errorbar`, `contour`, `contourf`
- `imshow`, `matshow`, `pie`
- `quiver`, `streamplot`
- `stem`, `step`, `eventplot`

### SciTeX Extensions
- `stx_line`, `stx_shaded_line`
- `stx_mean_std`, `stx_mean_ci`, `stx_median_iqr`
- `stx_kde`, `stx_ecdf`
- `stx_box`, `stx_violin`
- `stx_bar`, `stx_barh`
- `stx_scatter`, `stx_scatter_hist`
- `stx_heatmap`, `stx_conf_mat`
- `stx_image`, `stx_imshow`
- `stx_fillv`, `stx_contour`
- `stx_raster`, `stx_joyplot`
- `stx_rectangle`

### Seaborn Integration
- `sns_lineplot`, `sns_scatterplot`
- `sns_barplot`, `sns_boxplot`, `sns_violinplot`
- `sns_stripplot`, `sns_swarmplot`
- `sns_histplot`, `sns_kdeplot`
- `sns_heatmap`, `sns_jointplot`, `sns_pairplot`

## Integration with scitex.vis

stx.plt outputs can be used as `scitex` type panels in a canvas:

```
canvas/panels/panel_a/
├── panel.json    # Renamed from 01_plot.json
├── panel.csv     # Renamed from 01_plot.csv
└── panel.png     # Renamed from 01_plot.png
```

The JSON structure is compatible - canvas.json references the panel data via relative paths with hash verification.

## Numeric Precision

All numeric values in JSON are rounded to appropriate precision:

| Value Type    | Precision | Example              |
|---------------|-----------|----------------------|
| mm values     | 2 decimal | `40.0`, `28.0`       |
| inch values   | 3 decimal | `3.15`, `1.575`      |
| position      | 3 decimal | `0.25`, `0.294`      |
| axis limits   | 2 decimal | `-0.31`, `6.6`       |
| linewidth     | 2 decimal | `0.57`               |
| px values     | integer   | `944`, `472`         |
| font sizes    | 1 decimal | `7`, `8`             |

## Summary

| Aspect      | Description                        |
|-------------|------------------------------------|
| Output      | PNG + JSON + CSV per figure        |
| Units       | Millimeters for publication        |
| Tracking    | All traces tracked with IDs        |
| Metadata    | Dimensions, styles, axes info      |
| Data        | CSV with column references in JSON |
| Precision   | Appropriate rounding per value type|
| Integration | Direct use as canvas panels        |

<!-- EOF -->