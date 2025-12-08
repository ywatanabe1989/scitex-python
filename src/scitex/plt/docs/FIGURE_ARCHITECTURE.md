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

```json
{
  "metadata_version": "1.1.0",
  "scitex": {
    "version": "2.4.3",
    "created_at": "2025-12-08T15:40:58.453762",
    "created_with": "scitex.plt.subplots (mm-control)",
    "mode": "publication",
    "axes_size_mm": [40, 28],
    "position_in_grid": [0, 0],
    "style_mm": {
      "axis_thickness_mm": 0.2,
      "tick_length_mm": 0.8,
      "tick_thickness_mm": 0.2,
      "trace_thickness_mm": 0.2,
      "marker_size_mm": 0.8,
      "axis_font_size_pt": 7,
      "tick_font_size_pt": 7,
      "title_font_size_pt": 8,
      "legend_font_size_pt": 6,
      "font_family": "Arial",
      "n_ticks": 4
    }
  },
  "matplotlib": {
    "version": "3.10.3"
  },
  "id": "01_plot",
  "dimensions": {
    "figure_size_mm": [80.0, 68.0],
    "figure_size_inch": [3.15, 2.68],
    "figure_size_px": [944, 803],
    "axes_size_mm": [40.0, 28.0],
    "axes_size_inch": [1.57, 1.10],
    "axes_size_px": [472, 330],
    "axes_position": [0.25, 0.29, 0.5, 0.41],
    "dpi": 300
  },
  "margins_mm": {
    "left": 20.0,
    "bottom": 20.0,
    "right": 20.0,
    "top": 20.0
  },
  "axes_bbox_px": {
    "x0": 236,
    "y0": 236,
    "x1": 708,
    "y1": 566,
    "width": 472,
    "height": 330
  },
  "axes_bbox_mm": {
    "x0": 20.0,
    "y0": 20.0,
    "x1": 60.0,
    "y1": 48.0,
    "width": 40.0,
    "height": 28.0
  },
  "axes": {
    "x": {
      "label": "Time",
      "unit": "s",
      "scale": "linear",
      "lim": [-0.31, 6.60],
      "n_ticks": 4
    },
    "y": {
      "label": "Amplitude",
      "unit": "a.u.",
      "scale": "linear",
      "lim": [-1.10, 1.10],
      "n_ticks": 4
    }
  },
  "title": "ax.plot(x, y)",
  "plot_type": "line",
  "method": "plot",
  "traces": [
    {
      "id": "sine",
      "label": "sin(x)",
      "color": "#0000ff",
      "linestyle": "-",
      "linewidth": 0.57,
      "csv_columns": {
        "x": "ax_00_sine_plot_x",
        "y": "ax_00_sine_plot_y"
      }
    },
    {
      "id": "cosine",
      "label": "cos(x)",
      "color": "#ff0000",
      "linestyle": "--",
      "linewidth": 0.57,
      "csv_columns": {
        "x": "ax_00_cosine_plot_x",
        "y": "ax_00_cosine_plot_y"
      }
    }
  ],
  "legend": {
    "visible": true,
    "loc": 0,
    "frameon": false,
    "labels": ["sin(x)", "cos(x)"]
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

## Summary

| Aspect      | Description                        |
|-------------|------------------------------------|
| Output      | PNG + JSON + CSV per figure        |
| Units       | Millimeters for publication        |
| Tracking    | All traces tracked with IDs        |
| Metadata    | Dimensions, styles, axes info      |
| Data        | CSV with column references in JSON |
| Integration | Direct use as canvas panels        |

<!-- EOF -->