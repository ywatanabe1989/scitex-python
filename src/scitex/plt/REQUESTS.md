# Requests for scitex.plt JSON Schema (scitex.plt.figure.recipe v0.2.0)

These are requests from scitex.vis to ensure the JSON schema is compatible with
the interactive editor and renderer.

## 1. Consistent data_ref Column Naming

**Issue**: The `data_ref` keys in `calls` don't match the actual CSV column names.

**Current**:
```json
"calls": [{
    "data_ref": {
        "x": "ax-row-0-col-0_trace-id-ax_00_ch1_variable-x",
        "y": "ax-row-0-col-0_trace-id-ax_00_ch1_variable-y"
    }
}]
```

**CSV columns** (from `columns_actual`):
```
"ax-row-0-col-0_trace-id-ch1_variable-x"
"ax-row-0-col-0_trace-id-ch1_variable-y"
```

**Note**: The `data_ref` includes `ax_00_` prefix in trace-id but CSV doesn't.

**Request**: Ensure `data_ref` values exactly match CSV column names, OR document
the mapping convention clearly.

---

## 2. Add Panel Titles to Axis Spec

**Current**: No title field in axis spec.

**Request**: Add optional `title` field to each axis:
```json
"ax_00": {
    "title": "A) Multi-channel Recording",
    "grid_position": {...},
    "xaxis": {...},
    ...
}
```

---

## 3. Explicit Legend Configuration Per Axis

**Request**: Add legend configuration to each axis:
```json
"ax_00": {
    "legend": {
        "visible": true,
        "loc": "upper right",
        "frameon": false,
        "fontsize": 6
    }
}
```

---

## 4. Add `method_type` Classification

**Request**: Add a high-level classification for rendering strategies:
```json
"calls": [{
    "id": "ch1",
    "method": "plot",
    "method_type": "line",  // "line", "scatter", "bar", "fill", "image", "annotation"
    ...
}]
```

This helps the editor group similar elements and apply consistent styling.

---

## 5. Color Information in Calls

**Current**: Colors are in `kwargs` but not always present.

**Request**: Ensure color is always included for colored elements:
```json
"calls": [{
    "id": "ch1",
    "method": "plot",
    "kwargs": {
        "color": "#1f77b4",
        "label": "Ch1",
        "linewidth": 0.57
    }
}]
```

---

## 6. Trace Grouping for Related Elements

**Request**: Add optional `group` field to link related traces:
```json
"calls": [
    {"id": "mean-line", "group": "mean-std", "method": "plot", ...},
    {"id": "upper-bound", "group": "mean-std", "method": "fill_between", ...},
    {"id": "lower-bound", "group": "mean-std", "method": "fill_between", ...}
]
```

This allows the editor to select/edit related elements together.

---

## 7. Marker Information for Scatter/Point Plots

**Request**: Include marker style info explicitly:
```json
"kwargs": {
    "marker": "o",
    "markersize": 6,
    "markerfacecolor": "#1f77b4",
    "markeredgecolor": "none"
}
```

---

## 8. Add `editable` Flag

**Request**: Mark which properties are user-editable vs computed:
```json
"calls": [{
    "id": "regression-line",
    "editable": {
        "color": true,
        "linewidth": true,
        "data": false  // computed from scatter data
    }
}]
```

---

## 9. Colorbar Configuration for Heatmaps

**Request**: Add colorbar spec when imshow/heatmap is used:
```json
"ax_02": {
    "colorbar": {
        "visible": true,
        "label": "Intensity [a.u.]",
        "orientation": "vertical"
    }
}
```

---

## 10. Secondary Y-axis Support

**Request**: Support twin axes (twinx/twiny):
```json
"ax_00": {
    "grid_position": {"row": 0, "col": 0},
    "xaxis": {...},
    "yaxis": {...},
    "secondary_yaxis": {
        "label": "Mean Activity",
        "lim": [0, 20],
        "calls": [...]
    }
}
```

---

## Priority

| Request | Priority | Reason |
|---------|----------|--------|
| 1 (Column naming) | HIGH | Breaks data loading |
| 5 (Colors in kwargs) | HIGH | Required for rendering |
| 2 (Panel titles) | MEDIUM | UI feature |
| 3 (Legend config) | MEDIUM | UI feature |
| 4 (method_type) | LOW | Optimization |
| 6-10 | LOW | Future features |

---

*Generated: 2025-12-10 by scitex.vis editor refactoring*
