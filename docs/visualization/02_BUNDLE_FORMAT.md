# Bundle Format

## Structure

```
{name}.pltz/
├── spec.json           # Plot specification
├── data_info.json      # Column metadata
├── encoding.json       # Data-to-visual mapping
├── theme.json          # Visual styling (editable)
├── cache/              # Computed geometry
│   └── coordinates.json
├── children/           # Child bundles (for .figz)
└── exports/
    ├── figure.png
    └── figure.svg
```

## Bundle Types

| Type | Legacy | Preferred | Use |
|------|--------|-----------|-----|
| Plot | `.pltz` | `.plot.zip` | Single plot |
| Figure | `.figz` | `.figure.zip` | Multi-panel |
| Stats | `.statsz` | `.stats.zip` | Statistics |

Both formats supported. `.zip` variants are OS-friendly (double-click to open).

## Key Files

### spec.json
```json
{
  "kind": "plot",
  "id": "unique_id",
  "type": "line"
}
```

### theme.json (editable)
```json
{
  "colors": {"palette": ["#1f77b4", "#ff7f0e"]},
  "typography": {"family": "Arial", "size_pt": 7},
  "lines": {"width_mm": 0.2}
}
```

### encoding.json
```json
{
  "traces": [{
    "id": "data",
    "x": {"field": "time"},
    "y": {"field": "value"}
  }]
}
```

## API

```python
import scitex.io as sio

# Save
sio.save(fig, "plot.pltz")

# Load
bundle = sio.load("plot.pltz")
bundle.theme  # Access theme
bundle.data   # Access data
```

<!-- EOF -->
