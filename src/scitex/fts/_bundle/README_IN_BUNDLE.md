# FSB Bundle (Figure-Statistics Bundle)

This is a self-contained scientific figure bundle created with SciTeX.

## Bundle Structure

```
bundle_root/
├── README.md            # This file
├── canonical/           # Source of truth (editable)
│   ├── spec.json        # Main specification (type, id, elements)
│   ├── data.csv         # Source data
│   ├── encoding.json    # Data-to-visual channel mappings
│   ├── theme.json       # Visual aesthetics (colors, fonts)
│   ├── stats.json       # Statistical analysis results
│   ├── data_info.json   # Column metadata and units
│   └── runtime.json     # Runtime configuration
├── artifacts/           # Derived files (can be regenerated)
│   ├── cache/           # Computed values
│   └── exports/         # Rendered outputs (PNG, SVG, PDF)
└── children/            # Child bundles (for multi-panel figures)
```

## Key Files

### canonical/spec.json
Main specification defining the bundle structure:
- `id`: Unique bundle identifier
- `type`: Bundle type (figure, plot, text, etc.)
- `bbox_norm`: Normalized bounding box (0-1 coordinates)
- `size_mm`: Physical size in millimeters
- `elements`: Child elements or references

### canonical/encoding.json
Data-to-visual mappings for scientific reproducibility:
- `traces`: List of data series with column bindings
- Each trace maps data columns to visual channels (x, y, color, size)

### canonical/theme.json
Pure visual aesthetics (can be changed without affecting data):
- `colors`: Color palette and scheme
- `typography`: Font settings
- `lines`: Line styles and widths
- `markers`: Marker styles

### canonical/stats.json
Statistical analysis results with full provenance:
- `analyses`: List of statistical tests performed
- Each analysis includes method, inputs, and results
- `software`: Version information for reproducibility

### canonical/data_info.json
Metadata about data columns:
- Column names, types, and units
- Data source and hash for integrity
- Basic statistics for validation

## Opening This Bundle

### With Python (SciTeX)
```python
from scitex import fsb

# Load the bundle
bundle = fsb.Bundle("path/to/this/bundle.zip")

# Access components
print(bundle.node.name)
print(bundle.encoding)
print(bundle.theme)
```

### With Any JSON Tool
All files in `canonical/` are standard JSON and can be viewed/edited with any text editor or JSON tool.

### Exported Figures
Ready-to-use figures are in `artifacts/exports/`:
- `figure.svg` - Vector format (scalable)
- `figure.png` - Raster format (web/screen)
- `figure.pdf` - Print format

## Reproducibility

This bundle contains everything needed to reproduce the figure:
1. **Data**: Original data in `canonical/data.csv`
2. **Encoding**: How data maps to visuals in `encoding.json`
3. **Stats**: Statistical results with full provenance
4. **Software**: Version info in `stats.json`

To regenerate exports:
```python
from scitex import fsb
bundle = fsb.Bundle("this_bundle.zip")
# Exports will be regenerated on render
```

## License

The figure and data in this bundle are subject to the license terms specified by the original author. Contact information may be found in `spec.json`.

---
*Created with [SciTeX](https://scitex.ai) FSB v1.0.0*
