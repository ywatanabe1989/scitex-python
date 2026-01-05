<!-- ---
!-- Timestamp: 2025-12-20 07:20:45
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/README.md
!-- --- -->

# SciTeX FTS (Figure-Table-Statistics)

**FTS is the single source of truth for bundle schemas in SciTeX.**

FTS defines a standardized, self-contained format for reproducible scientific figures with:
- Complete data provenance
- Statistical analysis results
- Visual encoding specifications
- Theme/aesthetic configuration

## Design Philosophy

### Separation of Concerns

FTS strictly separates:

| Layer | File | Purpose | Affects Science? |
|-------|------|---------|------------------|
| **Node** | `spec.json` | Structure (id, type, bbox, children) | Yes |
| **Encoding** | `encoding.json` | Data-to-Visual mapping (columns, scales) | Yes |
| **Theme** | `theme.json` | Aesthetics (colors, fonts, lines) | No |
| **Stats** | `stats.json` | Statistical results with provenance | Yes |
| **Data Info** | `data_info.json` | Column metadata, units, data source | Yes |

**Key insight**: Theme changes don't affect reproducibility. Encoding changes do.

### Canonical vs Artifacts

```
bundle_root/
├── canonical/           # Source of truth (editable, human-readable)
│   ├── spec.json        # Main specification
│   ├── data.csv         # Source data
│   ├── encoding.json    # Data-to-visual mappings
│   ├── theme.json       # Visual aesthetics
│   ├── stats.json       # Statistical results
│   ├── data_info.json   # Column metadata
│   └── runtime.json     # Runtime configuration
├── artifacts/           # Derived/cached (can be deleted and regenerated)
│   ├── cache/
│   │   ├── geometry_px.json
│   │   └── render_manifest.json
│   └── exports/
│       ├── figure.svg
│       ├── figure.png
│       └── figure.pdf
└── children/            # Child bundles (for multi-panel figures)
    ├── panel_a/
    └── panel_b/
```

## Quick Start

```python
from scitex import fts

# Create a new bundle
bundle = fts.FTS("my_plot.zip", create=True, node_type="plot", name="My Plot")

# Set encoding (data-to-visual mapping)
bundle.encoding = {
    "traces": [
        {
            "trace_id": "main",
            "data_ref": "data/experiment.csv",
            "x": {"column": "time", "scale": "linear"},
            "y": {"column": "amplitude", "scale": "log"},
            "color": {"column": "condition"},
        }
    ]
}

# Set theme (pure aesthetics)
bundle.theme = {
    "colors": {"palette": ["#1f77b4", "#ff7f0e", "#2ca02c"]},
    "typography": {"family": "Arial", "size_pt": 10},
    "lines": {"width_pt": 1.5},
}

# Save
bundle.save()
```

## Module Structure

```
scitex/fts/
├── __init__.py       # Public API exports
├── _bundle.py        # FTS class
├── _models.py        # Core models (Node, BBox, SizeMM, Axes)
├── _encoding.py      # Encoding models (TraceEncoding, ChannelEncoding)
├── _theme.py         # Theme models (Theme, Colors, Typography)
├── _stats.py         # Stats models (Analysis, StatResult, EffectSize)
├── _data_info.py     # Data info models (DataInfo, ColumnDef)
├── _validation.py    # JSON schema validation
├── _conversion.py    # scitex to FTS format conversion
└── schemas/          # JSON Schema definitions
    ├── node.schema.json
    ├── encoding.schema.json
    ├── theme.schema.json
    ├── stats.schema.json
    └── data_info.schema.json
```

## Core Models

### Node (spec.json)

Structural metadata for the bundle:

```python
from scitex.fts import Node, BBox, SizeMM

node = Node(
    id="plot_001",
    type="plot",  # figure, plot, text, shape, image
    name="Figure 1A",
    bbox_norm=BBox(x0=0, y0=0, x1=1, y1=1),
    size_mm=SizeMM(width=85, height=60),  # Single column
    children=["panel_a", "panel_b"],  # For figures
)
```

### Encoding (encoding.json)

Data-to-visual channel mappings:

```python
from scitex.fts import Encoding, TraceEncoding, ChannelEncoding

encoding = Encoding(
    traces=[
        TraceEncoding(
            trace_id="line_1",
            data_ref="data/timeseries.csv",
            x=ChannelEncoding(column="time_ms", scale="linear"),
            y=ChannelEncoding(column="voltage_mV", scale="linear"),
            color=ChannelEncoding(column="channel", scale="categorical"),
        )
    ]
)
```

### Theme (theme.json)

Pure aesthetics (doesn't affect scientific meaning):

```python
from scitex.fts import Theme, Colors, Typography

theme = Theme(
    colors=Colors(
        palette=["#1f77b4", "#ff7f0e", "#2ca02c"],
        background="#ffffff",
        text="#000000",
    ),
    typography=Typography(
        family="Arial",
        size_pt=8,
        title_size_pt=10,
    ),
    preset="nature",  # Named presets: nature, science, dark
)
```

### Stats (stats/stats.json)

Statistical analysis results with full provenance:

```python
from scitex.fts import Stats, Analysis, StatMethod, StatResult, EffectSize

stats = Stats(
    analyses=[
        Analysis(
            result_id="ttest_1",
            method=StatMethod(
                name="t-test",
                variant="independent",
                parameters={"equal_var": False},
            ),
            inputs={
                "groups": ["control", "treatment"],
                "n_per_group": [30, 28],
            },
            results=StatResult(
                statistic=2.45,
                statistic_name="t",
                p_value=0.018,
                df=56,
                effect_size=EffectSize(
                    name="cohens_d",
                    value=0.65,
                    ci_lower=0.12,
                    ci_upper=1.18,
                ),
            ),
        )
    ],
    software={"python": "3.11", "scipy": "1.11.0"},
)
```

## Validation

Validate bundles against JSON schemas:

```python
from scitex.fts import validate_bundle, validate_node

# Validate entire bundle
errors = validate_bundle("my_figure.zip")
if errors:
    for filename, error_list in errors.items():
        print(f"{filename}: {error_list}")

# Validate individual components
node_errors = validate_node({"id": "test", "type": "plot", "bbox_norm": {...}})
```

## Conversion Utilities

Convert between scitex internal format and FTS format:

```python
from scitex.fts import from_scitex_spec, to_scitex_spec

# scitex spec to FTS format
fts_data = from_scitex_spec(spec_dict, style_dict)
# Returns: {"node": {...}, "encoding": {...}, "theme": {...}}

# FTS bundle to scitex format
spec, style = to_scitex_spec(bundle)
```

## Backward Compatibility

FTS is also available via the legacy import path:

```python
# Old path (still works)
from scitex import fsb  # Alias for fts

# New path (preferred)
from scitex import fts
```

## Best Practices

1. **Always specify units in data_info.json** - Critical for reproducibility
2. **Use encoding for data-driven styling** - Color by condition, size by value
3. **Keep theme separate from encoding** - Enables style changes without data loss
4. **Include stats provenance** - Software versions, parameters, correction methods
5. **Use canonical/ for source files** - artifacts/ can be regenerated

<!-- EOF -->