<!-- ---
!-- Timestamp: 2025-12-09 20:40:40
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/schema/README.md
!-- --- -->

# SciTeX Schema Module

Central source of truth for cross-module data structures and JSON schemas.

## Overview

This module defines standardized schemas for data that:
- Crosses module boundaries (vis, plt, stats, io, cloud)
- Is serialized to JSON/disk
- Needs version tracking for compatibility

## Schema Types

| Schema                     | Version | Module                                  | Purpose                               |
|----------------------------|---------|-----------------------------------------|---------------------------------------|
| `scitex.schema.canvas`     | 0.1.0   | `_canvas.py`                            | Multi-panel figure composition        |
| `scitex.schema.stats`      | 0.1.0   | `_stats.py`                             | Statistical test results              |
| `scitex.schema.figure`     | 0.1.0   | `__init__.py`                           | Figure specifications (via vis.model) |
| `scitex.plt.figure`        | 0.1.0   | `plt/utils/_collect_figure_metadata.py` | plt module figure metadata            |

## JSON Schema Identifiers

All SciTeX JSON files should include schema identification:

```json
{
  "scitex_schema": "scitex.schema.canvas",
  "scitex_schema_version": "0.1.0",
  ...
}
```

### Schema Naming Convention

- **Field name**: `scitex_schema` (identifies the schema type)
- **Version field**: `scitex_schema_version` (semantic versioning)
- **Format**: `scitex.schema.<type>` where type is one of:
  - `canvas` - Canvas/multi-panel layouts
  - `stats` - Statistical results
  - `figure` - Figure specifications
  - `plt.figure` - plt module figure metadata

## Module Structure

```
scitex/schema/
├── __init__.py      # Public API, re-exports, SCHEMA_VERSION
├── _canvas.py       # Canvas/panel specifications
├── _stats.py        # Statistical result schemas
├── _validation.py   # Validation functions
└── README.md        # This file
```

## Usage

### Import from Central Location

```python
# Recommended: Import from scitex.schema
from scitex.schema import (
    # Figure specs (from vis.model)
    FigureSpec, AxesSpec, PlotSpec,

    # Statistical results
    StatResult, create_stat_result,

    # Canvas specs
    CanvasSpec, PanelSpec,

    # Validation
    validate_figure, validate_stat_result,
)
```

### Creating Statistical Results

```python
from scitex.schema import StatResult, create_stat_result

# Quick creation
result = create_stat_result(
    test_type="pearson",
    statistic_name="r",
    statistic_value=0.85,
    p_value=0.001
)

# Full specification
result = StatResult(
    test_type="t-test",
    test_category="parametric",
    statistic={"name": "t", "value": 3.45},
    p_value=0.002,
    stars="**",
    effect_size={"name": "cohens_d", "value": 0.85}
)

# Format for display
print(result.format_text("publication"))  # (t = 3.45, p < 0.01)
print(result.format_text("compact"))      # t = 3.450**
```

### Creating Canvas Specifications

```python
from scitex.schema import CanvasSpec, PanelSpec

canvas = CanvasSpec(
    canvas_name="figure1",
    size=CanvasSizeSpec(width_mm=180, height_mm=120)
)

panel = PanelSpec(
    name="a",
    type="scitex",
    source="plot_a.png"
)
canvas.panels.append(panel)

# Serialize
json_dict = canvas.to_dict()
```

## Design Principles

1. **Single Source of Truth**: Schema definitions live here, not in consuming modules
2. **Backward Compatibility**: Version fields enable migration
3. **JSON-First**: All schemas are JSON-serializable
4. **GUI Integration**: Schemas support scitex-cloud GUI requirements
5. **Type Safety**: Full type hints and dataclass validation

## Schema Version History

### Canvas Schema (`scitex.schema.canvas`)
- **0.1.0**: Initial release with panel/annotation support

### Stats Schema (`scitex.schema.stats`)
- **0.1.0**: Initial release with GUI positioning support

### Figure Schema (`scitex.schema.figure`)
- **0.1.0**: Re-exports from vis.model

### plt.figure Schema (`scitex.plt.figure`)
- **0.1.0**: Figure metadata with traces, axes, dimensions

## Validation

```python
from scitex.schema import validate_figure, validate_stat_result, ValidationError

try:
    validate_stat_result(result)
except ValidationError as e:
    print(f"Invalid: {e}")
```

## Integration Points

| Consumer Module | Schema Used        | Purpose                 |
|-----------------|--------------------|-------------------------|
| `scitex.plt`    | `plt.figure`       | Figure metadata in JSON |
| `scitex.vis`    | `canvas`, `figure` | Canvas composition      |
| `scitex.stats`  | `stats`            | Test result output      |
| `scitex-cloud`  | All                | GUI rendering           |
| `scitex.bridge` | All                | Cross-module adapters   |

## Related Documentation

- `src/scitex/plt/docs/FIGURE_ARCHITECTURE.md` - plt module figure format
- `src/scitex/vis/docs/CANVAS_ARCHITECTURE.md` - Canvas architecture

<!-- EOF -->