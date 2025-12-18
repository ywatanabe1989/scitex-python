# Migration Guide: .stx Unified Bundle Format

This guide helps you migrate from legacy bundle formats (`.figz`, `.pltz`, `.statsz`) to the unified `.stx` format introduced in SciTeX v2.0.0.

## Overview

SciTeX v2.0.0 introduces a unified `.stx` bundle format that replaces the separate `.figz`, `.pltz`, and `.statsz` extensions. The new format provides:

- **Unified extension**: All bundles use `.stx`
- **Type discrimination**: Bundle type stored in `spec.json["type"]`
- **Self-recursive figures**: Figures can contain other figures
- **Safety constraints**: Depth limits and circular reference detection
- **Unique IDs**: Each bundle has a UUID for tracking

## Timeline

| Version | Status |
|---------|--------|
| v2.0.0 | `.stx` introduced, legacy formats fully supported |
| v2.x | Deprecation warnings for legacy formats |
| v3.0.0 | Legacy format support removed |

## Quick Migration

### Converting Files

Use the CLI tool to convert legacy bundles:

```bash
# Single file
scitex convert file old_figure.figz

# Batch conversion
scitex convert batch ./figures/*.figz

# With custom output directory
scitex convert batch ./figures/*.figz -o ./converted/

# Dry run (see what would happen)
scitex convert batch ./**/*.figz --dry-run
```

### Validating Bundles

```bash
# Validate converted bundles
scitex convert validate output.stx

# Show bundle info
scitex convert info output.stx
```

## Code Changes

### Creating Bundles

**Before (v1.x):**
```python
from scitex.fig import Figz
from scitex.plt import Pltz
from scitex.stats import Statsz

# Old: Explicit legacy extension
figz = Figz.create("figure.figz", "Figure1")
pltz = Pltz.create("plot.pltz", plot_type="line")
statsz = Statsz.create("results.statsz")
```

**After (v2.0.0):**
```python
from scitex.fig import Figz
from scitex.plt import Pltz
from scitex.stats import Statsz

# New: Uses .stx by default
figz = Figz.create("figure.stx", "Figure1")
pltz = Pltz.create("plot.stx", plot_type="line")
statsz = Statsz.create("results.stx")

# Or without extension (auto-adds .stx)
figz = Figz.create("figure", "Figure1")
```

### Loading Bundles

No changes required - both formats load automatically:

```python
# Both work
figz = Figz("figure.stx")    # New format
figz = Figz("figure.figz")   # Legacy format (auto-normalized)
```

### Self-Recursive Figures (New in v2.0.0)

Figures can now contain other figures:

```python
from scitex.fig import Figz

# Create a sub-figure
sub_fig = Figz.create("subfigure.stx", "SubFigure")
sub_fig.add_panel("A", plot_bytes)
sub_fig.save()

# Read sub-figure bytes
with open("subfigure.stx", "rb") as f:
    sub_fig_bytes = f.read()

# Add to main figure
main_fig = Figz.create("main.stx", "MainFigure")
main_fig.add_child_figure("inset", sub_fig_bytes,
                          position={"x_mm": 100, "y_mm": 50})
main_fig.save()

# List children
print(main_fig.list_child_ids())  # ['inset']
```

### Bundle Properties (New in v2.0.0)

```python
figz = Figz.create("figure.stx", "Figure1")

# New properties
print(figz.bundle_id)    # UUID: 'a1b2c3d4-...'
print(figz.depth)        # Current nesting depth: 0
print(figz.max_depth)    # Max allowed depth: 3
print(figz.elements)     # v2.0.0 element list
```

## Schema Changes

### v1.0.0 (Legacy)

```json
{
  "schema": {
    "name": "scitex.fig.figure",
    "version": "1.0.0"
  },
  "figure": {"id": "Figure1", "title": "Figure1"},
  "panels": [...]
}
```

### v2.0.0 (Unified)

```json
{
  "schema": {
    "name": "scitex.bundle",
    "version": "2.0.0"
  },
  "type": "figure",
  "bundle_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "constraints": {
    "allow_children": true,
    "max_depth": 3
  },
  "title": "Figure1",
  "elements": [...],
  "panels": [...]
}
```

### Key Changes

| Field | v1.0.0 | v2.0.0 |
|-------|--------|--------|
| Schema name | `scitex.fig.figure` | `scitex.bundle` |
| Type | Determined by extension | `spec["type"]` field |
| Bundle ID | Not present | UUID in `bundle_id` |
| Constraints | Not present | `constraints` object |
| Elements | `panels` array | `elements` + `panels` |
| Metadata (stats) | `metadata` | `provenance` |

## Deprecation Warnings

Starting in v2.x, you'll see deprecation warnings when:

1. Using `use_stx=False`:
   ```python
   # Triggers warning
   Figz.create("test", "Figure1", use_stx=False)
   ```

2. Using legacy extensions explicitly:
   ```python
   # Triggers warning
   Figz.create("figure.figz", "Figure1")
   ```

### Suppressing Warnings

For gradual migration, you can suppress warnings:

```python
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    figz = Figz.create("legacy.figz", "Figure1")
```

## Troubleshooting

### "Bundle not found" Error

Ensure the file exists and has a valid extension:
```python
from pathlib import Path
path = Path("myfile.stx")
print(f"Exists: {path.exists()}")
print(f"Extension: {path.suffix}")
```

### Invalid Schema Error

Check the spec.json inside the bundle:
```bash
scitex convert info problematic.stx
```

### Depth Limit Error

Reduce nesting depth or increase limit:
```python
figz._spec["constraints"]["max_depth"] = 5
figz.save()
```

### Circular Reference Error

Bundles cannot contain themselves. Check bundle_ids:
```python
print(f"Parent ID: {parent.bundle_id}")
print(f"Child ID: {child.bundle_id}")
```

## FAQ

**Q: Do I need to convert all my files immediately?**
A: No. Legacy formats are fully supported in v2.x. You can convert gradually.

**Q: Will my existing code break?**
A: No. All APIs are backward compatible. Only deprecation warnings are new.

**Q: Can I mix .stx and legacy files?**
A: Yes. The loader auto-detects format and normalizes internally.

**Q: How do I check if a file is v2.0.0?**
A: Use `scitex convert info file.stx` or check `spec["schema"]["version"]`.

## Support

- Issues: https://github.com/ywatanabe/scitex-code/issues
- Documentation: See `docs/STX_MIGRATION_PLAN.md` for technical details
