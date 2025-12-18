<!-- ---
!-- Timestamp: 2025-12-19
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/examples/fig/README.md
!-- --- -->

# scitex.fig Examples

**Examples demonstrating the Unified Element API for `.stx` bundles**

---

## Available Examples

### `unified_element_api.py`

Demonstrates the complete unified element API:

```bash
cd /path/to/scitex-code
python examples/fig/unified_element_api.py
```

**What it covers:**

1. **Creating Figure Bundles**
   - Using `Figz.create()` with custom sizes
   - Understanding bundle properties (ID, type, constraints)

2. **Adding Elements**
   - Text elements with positioning and styling
   - Shape elements (arrows, brackets)
   - Using the unified `add_element()` API

3. **Coordinate System**
   - Origin at top-left (0,0)
   - Positions in millimeters
   - Local vs absolute coordinates

4. **Auto-Layout**
   - Using `auto_layout_grid()` for automatic positioning
   - Grid arrangement of multiple elements

5. **Coordinate Transforms**
   - Converting local to absolute positions with `to_absolute()`
   - Understanding nested coordinate spaces

6. **Save and Reload**
   - Persisting bundles with `.save()`
   - Loading and querying existing bundles

---

## Quick Start

```python
from scitex.fig import Figz, auto_layout_grid, to_absolute

# Create a figure
figz = Figz.create("my_figure.stx", "My Figure")

# Add elements (everything uses add_element!)
figz.add_element("title", "text", "Figure Title", {"x_mm": 85, "y_mm": 5})
figz.add_element("arrow1", "shape", {"shape_type": "arrow", ...})

# Save
figz.save()
```

---

## Coordinate System Reference

```
(0,0) ──────────────────► x_mm
  │
  │   ┌─────────────────────────┐
  │   │  Figure Canvas          │
  │   │  (170mm × 120mm)        │
  │   │                         │
  │   │   Element A (10, 10)    │
  │   │   └── annotation (5, 3) │  ← absolute: (15, 13)
  │   │                         │
  │   └─────────────────────────┘
  ▼
y_mm
```

**Key principle**: Child positions are LOCAL to their parent.

---

## Related Documentation

- [scitex.fig README](../../src/scitex/fig/README.md) - Full API reference
- [scitex.io.bundle README](../../src/scitex/io/bundle/README.md) - Bundle I/O operations
- [Tests](../../tests/scitex/fig/) - Test coverage for the API

<!-- EOF -->
