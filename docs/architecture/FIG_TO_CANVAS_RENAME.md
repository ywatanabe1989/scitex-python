# scitex.fig → scitex.canvas Rename

**Status:** APPROVED
**Created:** 2026-01-07
**Author:** ywatanabe + Claude

---

## Decision

**APPROVED:** Rename `scitex.fig` to `scitex.canvas`

---

## Rationale

| Issue | Solution |
|-------|----------|
| `fig` conflicts with matplotlib's `fig` | Use distinct name `canvas` |
| Module purpose unclear from name | `canvas` = composition workspace |
| Terminology confusion | Aligns with TERMINOLOGY_GUIDE.md |

---

## Scope

### Files to Modify

```
~43 files reference scitex.fig

Key changes:
├── src/scitex/fig/           → src/scitex/canvas/
├── imports: scitex.fig       → scitex.canvas
├── tests/scitex/fig/         → tests/scitex/canvas/
└── documentation updates
```

### API Changes

| Before | After |
|--------|-------|
| `import scitex.fig as sfig` | `import scitex.canvas as scanvas` |
| `from scitex.fig import create_canvas` | `from scitex.canvas import create_canvas` |
| `scitex.fig.Canvas` | `scitex.canvas.Canvas` |

---

## Backward Compatibility

```python
# src/scitex/fig/__init__.py (kept for compatibility)
import warnings
warnings.warn(
    "scitex.fig is deprecated. Use scitex.canvas instead.",
    DeprecationWarning,
    stacklevel=2
)
from scitex.canvas import *
```

---

## Implementation Steps

1. Create `src/scitex/canvas/` directory
2. Move all files from `src/scitex/fig/`
3. Update internal imports
4. Create backward-compat `src/scitex/fig/__init__.py`
5. Update tests
6. Update documentation

---

## Timeline

- Implemented in: `refactor/module-terminology` branch
- Deprecation period: Until v2.0
- Removal of `scitex.fig` alias: v2.0

<!-- EOF -->
