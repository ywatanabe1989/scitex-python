# FTS Module Renaming Proposal

**Status:** APPROVED - Merge into io.bundle
**Created:** 2026-01-07
**Updated:** 2026-01-07
**Author:** ywatanabe + Claude

---

## Decision

**APPROVED:** Merge `scitex.fts` into `scitex.io.bundle`

- No separate schema module needed
- Bundle type identified by double extension (`.plot.zip`, `.figure.zip`, `.stats.zip`)
- Manifest file: `manifest.json` inside bundle confirms type
- Schemas moved to `scitex.io.bundle.schemas/`

---

## Problem Statement

The module `scitex.fts` has naming issues:

1. **Hard to remember**: "fts" is not intuitive
2. **Unclear meaning**: FTS = "Figure-Table-Statistics" requires explanation
3. **Overlap with io.bundle**: Both deal with bundle schemas
4. **Discoverability**: Users won't guess to `import scitex.fts`

---

## Current State

### What fts Contains

```
src/scitex/fts/
├── _bundle/          # Bundle handling (overlap with io.bundle)
├── _fig/             # Figure-specific schemas
│   └── _editor/      # GUI/CUI editors
├── _kinds/           # Plot type schemas (line, bar, scatter, etc.)
├── _schemas/         # JSON schema definitions
├── _stats/           # Statistics schemas
└── _tables/          # Table schemas
```

### What fts Does

| Component | Purpose |
|-----------|---------|
| Node schema | Structure: id, type, bbox, children |
| Encoding schema | Data-to-visual mapping |
| Theme schema | Aesthetics: colors, fonts, sizes |
| Stats schema | Statistical results with provenance |
| Data info | Column metadata, units |

### Overlap with io.bundle

| Feature | scitex.fts | scitex.io.bundle |
|---------|-----------|------------------|
| Bundle types | Yes (.figz, .pltz) | Yes (.figz, .pltz, .statsz) |
| Schema defs | Yes (JSON schemas) | No |
| Load/save | Partial | Full |
| Validation | Yes | Partial |
| GUI editor | Yes | No |

---

## Renaming Options

### Option A: `scitex.schema`

```python
# Before
from scitex.fts import FTS

# After
from scitex.schema import Schema
from scitex.schema import NodeSchema, EncodingSchema, ThemeSchema
```

| Pros | Cons |
|------|------|
| Clear purpose (schemas) | Generic name |
| Intuitive import | May imply JSON Schema only |
| Matches content |  |

**Recommendation:** Good option

### Option B: `scitex.spec`

```python
# Before
from scitex.fts import FTS

# After
from scitex.spec import Spec
from scitex.spec import NodeSpec, EncodingSpec
```

| Pros | Cons |
|------|------|
| Short, memorable | Confusion with spec.json file |
| "Specification" is accurate | |

**Recommendation:** Acceptable but may confuse

### Option C: `scitex.repro`

```python
# Before
from scitex.fts import FTS

# After
from scitex.repro import Bundle
from scitex.repro import ReproSpec
```

| Pros | Cons |
|------|------|
| Emphasizes reproducibility | Unclear what it contains |
| Unique name | Not widely used term |

**Recommendation:** Not recommended

### Option D: Merge into `scitex.io.bundle`

```python
# Before
from scitex.fts import FTS
from scitex.io.bundle import load, save

# After
from scitex.io.bundle import Bundle
from scitex.io.bundle.schemas import NodeSchema, EncodingSchema
```

| Pros | Cons |
|------|------|
| Eliminates redundancy | Larger module |
| Single location for bundles | Migration effort |
| Cleaner API | |

**Recommendation:** Best long-term option

### Option E: `scitex.bundle` (Top-Level)

```python
# Before
from scitex.fts import FTS
from scitex.io.bundle import load

# After
from scitex.bundle import Bundle, load, save
from scitex.bundle.schemas import NodeSchema
```

| Pros | Cons |
|------|------|
| Promotes bundles as core concept | New top-level module |
| Clean, memorable | Migration from io.bundle |
| Single import location | |

**Recommendation:** Good for emphasizing bundles

---

## Recommendation

### Short Term: Option A (`scitex.schema`)

Rename `fts` to `schema` with minimal disruption:

```python
# Alias for backward compatibility
import scitex.fts as fts  # deprecated
import scitex.schema as schema  # new

# fts/__init__.py
import warnings
warnings.warn(
    "scitex.fts is deprecated, use scitex.schema instead",
    DeprecationWarning
)
from scitex.schema import *
```

### Long Term: Option D (Merge into io.bundle)

After schema rename stabilizes, merge schema into io.bundle:

```
src/scitex/io/bundle/
├── __init__.py        # Bundle class, load(), save()
├── _core.py
├── _types.py
├── _zip.py
├── schemas/           # ← Moved from scitex.schema
│   ├── node.py
│   ├── encoding.py
│   ├── theme.py
│   └── stats.py
└── editor/            # ← Moved from scitex.schema (was fts)
    ├── gui/
    └── cui/
```

---

## Migration Path

### Phase 1: Create Alias

```python
# src/scitex/schema/__init__.py (new)
"""SciTeX Schema definitions for bundles."""
from scitex.fts import *  # Re-export everything

# src/scitex/fts/__init__.py (modified)
import warnings
warnings.warn(
    "scitex.fts is deprecated. Use scitex.schema instead.",
    DeprecationWarning,
    stacklevel=2
)
# ... existing code
```

**Effort:** Low (1 hour)

### Phase 2: Update Internal Imports

Update all internal usage:

```bash
# Find internal fts imports
grep -r "from scitex.fts" src/scitex/
grep -r "import scitex.fts" src/scitex/

# Replace with scitex.schema
```

**Effort:** Medium (2-3 hours)

### Phase 3: Documentation

- Update all docstrings
- Update README files
- Update examples
- Add migration note to changelog

**Effort:** Medium (2-3 hours)

### Phase 4: Remove fts (Major Version)

In v2.0:
- Remove `src/scitex/fts/` directory
- Remove deprecation warnings
- Update all imports

**Effort:** Low (1 hour)

---

## Naming Comparison Matrix

| Criterion | fts | schema | spec | repro | io.bundle |
|-----------|-----|--------|------|-------|-----------|
| Memorable | 2 | 5 | 4 | 3 | 4 |
| Clear purpose | 2 | 5 | 4 | 3 | 4 |
| Unique | 4 | 3 | 2 | 5 | 4 |
| Import UX | 2 | 5 | 4 | 3 | 4 |
| Migration ease | - | 5 | 5 | 5 | 3 |
| **Total** | **10** | **23** | **19** | **19** | **19** |

**Winner:** `scitex.schema` with score 23/25

---

## Impact Assessment

### Files to Modify

```
Internal imports of scitex.fts:
├── src/scitex/bridge/_figrecipe.py
├── src/scitex/fig/io/_bundle.py
├── src/scitex/plt/io/_bundle.py
├── src/scitex/io/_save.py
└── tests/scitex/fts/

Total: ~15 files
```

### API Changes

| Before | After |
|--------|-------|
| `from scitex.fts import FTS` | `from scitex.schema import Schema` |
| `from scitex.fts import Node` | `from scitex.schema import Node` |
| `scitex.fts.load()` | `scitex.schema.load()` |

### Backward Compatibility

Full compatibility via alias:
```python
# This will work (with deprecation warning)
from scitex.fts import FTS

# This is the new way
from scitex.schema import Schema
```

---

## Decision Checklist

- [ ] Approve renaming fts → schema
- [ ] Approve deprecation timeline
- [ ] Approve eventual merge into io.bundle (optional)
- [ ] Assign implementation

---

## Appendix: FTS Name Origin

FTS originally stood for:
- **F**igure
- **T**able
- **S**tatistics

The idea was a unified schema for all publication elements. However:
- Users don't know this acronym
- It's not intuitive
- "Schema" better describes what the module actually provides

<!-- EOF -->
