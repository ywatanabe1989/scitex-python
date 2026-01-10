<!-- ---
!-- Timestamp: 2025-12-19 04:45:00
!-- Author: ywatanabe (Claude Opus 4.5)
!-- File: /home/ywatanabe/proj/scitex-code/docs/STX_MIGRATION_PLAN.md
!-- --- -->

# Architecture Plan: Migration to Unified .stx Format

## Goal

Migrate from separate `.figz`, `.pltz`, `.statsz` extensions to a unified `.stx` format with type discrimination via `spec.json`, enabling self-recursive bundling while maintaining backward compatibility.

## Architecture Decisions (2025-12-19)

**Approved by user review:**

1. **Schema Approach**: Use `scitex.bundle` + `type` field
   ```json
   {"schema": {"name": "scitex.bundle", "version": "2.0.0"}, "type": "figure"}
   ```

2. **Module Location**: Enhance existing `scitex/io/bundle/` (minimal disruption)

3. **Self-Recursion**: Include in Phase 1 (build recursion support from the start)

---

## Current State (v1.0.0)

### Extension-Based Type System

```
Current Bundle Types:
.figz    → scitex.fig.figure v1.0.0     (Multi-panel compositions)
.pltz    → scitex.plt.plot v1.0.0       (Single plot with data)
.statsz  → scitex.stats.stats v1.0.0    (Statistical results)
```

### Current Architecture

```
src/scitex/
├── io/bundle/                  # Core bundle system
│   ├── _core.py               # load, save, pack, unpack, validate
│   ├── _zip.py                # ZipBundle class
│   ├── _types.py              # BundleType, EXTENSIONS, errors
│   └── _nested.py             # Nested bundle operations
├── fig/_bundle.py             # Figz class
├── plt/_bundle.py             # Pltz class
└── stats/_bundle.py           # Statsz class
```

### Current Type Detection

```python
# Extension-based (src/scitex/io/bundle/_types.py)
EXTENSIONS = (".figz", ".pltz", ".statsz")

def get_type(path: Path) -> Optional[str]:
    if p.suffix in EXTENSIONS:
        return p.suffix[1:]  # "figz", "pltz", "statsz"
```

---

## Target State (v2.0.0)

### Unified Extension with Type Field

```
New Bundle Format:
.stx → All bundle types
       Type determined by spec.json["schema"]["name"]
```

### Proposed Architecture

```
src/scitex/
├── bundle/                     # NEW: Unified bundle system
│   ├── __init__.py            # Public API: load, save, create
│   ├── _core.py               # Core operations (from io.bundle)
│   ├── _zip.py                # ZipBundle (from io.bundle)
│   ├── _types.py              # BundleSpec, StxType enums
│   ├── _validation.py         # Schema validation
│   ├── _nested.py             # Self-recursive operations
│   └── _safety.py             # UUID tracking, depth limits
│
├── types/                      # NEW: Type-specific implementations
│   ├── __init__.py
│   ├── figure/                # Figure type (was fig._bundle)
│   │   ├── __init__.py        # Figz class
│   │   ├── _spec.py           # Schema definition
│   │   ├── _validate.py       # Figure-specific validation
│   │   └── _render.py         # Rendering logic
│   ├── plot/                  # Plot type (was plt._bundle)
│   │   ├── __init__.py        # Pltz class
│   │   ├── _spec.py           # Schema definition
│   │   ├── _validate.py       # Plot-specific validation
│   │   └── _render.py         # Rendering logic
│   └── stats/                 # Stats type (was stats._bundle)
│       ├── __init__.py        # Statsz class
│       ├── _spec.py           # Schema definition
│       └── _validate.py       # Stats-specific validation
│
├── io/bundle/                  # DEPRECATED: Kept for compatibility
│   └── (symlinks or compatibility shims)
│
├── fig/                        # Existing module
│   ├── __init__.py            # Re-exports from scitex.types.figure
│   └── _bundle.py             # DEPRECATED (or symlink)
│
├── plt/                        # Existing module
│   ├── __init__.py            # Re-exports from scitex.types.plot
│   └── _bundle.py             # DEPRECATED (or symlink)
│
└── stats/                      # Existing module
    ├── __init__.py            # Re-exports from scitex.types.stats
    └── _bundle.py             # DEPRECATED (or symlink)
```

---

## Schema Design (v2.0.0)

### Type Discrimination via spec.json

```json
{
  "schema": {
    "name": "scitex.figure",      // Type identifier
    "version": "2.0.0"
  },
  "bundle": {
    "id": "550e8400-e29b-41d4-a716-446655440000",  // UUID
    "created": "2025-12-19T04:45:00Z",
    "modified": "2025-12-19T04:45:00Z"
  },
  "safety": {
    "max_depth": 5,                // Recursion limit
    "allow_children": true         // Can contain other .stx?
  },
  ...type-specific fields...
}
```

### Supported Types

| schema.name      | Description                | Allow Children | Equivalent Old Format |
|------------------|----------------------------|----------------|-----------------------|
| scitex.figure    | Multi-panel composition    | Yes            | .figz                 |
| scitex.plot      | Single plot with data      | No             | .pltz                 |
| scitex.stats     | Statistical results        | No             | .statsz               |
| scitex.dataset   | Data container (future)    | No             | -                     |
| scitex.notebook  | Computational doc (future) | Yes            | -                     |

---

## Migration Strategy

### Phase 1: Core Infrastructure (Week 1-2)

**Objective**: Establish unified bundle system without breaking existing code.

**Tasks**:
1. Create `src/scitex/bundle/` module
   - Copy `_core.py`, `_zip.py` from `io/bundle/`
   - Update `get_type()` to support both `.stx` and legacy extensions
   - Add `BundleSpec` dataclass for type-safe spec handling

2. Implement dual-format support
   ```python
   def get_type(path: Path) -> Optional[str]:
       # Check extension
       if path.suffix == ".stx":
           # Read spec.json to determine type
           spec = read_spec(path)
           return map_schema_to_type(spec["schema"]["name"])
       elif path.suffix in LEGACY_EXTENSIONS:
           return path.suffix[1:]  # figz, pltz, statsz
       return None
   ```

3. Add UUID and safety fields
   - `bundle.id` - UUID for tracking
   - `safety.max_depth` - Recursion protection
   - `safety.allow_children` - Type constraint

**Acceptance Criteria**:
- Existing `.figz/.pltz/.statsz` files load without modification
- New `.stx` files can be created and loaded
- `scitex.bundle.load()` works for all formats
- Tests pass for legacy formats

**Files Created**:
```
src/scitex/bundle/
├── __init__.py          # Public API
├── _core.py            # load, save, pack, unpack
├── _zip.py             # ZipBundle
├── _types.py           # StxType, BundleSpec
├── _validation.py      # Schema validators
└── _safety.py          # UUID tracking, depth checks
```

---

### Phase 2: Figure + Plot Types (Week 3-4)

**Objective**: Migrate Figz and Pltz to use `.stx` format internally while maintaining API compatibility.

**Tasks**:
1. Create `src/scitex/types/figure/`
   - Move `Figz` class logic from `src/scitex/fig/_bundle.py`
   - Update to use `.stx` extension
   - Add self-recursive support (figures can contain figures)
   - Schema: `scitex.figure v2.0.0`

2. Create `src/scitex/types/plot/`
   - Move `Pltz` class logic from `src/scitex/plt/_bundle.py`
   - Update to use `.stx` extension
   - Schema: `scitex.plot v2.0.0`

3. Implement compatibility layer in old locations
   ```python
   # src/scitex/fig/_bundle.py
   from scitex.types.figure import Figz
   __all__ = ["Figz"]  # Re-export
   ```

4. Update save behavior
   ```python
   # New API (preferred)
   figz.save("output.stx")  # Saves as .stx

   # Legacy API (still works)
   figz.save("output.figz")  # Warns, saves as .stx internally
   ```

**Acceptance Criteria**:
- `stx.fig.Figz.create()` produces `.stx` files
- Existing code using `from scitex.fig import Figz` works unchanged
- Legacy `.figz` files can be opened and auto-migrated
- Self-recursive figures work (fig contains fig)
- Depth limits enforced (max_depth=5)

**Files Modified**:
```
src/scitex/types/figure/
├── __init__.py          # Figz class
├── _spec.py            # FigureSpec schema
├── _validate.py        # Figure validation
└── _render.py          # Composition rendering

src/scitex/types/plot/
├── __init__.py          # Pltz class
├── _spec.py            # PlotSpec schema
├── _validate.py        # Plot validation
└── _render.py          # Plot rendering
```

---

### Phase 3: Stats + Cleanup (Week 5)

**Objective**: Complete migration and deprecate old paths.

**Tasks**:
1. Create `src/scitex/types/stats/`
   - Move `Statsz` class
   - Schema: `scitex.stats v2.0.0`

2. Deprecate old modules
   ```python
   # src/scitex/io/bundle/__init__.py
   import warnings
   warnings.warn(
       "scitex.io.bundle is deprecated. Use scitex.bundle instead.",
       DeprecationWarning,
       stacklevel=2
   )
   from scitex.bundle import *  # Re-export
   ```

3. Update documentation
   - Migration guide for users
   - API reference for new paths
   - Examples using `.stx` format

4. Add conversion utility
   ```bash
   $ scitex convert old_figure.figz new_figure.stx
   $ scitex convert --batch ./figures/*.figz ./output/
   ```

**Acceptance Criteria**:
- All bundle types use `.stx` format
- Deprecation warnings for old imports
- CLI conversion tool works
- Documentation updated
- All tests pass

**Files Created**:
```
src/scitex/types/stats/
├── __init__.py          # Statsz class
├── _spec.py            # StatsSpec schema
└── _validate.py        # Stats validation

src/scitex/cli/convert.py  # CLI conversion tool
```

---

## Backward Compatibility Strategy

### Read All, Write .stx

```python
# Read: Support all formats
bundle = stx.bundle.load("figure.figz")   # Works
bundle = stx.bundle.load("plot.pltz")     # Works
bundle = stx.bundle.load("stats.statsz")  # Works
bundle = stx.bundle.load("new.stx")       # Works

# Write: Always prefer .stx
figz.save("output.figz")  # Warning → saves as output.stx
figz.save("output.stx")   # Preferred, no warning
```

### Auto-Migration on Load

```python
def load(path: Path) -> Bundle:
    """Load bundle, auto-migrating legacy formats."""
    bundle_type = get_type(path)
    bundle_data = _read_bundle(path)

    # Check schema version
    version = bundle_data["spec"]["schema"]["version"]
    if version.startswith("1."):
        # Auto-migrate to v2.0.0
        bundle_data = migrate_v1_to_v2(bundle_data)
        warnings.warn(f"Migrated {path} from v{version} to v2.0.0")

    return Bundle(bundle_data)
```

### Deprecation Timeline

| Version | Action                                    | Timeline |
|---------|-------------------------------------------|----------|
| 2.0.0   | Introduce `.stx`, keep old extensions     | Month 1  |
| 2.1.0   | Deprecation warnings for old imports      | Month 3  |
| 2.2.0   | Legacy extensions read-only               | Month 6  |
| 3.0.0   | Remove support for `.figz/.pltz/.statsz`  | Month 12 |

---

## Safety Constraints

### UUID Tracking

```python
# Each bundle gets unique ID
bundle_id = str(uuid.uuid4())

# Track parent-child relationships
parent_bundle.spec["bundle"]["children"] = [
    {"id": child_id, "path": "panel_A.stx"}
]
```

### Depth Limits

```python
def validate_depth(bundle_path: Path, max_depth: int = 5) -> None:
    """Prevent infinite recursion."""
    depth = 0
    current = bundle_path
    while has_parent(current):
        depth += 1
        if depth > max_depth:
            raise BundleValidationError(
                f"Bundle nesting exceeds max_depth={max_depth}"
            )
        current = get_parent(current)
```

### Circular Reference Detection

```python
def detect_circular_refs(bundle: Bundle) -> List[str]:
    """Detect circular bundle references by UUID."""
    visited = set()
    stack = [bundle.spec["bundle"]["id"]]

    while stack:
        current_id = stack.pop()
        if current_id in visited:
            return f"Circular reference detected: {current_id}"
        visited.add(current_id)

        # Add children to stack
        children = get_child_bundles(current_id)
        stack.extend(children)

    return []  # No cycles
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `src/scitex/bundle/` module structure
- [ ] Implement dual-format type detection
- [ ] Add `BundleSpec` dataclass with UUID, safety fields
- [ ] Write tests for legacy format loading
- [ ] Add UUID generation and validation
- [ ] Implement depth limit checks

### Phase 2: Figure + Plot Types
- [ ] Create `src/scitex/types/figure/` module
- [ ] Migrate `Figz` class to use `.stx`
- [ ] Implement self-recursive figure support
- [ ] Create `src/scitex/types/plot/` module
- [ ] Migrate `Pltz` class to use `.stx`
- [ ] Add compatibility shims in old locations
- [ ] Update tests for new paths
- [ ] Test self-recursive figures

### Phase 3: Stats + Cleanup
- [ ] Create `src/scitex/types/stats/` module
- [ ] Migrate `Statsz` class to use `.stx`
- [ ] Add deprecation warnings to old modules
- [ ] Create CLI conversion tool
- [ ] Write migration guide documentation
- [ ] Update all examples to use `.stx`
- [ ] Run full test suite
- [ ] Update README and API docs

---

## Example Usage (v2.0.0)

### Creating Bundles

```python
import scitex as stx

# Figure (was .figz, now .stx)
fig = stx.fig.Figz.create("figure.stx", "Figure1")
fig.add_panel("A", pltz_bytes, position={"x_mm": 10, "y_mm": 10})
fig.save()

# Plot (was .pltz, now .stx)
pltz = stx.plt.Pltz.create("plot.stx", plot_type="line", data=df)
pltz.save()

# Stats (was .statsz, now .stx)
stats = stx.stats.Statsz.create("results.stx", comparisons=[...])
stats.save()
```

### Self-Recursive Figures

```python
# Create sub-figure
sub_fig = stx.fig.Figz.create("sub_figure.stx", "SubFigure")
sub_fig.add_panel("A", plot_a)
sub_fig.add_panel("B", plot_b)
sub_fig.save()

# Embed sub-figure in main figure
main_fig = stx.fig.Figz.create("main_figure.stx", "MainFigure")
main_fig.add_panel("Panel1", sub_fig_bytes, position={"x_mm": 10, "y_mm": 10})
main_fig.save()

# Depth limit enforced
main_fig.spec["safety"]["max_depth"]  # 5 (default)
```

### Loading Legacy Formats

```python
# Auto-detects type and migrates
bundle = stx.bundle.load("old_figure.figz")
bundle.spec["schema"]["version"]  # "2.0.0" (migrated)

# Save as .stx
bundle.save("new_figure.stx")
```

---

## Migration Path for Users

### Gradual Migration (Recommended)

```python
# Step 1: Update scitex to v2.0.0
pip install --upgrade scitex

# Step 2: Existing code works unchanged
from scitex.fig import Figz  # Still works (compatibility layer)
fig = Figz("old_figure.figz")  # Loads fine

# Step 3: Gradually move to .stx
fig.save("new_figure.stx")  # Preferred
```

### Batch Conversion

```bash
# Convert all bundles in directory
scitex convert --batch ./figures/*.figz ./output/
scitex convert --batch ./plots/*.pltz ./output/
scitex convert --batch ./stats/*.statsz ./output/

# Verify conversion
scitex validate ./output/*.stx
```

---

## Risk Mitigation

### Risks

1. **Breaking existing code**: Users have `.figz/.pltz/.statsz` files
2. **Performance**: Type detection requires reading spec.json
3. **Adoption**: Users may resist new format

### Mitigation

1. **Backward compatibility**: Full read support for legacy formats
2. **Caching**: Cache type info in memory during session
3. **Gradual deprecation**: 12-month timeline with warnings
4. **Clear benefits**: Self-recursive bundles, simpler API
5. **Tooling**: CLI conversion tool for easy migration

---

## Success Criteria

### Technical
- [ ] All existing tests pass with v2.0.0
- [ ] Legacy `.figz/.pltz/.statsz` files load correctly
- [ ] New `.stx` files can be created and loaded
- [ ] Self-recursive figures work (depth ≤ 5)
- [ ] UUID tracking prevents circular references
- [ ] Performance: No significant slowdown (<5%)

### User Experience
- [ ] Existing code works without modification
- [ ] Clear migration guide available
- [ ] CLI conversion tool provided
- [ ] Deprecation warnings are helpful
- [ ] Documentation updated with examples

### Quality
- [ ] Test coverage ≥ 90%
- [ ] No breaking changes in public API
- [ ] Type hints for all new code
- [ ] Docstrings for all public functions

---

## Timeline Summary

| Week  | Phase                      | Deliverable                          |
|-------|----------------------------|--------------------------------------|
| 1-2   | Phase 1: Core              | `scitex.bundle` module, dual-format  |
| 3-4   | Phase 2: Figure/Plot       | `.stx` for Figz/Pltz, self-recursive |
| 5     | Phase 3: Stats + Cleanup   | Statsz migration, CLI tool, docs     |
| 6     | Testing + Documentation    | Full test coverage, user guides      |

---

## Future Enhancements (Post-v2.0.0)

1. **New Bundle Types**
   - `scitex.dataset` - Data containers
   - `scitex.notebook` - Computational documents
   - `scitex.paper` - Full manuscript bundles

2. **Advanced Features**
   - Bundle diff/merge (version control)
   - Lazy loading for large bundles
   - Compression options (zip vs tar.gz)

3. **Ecosystem Integration**
   - Git LFS support for large bundles
   - Cloud storage adapters (S3, GCS)
   - Collaboration features (shared bundles)

---

## References

- Current bundle system: `src/scitex/io/bundle/`
- Bundle classes: `src/scitex/{fig,plt,stats}/_bundle.py`
- Architecture docs: `docs/CANVAS_OBJECTS_PLAN_v01.md`

---

## Contact

For questions or feedback on this migration plan:
- File issue: https://github.com/ywatanabe/scitex-code/issues
- Discussion: https://github.com/ywatanabe/scitex-code/discussions

<!-- EOF -->
