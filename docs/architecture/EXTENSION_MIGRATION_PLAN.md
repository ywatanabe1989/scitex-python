# Bundle Extension Migration Plan

**Status:** APPROVED - Hybrid .zip format
**Created:** 2026-01-07
**Updated:** 2026-01-07
**Author:** ywatanabe + Claude

---

## Decision

**APPROVED:** Hybrid extension format with `manifest.json`

| Type | Current | New |
|------|---------|-----|
| Plot | `.pltz` | `.plot.zip` |
| Figure | `.figz` | `.figure.zip` |
| Stats | `.statsz` | `.stats.zip` |

Manifest file: `manifest.json` inside bundle

---

## Summary

Migrate from custom extensions (`.pltz`, `.figz`, `.statsz`) to hybrid `.zip`
extensions (`.plot.zip`, `.figure.zip`, `.stats.zip`) for better OS compatibility.

---

## Problem Statement

### Current Extensions

| Type | Extension | Directory Form |
|------|-----------|----------------|
| Plot bundle | `.pltz` | `.pltz.d` |
| Figure bundle | `.figz` | `.figz.d` |
| Stats bundle | `.statsz` | `.statsz.d` |

### Issues

1. **OS Integration**: Custom extensions not recognized by OS file managers
2. **Usability**: Users can't double-click to unzip/explore contents
3. **Discoverability**: File type not obvious from extension
4. **Windows**: Poor integration with Explorer

---

## Proposed Solution

### Hybrid Extension Format

| Type | Current | Proposed | Directory Form |
|------|---------|----------|----------------|
| Plot bundle | `.pltz` | `.plot.zip` | `.plot/` |
| Figure bundle | `.figz` | `.figure.zip` | `.figure/` |
| Stats bundle | `.statsz` | `.stats.zip` | `.stats/` |

### Benefits

1. **Native unzip**: OS can extract with standard tools
2. **Clear type**: Extension indicates content type
3. **Backward compatible**: Old extensions still recognized
4. **Programmatic ID**: `__manifest__.json` inside confirms type

### Manifest Structure

```json
{
  "__scitex__": {
    "type": "plot",
    "version": "1.0.0",
    "schema": "scitex.bundle.plot",
    "created": "2026-01-07T12:00:00Z"
  }
}
```

---

## Impact Assessment

### Files Requiring Changes

```
Total files referencing extensions: 37

Breakdown by location:
├── src/scitex/io/bundle/     (5 files) - CORE: _types.py, _core.py, _zip.py, etc.
├── src/scitex/plt/           (6 files) - io/_bundle.py, gallery/, etc.
├── src/scitex/fig/           (8 files) - editor/, io/_bundle.py
├── src/scitex/fts/           (7 files) - _fig/, _bundle/
├── src/scitex/stats/         (2 files) - io/__init__.py, _bundle.py
├── src/scitex/cli/           (1 file)  - convert.py
├── docs/                     (8 files) - README files, architecture docs
```

### Core Change Location

All extension definitions centralized in:
```
src/scitex/io/bundle/_types.py
```

```python
# Current
EXTENSIONS: Tuple[str, ...] = (".figz", ".pltz", ".statsz")

# Proposed
EXTENSIONS_LEGACY = (".figz", ".pltz", ".statsz")
EXTENSIONS_NEW = (".figure.zip", ".plot.zip", ".stats.zip")
EXTENSIONS = EXTENSIONS_LEGACY + EXTENSIONS_NEW
```

---

## Migration Phases

### Phase 1: Add Support (Non-Breaking)

**Scope:** Support reading both old and new extensions

```python
# _types.py changes
EXTENSIONS_LEGACY = (".figz", ".pltz", ".statsz")
EXTENSIONS_NEW = (".figure.zip", ".plot.zip", ".stats.zip")
EXTENSIONS = EXTENSIONS_LEGACY + EXTENSIONS_NEW

EXTENSION_MAP = {
    ".pltz": ".plot.zip",
    ".figz": ".figure.zip",
    ".statsz": ".stats.zip",
}

# Directory forms
DIR_EXTENSIONS_LEGACY = (".pltz.d", ".figz.d", ".statsz.d")
DIR_EXTENSIONS_NEW = (".plot", ".figure", ".stats")
```

**Files to modify:**
- `src/scitex/io/bundle/_types.py`
- `src/scitex/io/bundle/_core.py` (detection logic)

**Effort:** Low (1-2 hours)

### Phase 2: Default to New Format

**Scope:** Create new bundles with `.plot.zip` format by default

```python
# _types.py
DEFAULT_EXTENSION_FORMAT = "new"  # or "legacy"

def get_extension(bundle_type: str) -> str:
    if DEFAULT_EXTENSION_FORMAT == "new":
        return {
            "pltz": ".plot.zip",
            "figz": ".figure.zip",
            "statsz": ".stats.zip",
        }[bundle_type]
    else:
        return f".{bundle_type}"
```

**Files to modify:**
- `src/scitex/io/bundle/_types.py`
- `src/scitex/io/bundle/_core.py`
- `src/scitex/plt/io/_bundle.py`
- `src/scitex/fig/io/_bundle.py`
- `src/scitex/stats/io/_bundle.py`

**Effort:** Medium (4-6 hours)

### Phase 3: Add Deprecation Warnings

**Scope:** Warn when reading old format, suggest migration

```python
import warnings

def load_bundle(path):
    if has_legacy_extension(path):
        warnings.warn(
            f"Legacy extension detected: {path.suffix}. "
            f"Consider migrating to {EXTENSION_MAP[path.suffix]}. "
            "Legacy extensions will be removed in v2.0.",
            DeprecationWarning
        )
    # ... load logic
```

**Files to modify:**
- `src/scitex/io/bundle/_core.py`

**Effort:** Low (1 hour)

### Phase 4: Migration Tool

**Scope:** CLI tool to convert old bundles to new format

```bash
# Usage
scitex migrate-bundles ./data/ --dry-run
scitex migrate-bundles ./data/ --execute

# What it does
# 1. Find all .pltz, .figz, .statsz files
# 2. Rename to .plot.zip, .figure.zip, .stats.zip
# 3. Add __manifest__.json if missing
# 4. Verify integrity
```

**Files to create:**
- `src/scitex/cli/migrate.py`

**Effort:** Medium (3-4 hours)

### Phase 5: Remove Legacy Support (Major Version)

**Scope:** In v2.0, remove legacy extension support

```python
# _types.py (v2.0)
EXTENSIONS = (".figure.zip", ".plot.zip", ".stats.zip")
DIR_EXTENSIONS = (".figure", ".plot", ".stats")
# No more legacy aliases
```

**Effort:** Low (documentation update)

---

## Timeline

| Phase | Description | When | Breaking? |
|-------|-------------|------|-----------|
| 1 | Add support for new extensions | v1.x.0 | No |
| 2 | Default to new format | v1.x+1.0 | No |
| 3 | Deprecation warnings | v1.x+2.0 | No |
| 4 | Migration tool | v1.x+2.0 | No |
| 5 | Remove legacy | v2.0.0 | Yes |

---

## Testing Strategy

### Unit Tests

```python
# test_extension_compat.py

def test_load_legacy_pltz():
    """Can load .pltz files"""
    bundle = load("test.pltz")
    assert bundle is not None

def test_load_new_plot_zip():
    """Can load .plot.zip files"""
    bundle = load("test.plot.zip")
    assert bundle is not None

def test_save_creates_new_format():
    """Save creates .plot.zip by default"""
    save(bundle, "output.plot.zip")
    assert Path("output.plot.zip").exists()

def test_legacy_deprecation_warning():
    """Loading legacy format shows warning"""
    with pytest.warns(DeprecationWarning):
        load("test.pltz")
```

### Integration Tests

```python
def test_roundtrip_legacy_to_new():
    """Load legacy, save new, verify identical content"""
    bundle = load("legacy.pltz")
    save(bundle, "new.plot.zip")
    bundle2 = load("new.plot.zip")
    assert bundle.data.equals(bundle2.data)
```

---

## Rollback Plan

If issues arise:

1. **Phase 1-3**: Simply revert `_types.py` changes
2. **Phase 4**: Migration tool is additive, no rollback needed
3. **Phase 5**: This is a major version, no rollback (by design)

---

## Documentation Updates

After migration:

- [ ] Update `README.md` files to use new extensions
- [ ] Update examples in `examples/`
- [ ] Update API documentation
- [ ] Add migration guide to docs
- [ ] Update CI/CD test fixtures

---

## Decision Checklist

- [ ] Approve hybrid extension strategy (`.plot.zip`)
- [ ] Approve phased migration approach
- [ ] Confirm v2.0 as legacy removal target
- [ ] Assign implementation to sprint/milestone

---

## Appendix: Current Extension Usage

```bash
# Files referencing .pltz/.figz/.statsz
$ grep -r "\.pltz\|\.figz\|\.statsz" src/scitex --include="*.py" | wc -l
37

# Breakdown by module
io/bundle: 5
plt: 6
fig: 8
fts: 7
stats: 2
cli: 1
docs: 8
```

<!-- EOF -->
