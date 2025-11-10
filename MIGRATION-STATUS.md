# SciTeX Module Migration Status

**Last Updated:** 2025-11-10
**Strategy:** Extract high-value modules as standalone packages following sklearn/pytorch pattern

## Overview

```
scitex-code (monorepo)  →  Separate standalone packages
```

## Migration Status

### ✅ Completed Migrations

| Module | Package Name | Status | Location | Notes |
|--------|-------------|--------|----------|-------|
| `logging/` | `scitex-core` | ✅ Extracted | `~/proj/scitex-core` | Core infrastructure |
| `errors.py` | `scitex-core` | ✅ Extracted | `~/proj/scitex-core` | Error classes |
| `sh/` | `scitex-core` | ✅ Extracted | `~/proj/scitex-core` | Shell utilities |
| `io/` | `scitex-io` | ✅ Extracted | `~/proj/scitex-io` | Data I/O (62 files) |
| `db/` | `scitex-db` | ✅ Extracted | `~/proj/scitex-db` | Database (57 files, SQLite3 + PostgreSQL) |

### 🔄 In Progress

| Module | Package Name | Status | Priority | Notes |
|--------|-------------|--------|----------|-------|
| `writer/` | `scitex-writer` | 🔄 Updating | HIGH | Needs scitex-core dependency |
| `scholar/` | `scitex-scholar` | 🔄 Planning | HIGH | 2313 files, 276MB |

### ⏳ Planned Extractions

| Module | Package Name | Priority | Reason | Dependencies |
|--------|-------------|----------|--------|--------------|
| `browser/` | Merge into `scitex-scholar` | HIGH | Used for paper access | scholar |
| `ml/` | `scitex-ml` | MEDIUM | ML utilities | scitex-core |
| `plt/` | `scitex-viz` | MEDIUM | Visualization | scitex-core |
| `web/` | `scitex-web` | LOW | Web utilities | scitex-core |

### 🏠 Staying in Monorepo (scitex-code)

These are lightweight utilities that stay in the main package:

| Module | Size | Reason |
|--------|------|--------|
| `str/` | Small | Simple string utilities |
| `path/` | Small | Path manipulation |
| `dict/` | Small | Dictionary utilities |
| `dt/` | Small | Datetime utilities |
| `gen/` | Small | General utilities |
| `utils/` | Small | Misc utilities |
| `types/` | Small | Type definitions |
| `decorators/` | Small | Python decorators |
| `context/` | Small | Context managers |
| `dev/` | Small | Development tools |
| `gists/` | Small | Code snippets |
| `units/` | Small | Unit conversions |
| `session/` | Medium | Session management |
| `repro/` | Medium | Reproducibility tools |
| `rng/` | Small | Random number generation |
| `template/` | Medium | Project templates |
| `capture/` | Small | Screen capture |
| `linalg/` | Small | Linear algebra |
| `parallel/` | Small | Parallel computing |
| `stats/` | Medium | Statistics |
| `dsp/` | Medium | Digital signal processing |
| `nn/` | Medium | Neural networks |
| `torch/` | Medium | PyTorch utilities |
| `ai/` | Medium | AI utilities |
| `pd/` | Medium | Pandas utilities |
| ~~`db/`~~ | ~~Medium~~ | ~~Extracted to scitex-db~~ |
| `git/` | Medium | Git operations |
| `tex/` | Small | LaTeX utilities |
| `resource/` | Small | Resource management |
| `security/` | Small | Security utilities |
| `cli/` | Small | CLI tools |
| `project/` | Small | Project management |
| `os/` | Small | OS utilities |
| `benchmark/` | Small | Benchmarking |

## Package Dependency Graph

```
scitex-core v0.1.0 (foundation)
    ↓
├── scitex-db v0.1.0 (depends on scitex-core)
├── scitex-io v0.1.0 (standalone with inlined utils)
├── scitex-writer (needs update to use scitex-core)
├── scitex-scholar (planned)
└── scitex (main package with utilities)
```

## Inlined Dependencies

To maintain standalone packages without circular dependencies, we've inlined:

### scitex-io inlined utilities:
- `DotDict` (from `scitex.dict`)
- `clean_path` (from `scitex.str`)
- `color_text` (from `scitex.str`)
- `readable_bytes` (from `scitex.str`)
- Path utilities (from `scitex.path`)
- String parsing (from `scitex.str`)

### scitex-core inlined utilities:
- `clean_path` (from `scitex.str`)
- `printc` (from `scitex.str`)

### scitex-db approach:
- Depends on `scitex-core>=0.1.0` for errors module
- Has minimal `_utils.py` with `printc` for colored output

## Migration Checklist

For each module extraction:

- [ ] Create package directory structure
- [ ] Copy module files
- [ ] Create `pyproject.toml` with dependencies
- [ ] Create `README.md`
- [ ] Fix imports (change `from scitex.x` → local imports)
- [ ] Inline small utilities or add to scitex-core
- [ ] Initialize git repository
- [ ] Test package installation
- [ ] Update this tracking document
- [ ] Commit changes

## Package Versions

| Package | Version | Git Commits | PyPI Published | Notes |
|---------|---------|-------------|----------------|-------|
| scitex-core | 0.1.0 | 3 | ❌ Not yet | logging, errors, sh modules |
| scitex-io | 0.1.0 | 3 | ❌ Not yet | 62 files, standalone |
| scitex-db | 0.1.0 | 2 | ❌ Not yet | 57 files, depends on scitex-core |
| scitex-writer | 2.0.0a0 | - | ✅ Published | Needs update for scitex-core |
| scitex-scholar | - | - | ❌ Not created | 2313 files, 276MB planned |

## Next Steps

1. ✅ **scitex-io**: Fix remaining imports, test, commit
2. 🔄 **scitex-scholar**: Extract (large module, 2313 files)
3. 🔄 **scitex-writer**: Update to depend on scitex-core
4. 📝 **Documentation**: Update each package's README
5. 🚀 **PyPI**: Publish packages

## Notes

- **Philosophy**: Extract high-value, discoverable modules; keep utilities in monorepo
- **Dependencies**: Prefer inlining small utilities over complex dependency chains
- **Testing**: Each package must work standalone
- **Documentation**: Each package needs clear README with examples
