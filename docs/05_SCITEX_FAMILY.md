<!-- ---
!-- Timestamp: 2026-01-27 18:23:50
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-python/docs/05_SCITEX_FAMILY.md
!-- --- -->

# SciTeX Family

SciTeX integrates standalone packages that can be used independently or through the unified `scitex` interface.

## Packages

| Package | scitex Module | Integration | API Items |
|---------|---------------|-------------|-----------|
| [figrecipe](https://github.com/ywatanabe1989/figrecipe) | `scitex.plt` | Enhanced | 14 → 3881 |
| [crossref-local](https://github.com/ywatanabe1989/crossref-local) | `scitex.scholar` | Enhanced | 98 → 2063 |
| [socialia](https://github.com/ywatanabe1989/socialia) | `scitex.social` | Thin | 2907 → 12 |
| [scitex-writer](https://github.com/ywatanabe1989/scitex-writer) | `scitex.writer` | Thin | 37 → 37 |

## Architecture

```
scitex (umbrella)
├── scitex.plt      ← figrecipe + matplotlib + local features
├── scitex.scholar  ← crossref-local + local features
├── scitex.social   ← socialia (thin wrapper)
└── scitex.writer   ← scitex-writer (thin wrapper)
```

## Integration Patterns

### Thin Wrapper (scitex.writer, scitex.social)

Re-export downstream package as-is, without modifications.

**Python API:**
```python
# scitex/writer/__init__.py
from scitex_writer import bib, compile, figures, guidelines, project, prompts, tables

# scitex/social/__init__.py
from socialia import Twitter, LinkedIn, Reddit, YouTube, GoogleAnalytics, BasePoster
```

**MCP Tools:**
```python
# scitex/_mcp_tools/writer.py
def register_writer_tools(mcp):
    from scitex_writer._mcp.tools import register_all_tools
    register_all_tools(mcp)  # Delegate to downstream

# scitex/_mcp_tools/social.py
def register_social_tools(mcp):
    from socialia._mcp.tools import register_all_tools
    register_all_tools(mcp)  # Delegate to downstream
```

- Single source of truth: downstream package
- API parity: `scitex.writer` ≈ `scitex_writer`, `scitex.social` ≈ `socialia`
- MCP tools delegated to downstream's `register_all_tools(mcp)`
- Use `scitex introspect api` to verify consistency

### Enhanced Wrapper (scitex.plt, scitex.scholar)

Integrate downstream package with scitex-specific features.

```python
# scitex/plt/__init__.py
from figrecipe import compose, crop, save, subplots  # Core from figrecipe
from . import ax, color, gallery, styles             # Local scitex features
```

- Downstream as foundation
- Local features added (color palettes, gallery, etc.)
- API larger than standalone package

## CLI

```bash
# Via scitex (hyphen or underscore both work)
scitex introspect api scitex-writer  # → 37 items
scitex introspect api scitex.writer  # → 37 items

# Standalone
scitex-writer --help
socialia --help
```

## Branding

Set environment variables before importing to apply scitex namespace:

```python
os.environ.setdefault("SCITEX_WRITER_BRAND", "scitex.writer")
os.environ.setdefault("FIGRECIPE_BRAND", "scitex.plt")
```

## Port Scheme

SciTeX uses `3129X` (sa-i-te-ku-su = 3-1-2-9):

| Port | Service |
|------|---------|
| 31290 | scitex-cloud |
| 31291 | crossref-local |
| 31292 | openalex |
| 31293 | scitex-audio |

<!-- EOF -->
