# SciTeX Family

SciTeX integrates standalone packages that can be used independently or through the unified `scitex` interface.

## Packages

| Package | scitex Module | Description | Status |
|---------|---------------|-------------|--------|
| [figrecipe](https://github.com/ywatanabe1989/figrecipe) | `scitex.plt` | Publication-ready matplotlib figures | Integrated |
| [crossref-local](https://github.com/ywatanabe1989/crossref-local) | `scitex.scholar.crossref` | Local CrossRef database (167M+ papers) | Integrated |
| [socialia](https://github.com/ywatanabe1989/socialia) | `scitex.social` | Social media posting (Twitter, LinkedIn) | Integrated |
| [scitex-writer](https://github.com/ywatanabe1989/scitex-writer) | `scitex.writer` | LaTeX manuscript compilation | Integrated |

## Architecture

```
scitex (umbrella)
├── scitex.plt      ← figrecipe
├── scitex.social   ← socialia
├── scitex.scholar  ← crossref-local
└── scitex.writer   ← scitex-writer
```

### Integration Pattern

- **Thin Wrapper**: Delegate without modification
- **Single Source of Truth**: Downstream packages are authoritative
- **Branding**: Apply `scitex.*` namespace via environment variables

### Usage

```python
# Via scitex (recommended)
import scitex as stx
fig, ax = stx.plt.subplots()

# Standalone (also works)
import figrecipe as fr
fig, ax = fr.subplots()
```

### CLI

```bash
# Via scitex
scitex writer compile manuscript
scitex social post "Hello"

# Standalone
scitex-writer compile manuscript
socialia post "Hello"
```

## Port Scheme

SciTeX uses `3129X` (sa-i-te-ku-su = 3-1-2-9):

| Port | Service |
|------|---------|
| 31290 | scitex-cloud |
| 31291 | crossref-local |
| 31292 | openalex |
| 31293 | scitex-audio |
