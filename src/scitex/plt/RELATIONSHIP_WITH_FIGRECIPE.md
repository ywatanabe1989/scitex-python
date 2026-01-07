<!-- ---
!-- Timestamp: 2026-01-06 23:30:00
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/RELATIONSHIP_WITH_FIGRECIPE.md
!-- --- -->

# Relationship: figrecipe and scitex

## Executive Summary

**figrecipe** and **scitex** serve different but complementary roles:

| Library | Role | Target User |
|---------|------|-------------|
| figrecipe | Simple entry point with GUI editor | New users, quick plots |
| scitex | Full Research OS platform | Power users, publication workflows |

Both are maintained by the same developer. The current architecture is reasonable.

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        scitex.ai (Research OS)                      │
│                                                                     │
│  ┌───────────────────┐         ┌─────────────────────────────────┐  │
│  │    figrecipe      │         │           scitex                │  │
│  │  (Entry Point)    │   →     │      (Full Platform)            │  │
│  │                   │         │                                 │  │
│  │  • Simple API     │         │  • .pltz/.figz/.statsz bundles  │  │
│  │  • GUI editor     │         │  • 50+ CSV formatters           │  │
│  │  • Recipe YAML    │         │  • Publication quality          │  │
│  │  • Reproduction   │         │  • Cloud integration            │  │
│  │                   │         │  • Statistical analysis         │  │
│  └───────────────────┘         └─────────────────────────────────┘  │
│                                                                     │
│  pip install figrecipe         pip install scitex[plt]              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How They Relate

### figrecipe (External Library)

**Purpose:** Simple, focused tool for reproducible matplotlib figures.

```python
import figrecipe as fr

# Create figure with recording
fig, ax = fr.subplots()
ax.plot([1, 2, 3], [4, 5, 6])

# Save recipe
fr.save(fig, 'plot.yaml')

# Later: reproduce exactly
fig, ax = fr.reproduce('plot.yaml')
```

**Features:**
- Recipe recording (YAML format)
- GUI editor for visual adjustments
- Reproduction from recipe
- Standalone, minimal dependencies

**Target:** Users who want simple reproducible plots without the full scitex stack.

### scitex.plt (Part of scitex)

**Purpose:** Publication-quality plotting with full data traceability.

```python
import scitex.plt as splt
import scitex.io as sio

# Create figure with tracking
fig, ax = splt.subplots()
ax.plot([1, 2, 3], [4, 5, 6], id='data')

# Save as atomic bundle
sio.save(fig, 'plot.pltz')
# Creates: spec.json, style.json, data.csv, exports/
```

**Features:**
- MM-based sizing for publications
- SigmaPlot-compatible CSV export (50+ formatters)
- Atomic bundles (.pltz/.figz/.statsz)
- Style system (editable style.json)
- Integration with scitex-cloud

**Target:** Researchers needing full publication workflow.

---

## Bundle Formats

### scitex Bundle (.pltz)

```
plot.pltz.d/
├── spec.json          # Plot specification
├── style.json         # Visual styling (EDITABLE)
├── data.csv           # Source data (SigmaPlot format)
├── exports/
│   ├── plot.png       # Rendered image
│   └── plot.svg       # Vector format
└── cache/
    └── geometry_px.json
```

### figrecipe Recipe (.yaml)

```yaml
# plot.yaml
figure:
  figsize: [6, 4]
  dpi: 100
calls:
  - method: plot
    args: [[1, 2, 3], [4, 5, 6]]
    kwargs: {}
data:
  - file: plot_data.csv
```

### Comparison

| Aspect | figrecipe | scitex |
|--------|-----------|--------|
| Format | YAML + CSV | ZIP bundle (.pltz) |
| Spec | Recipe (calls) | spec.json |
| Style | In recipe | Separate style.json |
| Data | Separate CSVs | Bundled data.csv |
| Stats | Not included | Available (.statsz) |
| GUI | Yes (editor) | Via scitex-cloud |

---

## Integration Points

### scitex.plt can use figrecipe (optional)

```python
import scitex.plt as splt

# Auto-enables figrecipe if installed
fig, ax = splt.subplots(use_figrecipe=True)

# Or disable even if installed
fig, ax = splt.subplots(use_figrecipe=False)
```

### Current Integration Files

| File | Purpose |
|------|---------|
| `scitex/plt/_figrecipe.py` | Wrapper for figrecipe functions |
| `scitex/bridge/_figrecipe.py` | Bridge to bundle format |
| `scitex/plt/_subplots/_SubplotsWrapper.py` | Auto-attaches recorder |

---

## Current Module Inventory

### In scitex

| Module | Purpose | Status |
|--------|---------|--------|
| `scitex.plt` | Plotting + CSV export | Core, keep |
| `scitex.io.bundle` | .pltz/.figz/.statsz | Core, keep |
| `scitex.fts` | FTS schemas | Overlaps with bundle, review |
| `scitex.fig` | Multi-panel canvas | Overlaps with .figz, review |
| `scitex.bridge` | figrecipe integration | Glue code, review |

### figrecipe (separate repo)

| Component | Purpose |
|-----------|---------|
| Recording | Captures matplotlib calls |
| Serialization | YAML recipe format |
| Reproduction | Recreates from recipe |
| GUI Editor | Visual editing interface |

---

## User Journey

```
New User                          Power User
   │                                  │
   ▼                                  │
┌──────────────┐                      │
│  figrecipe   │                      │
│  (simple)    │                      │
└──────┬───────┘                      │
       │                              │
       │ Needs more features?         │
       ▼                              ▼
┌─────────────────────────────────────────┐
│              scitex                     │
│  (full Research OS)                     │
│                                         │
│  • Publication workflow                 │
│  • Statistical analysis                 │
│  • Cloud collaboration                  │
│  • Bundle formats                       │
└─────────────────────────────────────────┘
```

---

## Decisions for Future

### Options Under Consideration

1. **Keep separate (current):**
   - figrecipe = entry point, marketing
   - scitex = full platform
   - Gradual cleanup of overlap

2. **Merge into scitex:**
   - figrecipe logic → scitex.plt
   - Single package
   - Archive figrecipe repo

3. **Strengthen integration:**
   - Better conversion between formats
   - figrecipe → .pltz pipeline
   - Keep both, improve bridge

### Current Assessment

The current architecture is **reasonable**. No urgent refactor needed.

**Recommendation:** Document clearly, clean up gradually, decide later.

---

## Related Documentation

- [ATOMIC_BUNDLE_PLAN.md](/docs/architecture/ATOMIC_BUNDLE_PLAN.md) - Architecture planning
- [scitex.io.bundle README](/src/scitex/io/bundle/README.md) - Bundle system docs
- [figrecipe README](https://github.com/ywatanabe1989/figrecipe) - figrecipe docs

<!-- EOF -->
