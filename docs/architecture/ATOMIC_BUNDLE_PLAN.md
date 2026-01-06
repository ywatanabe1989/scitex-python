# Atomic Bundle Architecture Plan

**Status:** For Review (No urgent action needed)
**Created:** 2026-01-06
**Updated:** 2026-01-06
**Author:** ywatanabe + Claude

> **Current Assessment:** The existing architecture is reasonable. Both figrecipe
> and scitex serve valid purposes. This document captures thinking for future
> reference, not immediate action.

---

## 1. Problem Statement

### Current Complexity

The scitex plotting ecosystem has grown organically with overlapping concerns:

```
Current modules:
├── scitex.plt          → Plotting + CSV export (50+ formatters)
├── figrecipe (external)→ Recipe recording for reproducibility
├── scitex.fts          → Bundle schemas (Node/Encoding/Theme/Stats)
├── scitex.fig          → Multi-panel canvas composition
└── scitex.bridge       → Glue between figrecipe and fts
```

**Issues:**
- Multiple "source of truth" candidates
- Unclear boundaries between modules
- Complex data flow
- Maintenance burden
- Cognitive overhead for users

### The Goal

A **single atomic bundle** that is the definitive source of truth for any scientific figure, containing everything needed to:
- View the figure
- Access the underlying data
- Modify the style
- Understand the statistics
- Reproduce the figure exactly

---

## 2. Requirements

### Must Have

| Requirement | Description |
|-------------|-------------|
| **R1: Image** | PNG/PDF/SVG output for publication |
| **R2: Data** | Underlying data in accessible format (CSV) |
| **R3: Style** | Editable aesthetics (colors, fonts, sizes) |
| **R4: Stats** | Statistical results with provenance |
| **R5: Recipe** | Ability to reproduce the figure exactly |
| **R6: Atomicity** | All components tightly linked as single unit |
| **R7: Backward Compat** | Existing workflows must continue to work |

### Should Have

| Requirement | Description |
|-------------|-------------|
| **R8: SigmaPlot CSV** | CSV format compatible with SigmaPlot |
| **R9: Raw matplotlib** | Option to use matplotlib without recording overhead |
| **R10: Multi-panel** | Compose multiple plots into publication figures |

### Nice to Have

| Requirement | Description |
|-------------|-------------|
| **R11: Versioning** | Track changes to bundles over time |
| **R12: Validation** | Schema validation for bundle contents |

---

## 3. Current State Analysis

### What Each Module Provides

#### scitex.plt
```
Strengths:
✓ Publication-quality defaults (mm-based sizing)
✓ Extensive CSV export (50+ formatters, SigmaPlot format)
✓ Style system (SCITEX_STYLE.yaml)
✓ Wrapper around matplotlib

Limitations:
✗ CSV export is scitex-specific, not standard
✗ Recipe recording depends on figrecipe
```

#### figrecipe (external)
```
Strengths:
✓ Records matplotlib calls
✓ Saves recipe as YAML
✓ Can reproduce figures exactly
✓ Well-designed, focused library

Limitations:
✗ No stats tracking
✗ CSV format differs from scitex
✗ External dependency
```

#### scitex.fts
```
Strengths:
✓ Stats tracking (stats.json)
✓ Schema separation (Node/Encoding/Theme)
✓ JSON schema validation

Limitations:
✗ Parallel structure to figrecipe
✗ Complex, overlapping with figrecipe goals
✗ Not widely adopted internally
```

#### scitex.fig
```
Strengths:
✓ Multi-panel composition
✓ Canvas-based layout
✓ Panel positioning in mm

Limitations:
✗ Composes separate images (not live plots)
✗ Different paradigm from figrecipe subplots
```

#### scitex.bridge._figrecipe
```
Purpose: Connect figrecipe to fts bundle format

Assessment: Glue code that adds complexity.
If we simplify, this likely becomes unnecessary.
```

---

## 4. Design Options

### ~~Option A: Extend figrecipe (External)~~ - Rejected

Rejected: Adds coordination overhead between two repos.

### ~~Option B: scitex-native Bundle (Replace figrecipe)~~ - Rejected

Rejected: Reinventing the wheel.

### ~~Option C: Layered Coexistence~~ - Rejected

Rejected: Two recording systems adds complexity.

### ~~Option D: Hybrid with scitex as Primary~~ - Superseded

Superseded by Option E.

---

### Option E: Monorepo with Extras (SELECTED)

**Context:** Same maintainer controls both figrecipe and scitex.

```
Approach:
- Merge figrecipe INTO scitex.plt
- Single repo, single package
- Optional dependencies via pip extras
- Archive/deprecate figrecipe repo

Installation:
  pip install scitex          # Core only
  pip install scitex[plt]     # + plotting + bundles
  pip install scitex[stats]   # + statistical analysis
  pip install scitex[all]     # Everything

Package structure:
scitex/
├── plt/
│   ├── _bundle.py          ← Atomic bundle logic (from figrecipe)
│   ├── _recipe.py          ← Recording logic (from figrecipe)
│   ├── _export_as_csv.py   ← CSV export (existing, 50+ formatters)
│   ├── _subplots/          ← Plotting wrappers (existing)
│   └── styles/             ← Theme system (existing)
├── stats/                  ← Statistical analysis
├── io/                     ← Save/load (uses plt._bundle)
└── ...

Bundle format (.stxb):
plot.stxb/
├── manifest.json       ← Metadata, checksums
├── figure.png          ← Image output
├── data.csv            ← SigmaPlot-compatible CSV
├── style.yaml          ← Editable theme
├── stats.json          ← Statistical results
└── recipe.yaml         ← Reproduction recipe

Pros:
+ Single package (pip install scitex[plt])
+ Single repo to maintain
+ No version coordination
+ Full control over everything
+ Simpler for sole maintainer
+ Optional deps keep base lightweight

Cons:
- figrecipe no longer standalone (but no external users anyway)
- Larger monorepo (but already is)
```

---

## 5. Recommendation

**Option E: Monorepo with Extras**

### Rationale

1. **Same maintainer for both** - No external coordination needed
2. **Sole user** - No community to break
3. **scitex CSV export is valuable** - 50+ formatters, keep as-is
4. **figrecipe logic is valuable** - Merge into scitex, don't discard
5. **Optional extras already exist** - `pip install scitex[plt]` works

### Final Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  pip install scitex[plt]                                    │
│                                                             │
│  scitex.plt                                                 │
│  ├── subplots()                                             │
│  │   ├── Creates matplotlib figure                          │
│  │   ├── Attaches recipe recorder (merged from figrecipe)   │
│  │   └── Attaches CSV tracker (existing)                    │
│  │                                                          │
│  ├── save(fig, "plot.stxb")                                 │
│  │   ├── figure.png    (matplotlib savefig)                 │
│  │   ├── data.csv      (scitex CSV export, SigmaPlot fmt)   │
│  │   ├── style.yaml    (scitex style system)                │
│  │   ├── stats.json    (statistical results)                │
│  │   └── recipe.yaml   (reproduction recipe)                │
│  │                                                          │
│  └── reproduce("plot.stxb")                                 │
│      └── Recreates figure from recipe                       │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Removed/Archived:                                          │
│  ├── figrecipe repo    → merged into scitex.plt             │
│  ├── scitex.fts        → simplified, merged into plt        │
│  ├── scitex.fig        → use native subplots()              │
│  └── scitex.bridge     → no longer needed                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Migration Plan

### Phase 1: Document & Stabilize (Current)
- [x] Document current architecture (this document)
- [ ] Document exact bundle format specification
- [ ] Add tests for current behavior (regression baseline)

### Phase 2: Create New Branch
- [ ] Create branch: `refactor/atomic-bundle`
- [ ] Keep `main` stable

### Phase 3: Implement Core Bundle
- [ ] Define `.stxb` bundle format specification
- [ ] Implement `save_bundle()` in scitex.plt
- [ ] Implement `load_bundle()` in scitex.plt
- [ ] Add stats tracking to axis wrapper

### Phase 4: Simplify
- [ ] Deprecate scitex.fts (or merge useful parts)
- [ ] Deprecate scitex.fig (document migration to subplots)
- [ ] Remove scitex.bridge._figrecipe

### Phase 5: Documentation & Release
- [ ] Update all documentation
- [ ] Migration guide for existing users
- [ ] Release with deprecation warnings

---

## 7. Bundle Format Specification (Draft)

### Directory Structure

```
{name}.stxb/
├── manifest.json       ← Bundle metadata, version, checksums
├── figure.png          ← Primary image
├── figure.pdf          ← Vector format (optional)
├── data.csv            ← SigmaPlot-compatible CSV
├── style.yaml          ← Editable visual style
├── stats.json          ← Statistical results (optional)
└── recipe.yaml         ← figrecipe recipe (optional)
```

### manifest.json

```json
{
  "version": "1.0.0",
  "created": "2026-01-06T12:00:00Z",
  "scitex_version": "0.x.x",
  "files": {
    "image": "figure.png",
    "data": "data.csv",
    "style": "style.yaml",
    "stats": "stats.json",
    "recipe": "recipe.yaml"
  },
  "checksums": {
    "data.csv": "sha256:abc123...",
    "figure.png": "sha256:def456..."
  }
}
```

### style.yaml

```yaml
# Editable style - changes here regenerate the figure
theme: light
colors:
  palette: ["#1f77b4", "#ff7f0e", "#2ca02c"]
  background: "#ffffff"
typography:
  family: Arial
  axis_label_pt: 7
  tick_label_pt: 6
lines:
  width_mm: 0.2
axes:
  width_mm: 40
  height_mm: 28
```

### stats.json

```json
{
  "version": "1.0.0",
  "analyses": [
    {
      "id": "comparison_1",
      "test": "t-test",
      "variant": "independent",
      "groups": ["control", "treatment"],
      "n": [30, 28],
      "statistic": 2.45,
      "p_value": 0.018,
      "effect_size": {
        "name": "cohens_d",
        "value": 0.65,
        "ci": [0.12, 1.18]
      }
    }
  ],
  "software": {
    "python": "3.11",
    "scipy": "1.11.0"
  }
}
```

---

## 8. Open Questions

1. **Multi-panel figures**: Use figrecipe native `subplots(2,2)` or keep scitex.fig?
2. **ZIP vs Directory**: `.stxb` as zip file or directory?
3. **Recipe optionality**: Always include recipe, or only when `use_figrecipe=True`?
4. **Style regeneration**: If style.yaml is edited, how to regenerate figure?
5. **Backward compat timeline**: How long to support old formats?

---

## 9. Next Steps

1. **Review this document** - Get feedback on the approach
2. **Finalize bundle spec** - Answer open questions
3. **Create feature branch** - `refactor/atomic-bundle`
4. **Prototype** - Implement minimal bundle save/load
5. **Iterate** - Refine based on real usage

---

## 10. Current Assessment (2026-01-06)

After thorough analysis, the current architecture is **reasonable**:

### What's Working Well

| Component | Status | Notes |
|-----------|--------|-------|
| figrecipe | Good | Simple entry point, GUI editor, attracts users |
| scitex.plt | Good | Extensive CSV export (50+ formatters), publication quality |
| scitex.io.bundle | Good | Well-designed .pltz/.figz/.statsz system |
| Bundle format | Good | Already atomic: spec + style + data + exports |

### What Could Be Cleaner (Low Priority)

| Component | Issue | Priority |
|-----------|-------|----------|
| scitex.fts | Some overlap with bundle schemas | Low |
| scitex.fig | Some overlap with .figz format | Low |
| scitex.bridge | Glue code, could simplify | Low |

### Strategic Value of Keeping Both

```
figrecipe (Entry Point)           scitex (Full Platform)
┌─────────────────────┐           ┌─────────────────────┐
│ • Simple API        │           │ • Research OS       │
│ • GUI editor        │    →      │ • Cloud (scitex.ai) │
│ • Marketing appeal  │  users    │ • Full workflow     │
│ • Low barrier       │  graduate │ • Publication ready │
└─────────────────────┘           └─────────────────────┘
```

### Recommendation

**No urgent refactor needed.** Instead:

1. Keep current architecture
2. Document relationships clearly (done)
3. Clean up overlap gradually over time
4. Revisit decision if user base grows

---

## Appendix: File Counts (Current State)

```
scitex.plt:
  - _export_as_csv_formatters/: 50+ files
  - Core files: ~20

scitex.fts:
  - Core files: ~15
  - Schemas: 5

scitex.fig:
  - Core files: ~20

scitex.bridge:
  - _figrecipe.py: 1 file
```

Total complexity to potentially simplify: ~100+ files across modules.

<!-- EOF -->
