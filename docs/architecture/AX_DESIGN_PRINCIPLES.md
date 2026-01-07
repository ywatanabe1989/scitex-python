# AX (AI Experience) Design Principles

**Status:** Vision Document
**Created:** 2026-01-07
**Author:** ywatanabe + Claude

---

## What is AX?

**AX (AI Experience)** is to AI agents what **UX (User Experience)** is to humans.

| Term | Focus | Goal |
|------|-------|------|
| UI | Human interface | Visual clarity |
| UX | Human workflow | Ease of use |
| **AX** | AI agent workflow | Structured, constrained, verifiable |

scitex is designed for **both** human researchers and AI research agents.

---

## Core Philosophy: The Highway Principle

> "Reduce degrees of freedom. AI agents don't want choices - they want correctness."

### Country Road (Bad AX)

```
High degrees of freedom:
├── Which library?      → matplotlib? plotly? seaborn?
├── Which format?       → png? svg? pdf? pickle?
├── Where to save?      → ./output? ./figures? ./data?
├── What naming?        → fig1.png? figure_2024_01.png?
├── How to store data?  → csv? json? parquet?
├── What structure?     → flat? nested? normalized?
└── Each decision point = potential error
```

**Result:** AI agent must make many decisions, each a potential failure point.

### Highway (Good AX)

```
Low degrees of freedom:
└── scitex.save(fig, "analysis.plot.zip")
    ├── Format: decided (bundle)
    ├── Structure: decided (canonical/)
    ├── Naming: decided (manifest.json)
    ├── Data: included (data.csv)
    └── Provenance: tracked (recipe.yaml)

One way. Correct way. Done.
```

**Result:** AI agent follows the highway - fast, predictable, error-free.

---

## The Four Pillars of AX

### 1. Provenance (Where Data Comes From)

AI agents need to trace data origin for trust and reproducibility.

```
Bad AX:
- Figure exists, but where's the data?
- Data exists, but how was it processed?
- Code exists, but which version made this?

Good AX (scitex bundle):
analysis.plot.zip/
├── manifest.json     ← Created when, by what, version
├── data.csv          ← Source data, checksummed
├── recipe.yaml       ← Exact code that generated figure
└── lineage.json      ← Parent bundles, transformations
```

**Principle:** Every artifact must answer "where did you come from?"

### 2. Structure (How Data is Organized)

AI agents parse structure, not intent. Ambiguity is failure.

```
Bad AX:
- "The data is somewhere in the repo"
- "Column names vary by experiment"
- "Format depends on the instrument"

Good AX:
- Strict JSON schema for all specs
- Canonical directory structure
- Validated on save and load
- Self-describing via manifest
```

**Principle:** Structure should be machine-verifiable, not human-interpretable.

### 3. Validation (How to Verify Correctness)

AI agents need binary yes/no, not "looks okay."

```python
# Bad AX: Implicit validation
fig.savefig("output.png")  # Did it work? Check manually?

# Good AX: Explicit validation
bundle = scitex.save(fig, "output.plot.zip")
assert bundle.validate()      # Schema conformance
assert bundle.verify()        # Reproducibility check
assert bundle.checksum()      # Integrity check
```

**Principle:** Every operation should be verifiable programmatically.

### 4. Constraint (Reduced Decision Points)

Every decision point is a potential error. Eliminate them.

```python
# Bad AX: Many parameters, many decisions
plt.savefig(
    filename,           # What name?
    format="png",       # What format?
    dpi=300,            # What resolution?
    bbox_inches="tight",# What cropping?
    facecolor="white",  # What background?
    transparent=False,  # Transparent?
)

# Good AX: One way, correct way
scitex.save(fig, "name.plot.zip")
# Format: bundle (decided)
# Contents: everything needed (decided)
# Structure: canonical (decided)
```

**Principle:** Opinionated defaults > flexible options.

---

## AX Design Patterns

### Pattern 1: Self-Describing Data

Data should contain its own documentation.

```json
{
  "manifest": {
    "type": "plot",
    "version": "1.0.0",
    "created": "2026-01-07T12:00:00Z",
    "creator": "scitex.plt",
    "schema": "https://scitex.ai/schemas/plot/1.0"
  },
  "contents": {
    "data": "data.csv",
    "spec": "spec.json",
    "recipe": "recipe.yaml"
  },
  "checksums": {
    "data.csv": "sha256:abc123..."
  }
}
```

### Pattern 2: Atomic Operations

All-or-nothing. No partial states.

```python
# Bad AX: Partial failure possible
save_data(data, "data.csv")      # Success
save_spec(spec, "spec.json")     # Success
save_figure(fig, "fig.png")      # FAILURE - now inconsistent state

# Good AX: Atomic bundle
scitex.save(fig, "output.plot.zip")  # All or nothing
```

### Pattern 3: Deterministic Output

Same input must produce same output.

```python
# Bad AX: Non-deterministic
fig.savefig("out.png")  # Timestamp in metadata? Random seed?

# Good AX: Deterministic, content-addressable
bundle = scitex.save(fig, "out.plot.zip")
assert bundle.hash() == "sha256:abc123..."  # Always same hash for same content
```

### Pattern 4: Introspectable State

AI agents need to query, not guess.

```python
# Bad AX: Opaque state
fig = plt.figure()
# What's in it? How many axes? What data?

# Good AX: Queryable state
bundle.describe()
# → {type: "plot", axes: 1, traces: 3, rows: 150, ...}

bundle.schema()
# → Full JSON schema of contents

bundle.diff(other_bundle)
# → Structured diff of what changed
```

### Pattern 5: Explicit Lineage

Track derivation, not just state.

```python
# Create derived bundle with lineage
new_bundle = original_bundle.derive(
    modifications={"theme": "dark"},
    reason="Style change for publication"
)

new_bundle.lineage()
# → {
#     parent: "original.plot.zip",
#     parent_hash: "sha256:abc...",
#     modifications: {...},
#     reason: "Style change for publication"
#   }
```

---

## AX vs UX Comparison

| Aspect | UX (Human) | AX (AI Agent) |
|--------|------------|---------------|
| Interface | Visual, intuitive | Structured, programmatic |
| Feedback | Messages, colors | Return codes, schemas |
| Flexibility | Appreciated | Burden |
| Defaults | Suggestions | Requirements |
| Validation | "Looks right" | Schema conformance |
| Documentation | Prose | Type hints, schemas |
| Errors | Helpful messages | Structured error objects |
| State | Can be implicit | Must be explicit |

---

## Implementation Guidelines

### For scitex Developers

1. **Every function should be deterministic** where possible
2. **Every data structure should have a JSON schema**
3. **Every operation should be validatable**
4. **Minimize parameters** - use opinionated defaults
5. **Return structured results**, not just success/failure
6. **Include provenance** in all artifacts

### API Design for AX

```python
# Return structured results
result = scitex.save(fig, "output.plot.zip")
result.success      # bool
result.path         # Path
result.hash         # str
result.warnings     # List[Warning]
result.manifest     # Dict

# Structured errors
try:
    bundle.validate()
except ValidationError as e:
    e.code          # "SCHEMA_MISMATCH"
    e.path          # "spec.json/axes/0/type"
    e.expected      # "string"
    e.actual        # "integer"
    e.suggestion    # "Convert axis type to string"
```

### Schema-First Development

```yaml
# Define schema FIRST
# schemas/plot-bundle.schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["manifest", "data", "spec"],
  "properties": {
    "manifest": {"$ref": "#/definitions/manifest"},
    "data": {"$ref": "#/definitions/data"},
    "spec": {"$ref": "#/definitions/spec"}
  }
}

# Then implement to match schema
# Not: implement first, document later
```

---

## Metrics for AX Quality

| Metric | Description | Target |
|--------|-------------|--------|
| Decision points | Number of required parameters | Minimize |
| Schema coverage | % of data with JSON schema | 100% |
| Validation coverage | % of operations validatable | 100% |
| Determinism | Same input → same output | Always |
| Introspectability | Can query any state? | Yes |
| Provenance depth | Can trace to origin? | Always |

---

## Vision: AI Research Agents

scitex is infrastructure for AI research agents:

```
AI Research Workflow:

  ┌─────────────┐
  │ Hypothesis  │
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Experiment  │ ← scitex.stats
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │   Data      │ ← scitex bundles
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Figure     │ ← scitex.plt
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Verify     │ ← bundle.verify()
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Iterate    │ ← bundle.derive()
  └─────────────┘

Each step produces traceable, verifiable, reproducible artifacts.
```

---

## Summary

**AX Design Principles:**

1. **Provenance** - Know where everything comes from
2. **Structure** - Machine-verifiable, schema-validated
3. **Validation** - Binary yes/no, not "looks okay"
4. **Constraint** - Highway, not country roads

**The Goal:**

> AI agents should be able to conduct research using scitex
> with the same rigor as human scientists - but faster,
> more consistent, and fully traceable.

---

## References

- scitex bundle specification: `docs/architecture/EXTENSION_MIGRATION_PLAN.md`
- Terminology guide: `docs/architecture/TERMINOLOGY_GUIDE.md`
- JSON Schema: https://json-schema.org/

<!-- EOF -->
