# Visualization Architecture

## Modules

```
scitex.plt      → Plotting (matplotlib wrapper, CSV export)
scitex.canvas   → Multi-panel composition
scitex.io.bundle→ Bundle format (.pltz/.figz/.statsz)
figrecipe       → External: simple reproducible plots
```

## Data Flow

```
User Code
    │
    ▼
scitex.plt.subplots()  →  fig, ax (matplotlib)
    │
    ▼
ax.plot(x, y, id="trace_name")
    │
    ▼
scitex.io.save(fig, "plot.pltz")
    │
    ▼
Bundle (.pltz)
├── spec.json      # Plot specification
├── data.csv       # Source data (SigmaPlot format)
├── theme.json     # Visual styling
└── exports/       # PNG, SVG, PDF
```

## Related Docs

1. [Terminology](01_TERMINOLOGY.md) - Naming conventions
2. [Bundle Format](02_BUNDLE_FORMAT.md) - .pltz/.figz/.statsz specification
3. [Modules](03_MODULES.md) - Module responsibilities

<!-- EOF -->
