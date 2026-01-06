# SciTeX Terminology Guide

**Status:** Approved
**Created:** 2026-01-07
**Author:** ywatanabe + Claude

---

## Overview

This document establishes consistent terminology across scitex and figrecipe libraries.
The goal is to eliminate confusion while respecting established standards.

---

## Terminology Hierarchy

### Level 1: matplotlib Standards (DO NOT CHANGE)

These terms are established by matplotlib and must remain consistent:

| Term | Type | Definition | Example |
|------|------|------------|---------|
| `fig` | Variable | matplotlib.figure.Figure instance | `fig = plt.figure()` |
| `figure` | Class/noun | The Figure object or class | `matplotlib.figure.Figure` |
| `ax` | Variable | Single matplotlib.axes.Axes instance | `ax = fig.add_subplot()` |
| `axes` | Variable/Class | Axes instance(s) or class | `fig, axes = plt.subplots(2,2)` |
| `plot()` | Method | Line plot visualization call | `ax.plot(x, y)` |
| `scatter()` | Method | Scatter plot visualization call | `ax.scatter(x, y)` |

**Rule:** Never redefine these terms to mean something else.

### Level 2: Scientific Publication Standards (DO NOT CHANGE)

| Term | Context | Definition | Example |
|------|---------|------------|---------|
| Figure | Publication | Numbered figure in paper | "Figure 1 shows..." |
| Table | Publication | Numbered table in paper | "Table 2 summarizes..." |
| Panel | Publication | Sub-part of a figure | "Figure 1A (left panel)" |

**Rule:** When referring to publication elements, use capitalized form.

### Level 3: scitex-Specific Terms (USE CONSISTENTLY)

| Term | Module | Definition | Replaces |
|------|--------|------------|----------|
| `trace` | plt, fts | Single data series (x,y pair) within a plot | "series", "line" |
| `panel` | fig | A plot positioned on a canvas | "subplot" in composition context |
| `canvas` | fig | Multi-panel composition workspace | "figure" in composition context |
| `bundle` | io, fts | Atomic package containing data + spec + exports | "package", "archive" |
| `recipe` | figrecipe | YAML reproduction specification | "config", "spec" |
| `encoding` | fts | Data-to-visual mapping specification | "mapping" |
| `theme` | fts, plt | Visual aesthetics (colors, fonts, sizes) | "style" (when referring to scitex themes) |

---

## Context-Specific Usage

### When Creating Plots (scitex.plt)

```python
import scitex.plt as splt

# matplotlib terms - unchanged
fig, ax = splt.subplots()
ax.plot(x, y, id="my_trace")  # 'trace' = data series identifier

# scitex terms
ax.add_trace(...)  # Adding a data series
df = ax.export_as_csv()  # Export trace data
```

### When Composing Multi-Panel Figures (scitex.fig)

```python
import scitex.fig as sfig

# scitex terms
canvas = sfig.create_canvas(...)  # 'canvas' = composition workspace
sfig.add_panel(canvas, ...)       # 'panel' = positioned plot
sfig.export_figure(canvas, ...)   # Creates publication Figure
```

### When Working with Bundles (scitex.io, scitex.fts)

```python
import scitex.io as sio

# scitex terms
bundle = sio.load("plot.pltz")     # 'bundle' = atomic package
bundle.encoding                     # 'encoding' = data-visual mapping
bundle.theme                        # 'theme' = aesthetics
```

### When Using figrecipe

```python
import figrecipe as fr

# figrecipe terms
fig, ax = fr.subplots()
fr.save(fig, "plot.yaml")          # 'recipe' = reproduction spec
fr.reproduce("plot.yaml")          # Recreate from recipe
```

---

## Confusion Matrix (Before This Guide)

| Concept | matplotlib | scitex.plt | scitex.fig | scitex.fts | figrecipe |
|---------|------------|------------|------------|------------|-----------|
| Figure object | fig/figure | fig | - | figure | fig |
| Axes object | axes/ax | axes/ax | - | axes | axes |
| Single plot | - | plot | panel | plot | - |
| Multi-panel | - | - | canvas | figure | - |
| Data series | - | trace | - | trace | - |
| Package | - | - | - | bundle | recipe |

## Confusion Matrix (After This Guide)

| Concept | Term to Use | Notes |
|---------|-------------|-------|
| matplotlib Figure | `fig`, `figure` | Keep matplotlib standard |
| matplotlib Axes | `ax`, `axes` | Keep matplotlib standard |
| Single visualization | `plot` | The act of plotting |
| Data series | `trace` | x,y data pair with id |
| Composition workspace | `canvas` | scitex.fig only |
| Positioned plot | `panel` | On a canvas |
| Atomic package | `bundle` | .pltz/.figz/.statsz |
| Reproduction spec | `recipe` | figrecipe YAML |
| Publication figure | `Figure` (capitalized) | "Figure 1" |

---

## Naming Conventions

### File/Directory Names

| Type | Convention | Example |
|------|------------|---------|
| Plot bundle | `{name}.pltz` or `{name}.plot.zip` | `scatter_analysis.pltz` |
| Figure bundle | `{name}.figz` or `{name}.figure.zip` | `fig1_results.figz` |
| Stats bundle | `{name}.statsz` or `{name}.stats.zip` | `anova_results.statsz` |
| Recipe file | `{name}.yaml` | `plot_recipe.yaml` |

### Variable Names

| Context | Convention | Example |
|---------|------------|---------|
| Figure object | `fig` | `fig, ax = plt.subplots()` |
| Single axes | `ax` | `ax.plot(...)` |
| Multiple axes | `axes` or `axs` | `fig, axes = plt.subplots(2,2)` |
| Specific axes | `ax_{row}_{col}` | `ax_0_1` |
| Trace identifier | `id="descriptive_name"` | `id="control_group"` |
| Canvas | `canvas` | `canvas = create_canvas(...)` |
| Bundle | `bundle` | `bundle = load("plot.pltz")` |

### Function Names

| Action | Convention | Example |
|--------|------------|---------|
| Create | `create_*` | `create_canvas()` |
| Load | `load()` or `load_*` | `load("plot.pltz")` |
| Save | `save()` or `save_*` | `save(fig, "plot.png")` |
| Export | `export_*` | `export_as_csv()` |
| Add | `add_*` | `add_panel()` |

---

## Migration Notes

If updating existing code:

1. **Keep matplotlib terms** - No changes needed for `fig`, `ax`, `axes`, `plot()`
2. **Rename "series" to "trace"** - If used for data series
3. **Rename "subplot" to "panel"** - In composition context only
4. **Rename "workspace" to "canvas"** - In scitex.fig
5. **Use "bundle" consistently** - For .pltz/.figz/.statsz packages

---

## References

- matplotlib terminology: https://matplotlib.org/stable/api/index.html
- scitex.plt: `src/scitex/plt/`
- scitex.fig: `src/scitex/fig/`
- scitex.fts: `src/scitex/fts/`
- figrecipe: https://github.com/ywatanabe1989/figrecipe

<!-- EOF -->
