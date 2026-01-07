# Modules

## scitex.plt

**Purpose:** Publication-quality plotting

```python
import scitex.plt as splt

fig, ax = splt.subplots()
ax.plot(x, y, id="trace_name")
ax.export_as_csv("data.csv")  # SigmaPlot format
```

**Features:**
- MM-based sizing for publications
- 50+ CSV formatters (SigmaPlot compatible)
- Style system (SCITEX_STYLE.yaml)

## scitex.canvas

**Purpose:** Multi-panel figure composition

```python
import scitex.canvas as scanvas

canvas = scanvas.create_canvas(width_mm=180, height_mm=120)
scanvas.add_panel(canvas, plot_bundle, bbox=[0, 0, 90, 60])
```

**Features:**
- Panel positioning in mm
- Exports to .figz bundle

## scitex.io.bundle

**Purpose:** Atomic bundle I/O

```python
import scitex.io as sio

# Save figure as bundle
sio.save(fig, "plot.pltz")

# Load bundle
bundle = sio.load("plot.pltz")
```

**Features:**
- .pltz/.figz/.statsz formats
- Data + spec + theme in one package
- Schemas for validation

## figrecipe (external)

**Purpose:** Simple reproducible plots

```python
import figrecipe as fr

fig, ax = fr.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
fr.save(fig, "plot.yaml")  # Recipe format
```

**Relationship:** Entry point for new users. Power users graduate to scitex.

<!-- EOF -->
