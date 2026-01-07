<!-- ---
!-- Timestamp: 2026-01-07 15:28:47
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/docs/visualization/01_TERMINOLOGY.md
!-- --- -->

# Terminology

## matplotlib (unchanged)

| Term | Usage |
|------|-------|
| `fig` | Figure instance |
| `ax` | Single Axes |
| `axes` | Multiple Axes |
| `plot()` | Line plot method |

## scitex Terms

| Term | Definition |
|------|------------|
| `trace` | Single data series (x,y pair with id) |
| `panel` | Positioned plot on a canvas |
| `canvas` | Multi-panel composition workspace |
| `bundle` | Atomic package (.pltz/.figz/.statsz) |
| `encoding` | Data-to-visual mapping |
| `theme` | Visual aesthetics (colors, fonts) |

## File Extensions

| Extension | Contents |
|-----------|----------|
| `.pltz` | Single plot bundle |
| `.figz` | Multi-panel figure bundle |
| `.statsz` | Statistical results bundle |

## Usage Examples

```python
import scitex.plt as splt
import scitex.canvas as scanvas
import scitex.io as sio

# Data
x = y = [0, 1, 2]

# Plotting (trace = data series with id)
fig, ax = splt.subplots()
ax.plot(x, y, id="my_trace")

# Save figure (creates bundle + exports)
sio.save(fig, "/tmp/my_plot.png")

# Canvas composition (panel = positioned plot)
scanvas.create_canvas("/tmp", "my_figure")
scanvas.add_panel("/tmp", "my_figure", "panel_a", "/tmp/my_plot.png",
                  xy_mm=(10, 10), size_mm=(80, 60), label="A")
```

<!-- EOF -->