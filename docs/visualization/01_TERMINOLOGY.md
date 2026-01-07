<!-- ---
!-- Timestamp: 2026-01-07 15:43:56
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/docs/visualization/01_TERMINOLOGY.md
!-- --- -->

# Terminology

## matplotlib (unchanged)

| Term     | Usage            |
|----------|------------------|
| `fig`    | Figure instance  |
| `ax`     | Single Axes      |
| `axes`   | Multiple Axes    |
| `plot()` | Line plot method |

## scitex Terms

| Term       | Definition                            | Class        |
|------------|---------------------------------------|--------------|
| `trace`    | Single data series (x,y pair with id) | (dataclass)  |
| `panel`    | Positioned plot on a canvas           | (via Canvas) |
| `canvas`   | Multi-panel composition workspace     | `Canvas`     |
| `bundle`   | Atomic package (.pltz/.figz/.statsz)  | `Bundle`     |
| `encoding` | Data-to-visual mapping                | (via Bundle) |
| `theme`    | Visual aesthetics (colors, fonts)     | (via Bundle) |

## File Extensions

| xtension | Contents                   |
|-----------|----------------------------|
| `.pltz`   | Single plot bundle         |
| `.figz`   | Multi-panel figure bundle  |
| `.statsz` | Statistical results bundle |

## Usage Examples

```python
import scitex.plt as splt
import scitex.canvas as scanvas
import scitex.io as sio
from scitex.io.bundle import Bundle

# Data
x = y = [0, 1, 2]

# Plotting (trace = data series with id)
fig, ax = splt.subplots()
ax.plot(x, y, id="my_trace")

# Save figure (creates bundle + exports)
sio.save(fig, "/tmp/my_plot.png")

# Canvas composition (panel = positioned plot)
canvas = scanvas.Canvas("my_figure", width_mm=180, height_mm=120)
canvas.add_panel("panel_a", "/tmp/my_plot.png",
                 xy_mm=(10, 10), size_mm=(80, 60), label="A")
canvas.save("/tmp/my_figure.canvas")

# Bundle I/O (OOP)
bundle = Bundle("/tmp/my_plot.pltz.d")
bundle.save()
```

<!-- EOF -->
