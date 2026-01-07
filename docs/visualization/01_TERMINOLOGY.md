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
# Plotting
fig, ax = splt.subplots()
ax.plot(x, y, id="my_trace")  # trace = data series

# Composition
canvas = scanvas.create_canvas()
scanvas.add_panel(canvas, ...)  # panel = positioned plot

# Bundles
bundle = sio.load("plot.pltz")  # bundle = atomic package
```

<!-- EOF -->
