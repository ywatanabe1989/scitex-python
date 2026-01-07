<!-- ---
!-- Timestamp: 2026-01-07
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/io/bundle/README.md
!-- --- -->

# SciTeX Bundle I/O

## Overview

SciTeX uses bundle formats for reproducible scientific figures:

| Format         | Purpose             | Contents                                       |
|----------------|---------------------|------------------------------------------------|
| `.plot.zip`    | Single plot bundle  | spec.json, encoding.json, data.csv, exports/   |
| `.figure.zip`  | Multi-panel figure  | spec.json, encoding.json, nested plot bundles  |
| `.stats.zip`   | Statistical results | spec.json, stats data, comparison metadata     |

## Format Variants

Each bundle can exist in two forms:

| Suffix                                       | Type        | Use Case                    |
|----------------------------------------------|-------------|-----------------------------|
| `.plot.zip` / `.figure.zip` / `.stats.zip`   | ZIP archive | Storage, transfer, download |
| `.plot` / `.figure` / `.stats`               | Directory   | Editing, development        |

## Quick Start

```python
import scitex.io.bundle as bundle

# Load a bundle (works with both ZIP and directory)
data = bundle.load("Figure1.figure.zip")
print(data['spec'])
print(data['type'])  # 'figure'

# Save a bundle
bundle.save(data, "output.plot.zip", as_zip=True)

# Copy a bundle
bundle.copy("template.plot.zip", "my_plot.plot")

# Pack/unpack between formats
bundle.pack("plot.plot")        # -> plot.plot.zip
bundle.unpack("plot.plot.zip")  # -> plot.plot/
```

## ZipBundle Class

In-memory access to ZIP bundles without extraction:

```python
from scitex.io.bundle import ZipBundle

# Reading
with ZipBundle("figure.figure.zip") as zb:
    spec = zb.read_json("spec.json")
    data = zb.read_csv("data.csv")
    png = zb.read_bytes("exports/figure.png")

# Writing (atomic)
with ZipBundle("output.plot.zip", mode="w") as zb:
    zb.write_json("spec.json", spec_dict)
    zb.write_csv("data.csv", dataframe)
    zb.write_bytes("exports/plot.png", png_bytes)

# Modifying (read + write atomically)
with ZipBundle("figure.figure.zip", mode="a") as zb:
    spec = zb.read_json("spec.json")
    spec["title"] = "Updated"
    zb.write_json("spec.json", spec)
```

## Nested Bundle Access

Access plot bundles nested inside figure bundles:

```python
from scitex.io.bundle import nested

# Get preview image
preview = nested.get_preview("Figure1.figure/A.plot")

# Get JSON
spec = nested.get_json("Figure1.figure/A.plot/spec.json")

# Get any file
png = nested.get_file("Figure1.figure/A.plot/exports/plot.png")

# Write to nested bundle
nested.put_json("Figure1.figure/A.plot/spec.json", updated_spec)

# List files
files = nested.list_files("Figure1.figure/A.plot")

# Resolve full bundle data
data = nested.resolve("Figure1.figure/A.plot")
```

## Bundle Structure

### .plot Bundle

```
plot.plot/
├── canonical/
│   ├── spec.json      # Plot specification (kind, id, size, etc.)
│   ├── encoding.json  # Encoding specification (traces, axes)
│   └── theme.json     # Visual styling (colors, fonts, sizes)
├── payload/
│   └── data.csv       # Source data
├── artifacts/
│   ├── exports/
│   │   ├── plot.png   # Rendered preview
│   │   └── plot.svg
│   └── cache/
│       └── geometry_px.json
└── children/          # Empty for leaf bundles
```

### .figure Bundle

```
Figure1.figure/
├── canonical/
│   ├── spec.json      # Figure layout, panel positions, children list
│   ├── encoding.json  # Figure-level encoding
│   └── theme.json     # Figure-level styling
├── payload/           # Empty for composite bundles
├── artifacts/
│   └── exports/
│       └── Figure1.png
└── children/
    ├── panel-a.zip    # Panel A (nested plot bundle)
    └── panel-b.zip    # Panel B (nested plot bundle)
```

## API Reference

### Core Operations

| Function | Description |
|----------|-------------|
| `load(path)` | Load bundle from ZIP or directory |
| `save(data, path)` | Save bundle to ZIP or directory |
| `copy(src, dst)` | Copy bundle between locations |
| `pack(dir_path)` | Convert directory to ZIP |
| `unpack(zip_path)` | Convert ZIP to directory |
| `validate(path)` | Validate bundle structure |
| `is_bundle(path)` | Check if path is a bundle |
| `get_type(path)` | Get bundle type ('figure', 'plot', 'stats') |

### ZipBundle Methods

| Method | Description |
|--------|-------------|
| `read_bytes(name)` | Read file as bytes |
| `read_text(name)` | Read file as string |
| `read_json(name)` | Read and parse JSON |
| `read_csv(name)` | Read CSV as DataFrame |
| `write_bytes(name, data)` | Write bytes |
| `write_text(name, text)` | Write string |
| `write_json(name, data)` | Write JSON |
| `write_csv(name, df)` | Write DataFrame as CSV |
| `namelist()` | List files in bundle |

### Nested Access (bundle.nested)

| Function | Description |
|----------|-------------|
| `nested.get_file(path)` | Get file from nested bundle |
| `nested.get_json(path)` | Get JSON from nested bundle |
| `nested.get_preview(path)` | Get preview PNG |
| `nested.put_file(path, data)` | Write file to nested bundle |
| `nested.put_json(path, data)` | Write JSON to nested bundle |
| `nested.list_files(path)` | List files in nested bundle |
| `nested.resolve(path)` | Load full nested bundle data |
| `nested.parse_path(path)` | Parse nested path components |

### Bundle and Spec Classes

```python
from scitex.io.bundle import Bundle, Spec, SpecRefs

# Create a new bundle
bundle = Bundle("my_plot.plot", create=True, kind="plot")
bundle.spec.name = "My Plot"
bundle.save()

# Load existing bundle
bundle = Bundle("existing.plot.zip")
print(bundle.spec.kind)  # 'plot'
print(bundle.spec.id)

# Create from matplotlib
from scitex.io.bundle import from_matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
bundle = from_matplotlib(fig, "output.plot")
```

## Module Structure

```
scitex/io/bundle/
├── __init__.py        # Public API exports
├── _types.py          # BundleType, errors, constants
├── _core.py           # load, save, copy, pack, unpack, validate
├── _zip.py            # ZipBundle class and functions
├── _nested.py         # Nested bundle access
├── _Bundle.py         # Bundle class
├── _dataclasses/      # Spec, SpecRefs, BBox, SizeMM, etc.
├── _loader.py         # Bundle component loading
├── _saver.py          # Bundle component saving
└── README.md          # This documentation
```

<!-- EOF -->
