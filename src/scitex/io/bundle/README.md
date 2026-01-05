<!-- ---
!-- Timestamp: 2025-12-16 20:36:14
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/io/bundle/README.md
!-- --- -->

# SciTeX Bundle I/O

## Overview

SciTeX uses bundle formats for reproducible scientific figures:

| Format    | Purpose             | Contents                                    |
|-----------|---------------------|---------------------------------------------|
| `.pltz`   | Single plot bundle  | spec.json, style.json, data.csv, exports/   |
| `.figz`   | Multi-panel figure  | spec.json, style.json, nested .pltz bundles |
| `.statsz` | Statistical results | spec.json, stats data, comparison metadata  |

## Format Variants

Each bundle can exist in two forms:

| Suffix                              | Type        | Use Case                    |
|-------------------------------------|-------------|-----------------------------|
| `.pltz` / `.figz` / `.statsz`       | ZIP archive | Storage, transfer, download |
| `.pltz.d` / `.figz.d` / `.statsz.d` | Directory   | Editing, development        |

## Quick Start

```python
import scitex.io.bundle as bundle

# Load a bundle (works with both ZIP and directory)
data = bundle.load("Figure1.figz")
print(data['spec'])
print(data['type'])  # 'figz'

# Save a bundle
bundle.save(data, "output.pltz", as_zip=True)

# Copy a bundle
bundle.copy("template.pltz", "my_plot.pltz.d")

# Pack/unpack between formats
bundle.pack("plot.pltz.d")      # -> plot.pltz
bundle.unpack("plot.pltz")       # -> plot.pltz.d/
```

## ZipBundle Class

In-memory access to ZIP bundles without extraction:

```python
from scitex.io.bundle import ZipBundle

# Reading
with ZipBundle("figure.figz") as zb:
    spec = zb.read_json("spec.json")
    data = zb.read_csv("data.csv")
    png = zb.read_bytes("exports/figure.png")

# Writing (atomic)
with ZipBundle("output.pltz", mode="w") as zb:
    zb.write_json("spec.json", spec_dict)
    zb.write_csv("data.csv", dataframe)
    zb.write_bytes("exports/plot.png", png_bytes)

# Modifying (read + write atomically)
with ZipBundle("figure.figz", mode="a") as zb:
    spec = zb.read_json("spec.json")
    spec["title"] = "Updated"
    zb.write_json("spec.json", spec)
```

## Nested Bundle Access

Access pltz bundles nested inside figz:

```python
from scitex.io.bundle import nested

# Get preview image
preview = nested.get_preview("Figure1.figz/A.pltz.d")

# Get JSON
spec = nested.get_json("Figure1.figz/A.pltz.d/spec.json")

# Get any file
png = nested.get_file("Figure1.figz/A.pltz.d/exports/plot.png")

# Write to nested bundle
nested.put_json("Figure1.figz/A.pltz.d/spec.json", updated_spec)

# List files
files = nested.list_files("Figure1.figz/A.pltz.d")

# Resolve full bundle data
data = nested.resolve("Figure1.figz/A.pltz.d")
```

## Bundle Structure

### .pltz Bundle

```
plot.pltz.d/
├── spec.json          # Plot specification (traces, axes, data refs)
├── style.json         # Visual styling (colors, fonts, sizes)
├── data.csv           # Source data
├── exports/
│   ├── plot.png       # Rendered preview
│   ├── plot_hitmap.png
│   └── plot.svg
└── cache/
    └── geometry_px.json
```

### .figz Bundle

```
Figure1.figz.d/
├── spec.json          # Figure layout, panel positions
├── style.json         # Figure-level styling
├── A.pltz.d/          # Panel A (nested pltz)
├── B.pltz.d/          # Panel B (nested pltz)
└── exports/
    └── Figure1.png    # Composed figure
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
| `get_type(path)` | Get bundle type ('figz', 'pltz', 'statsz') |

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

## Migration from Old API

| Old Import | New Import |
|------------|------------|
| `from scitex.io._bundle import load_bundle` | `from scitex.io.bundle import load` |
| `from scitex.io._bundle import save_bundle` | `from scitex.io.bundle import save` |
| `from scitex.io._bundle import copy_bundle` | `from scitex.io.bundle import copy` |
| `from scitex.io._bundle import pack_bundle` | `from scitex.io.bundle import pack` |
| `from scitex.io._bundle import unpack_bundle` | `from scitex.io.bundle import unpack` |
| `from scitex.io._bundle import validate_bundle` | `from scitex.io.bundle import validate` |
| `from scitex.io._bundle import BundleType` | `from scitex.io.bundle import BundleType` |
| `from scitex.io._bundle import BundleValidationError` | `from scitex.io.bundle import BundleValidationError` |
| `from scitex.io._bundle import BUNDLE_EXTENSIONS` | `from scitex.io.bundle import EXTENSIONS` |
| `from scitex.io._bundle import get_bundle_type` | `from scitex.io.bundle import get_type` |
| `from scitex.io._zip_bundle import ZipBundle` | `from scitex.io.bundle import ZipBundle` |
| `from scitex.io._zip_bundle import open_bundle` | `from scitex.io.bundle import open_zip` |
| `from scitex.io._zip_bundle import create_bundle` | `from scitex.io.bundle import create_zip` |
| `from scitex.io._zip_bundle import zip_directory_bundle` | `from scitex.io.bundle import zip_directory` |
| `from scitex.io._nested_bundle import resolve_nested_bundle` | `from scitex.io.bundle import nested; nested.resolve` |
| `from scitex.io._nested_bundle import get_nested_file` | `from scitex.io.bundle import nested; nested.get_file` |
| `from scitex.io._nested_bundle import get_nested_json` | `from scitex.io.bundle import nested; nested.get_json` |
| `from scitex.io._nested_bundle import get_nested_preview` | `from scitex.io.bundle import nested; nested.get_preview` |
| `from scitex.io._nested_bundle import put_nested_file` | `from scitex.io.bundle import nested; nested.put_file` |
| `from scitex.io._nested_bundle import put_nested_json` | `from scitex.io.bundle import nested; nested.put_json` |
| `from scitex.io._nested_bundle import list_nested_files` | `from scitex.io.bundle import nested; nested.list_files` |
| `from scitex.io._nested_bundle import parse_nested_path` | `from scitex.io.bundle import nested; nested.parse_path` |
| `from scitex.io._nested_bundle import NestedBundleNotFoundError` | `from scitex.io.bundle import NestedBundleNotFoundError` |

## Module Structure

```
scitex/io/bundle/
├── __init__.py      # Public API exports
├── _types.py        # BundleType, errors, constants
├── _core.py         # load, save, copy, pack, unpack, validate
├── _zip.py          # ZipBundle class and functions
├── _nested.py       # Nested bundle access
└── README.md        # This documentation
```

<!-- EOF -->