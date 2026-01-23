#!/usr/bin/env python3
# Timestamp: 2026-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_resources/_formats.py
"""IO format documentation resource."""

from __future__ import annotations

__all__ = ["register_format_resources"]

IO_FORMATS = """\
# stx.io Supported Formats
===========================

## Data Formats

| Extension | Type | Save | Load | Notes |
|-----------|------|:----:|:----:|-------|
| .csv | DataFrame/Array | ✓ | ✓ | Comma-separated |
| .tsv | DataFrame/Array | ✓ | ✓ | Tab-separated |
| .xlsx | DataFrame | ✓ | ✓ | Excel workbook |
| .json | Dict/List | ✓ | ✓ | JSON format |
| .yaml/.yml | Dict/List | ✓ | ✓ | YAML format |
| .pkl/.pickle | Any | ✓ | ✓ | Python pickle |
| .npy | Array | ✓ | ✓ | NumPy array |
| .npz | Dict[Array] | ✓ | ✓ | Compressed NumPy |
| .h5/.hdf5 | Array/Dict | ✓ | ✓ | HDF5 format |
| .zarr | Array | ✓ | ✓ | Zarr array |
| .parquet | DataFrame | ✓ | ✓ | Apache Parquet |
| .feather | DataFrame | ✓ | ✓ | Feather format |

## Image Formats

| Extension | Type | Save | Load | Notes |
|-----------|------|:----:|:----:|-------|
| .png | Figure/Array | ✓ | ✓ | Lossless, metadata |
| .jpg/.jpeg | Figure/Array | ✓ | ✓ | Lossy, metadata |
| .tiff/.tif | Figure/Array | ✓ | ✓ | Scientific imaging |
| .pdf | Figure | ✓ | ✓ | Vector format |
| .svg | Figure | ✓ | - | Vector format |

## Scientific Formats

| Extension | Type | Load | Notes |
|-----------|------|:----:|-------|
| .edf | EEG | ✓ | European Data Format |
| .fif | MNE | ✓ | MNE-Python |
| .set | EEGLAB | ✓ | EEGLAB format |
| .mat | MATLAB | ✓ | MATLAB files |

## PyTorch Formats

| Extension | Type | Save | Load |
|-----------|------|:----:|:----:|
| .pt | Tensor/Model | ✓ | ✓ |
| .pth | Model | ✓ | ✓ |

## Usage Notes

1. **Extension determines handler**: `stx.io.save(df, "data.csv")` uses CSV
2. **Metadata embedding**: PNG/JPEG support embedded metadata
3. **Figure auto-CSV**: Saving figures also exports plotted data as CSV
4. **Symlink support**: `symlink_to="./data"` creates symlinks

## Examples

```python
import scitex as stx
import pandas as pd
import numpy as np

# DataFrame
df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
stx.io.save(df, "data.csv")
stx.io.save(df, "data.parquet")

# NumPy
arr = np.random.randn(100, 100)
stx.io.save(arr, "array.npy")
stx.io.save(arr, "array.npz")

# Any Python object
stx.io.save({"config": "value"}, "config.yaml")
stx.io.save(complex_obj, "obj.pkl")

# Figure with auto-CSV
fig, ax = stx.plt.subplots()
ax.stx_line([1, 2, 3], [4, 5, 6])
stx.io.save(fig, "plot.png")  # Creates plot.png + plot.csv
```
"""


def register_format_resources(mcp) -> None:
    """Register IO formats resource."""

    @mcp.resource("scitex://io-formats")
    def io_formats() -> str:
        """List all supported file formats for stx.io."""
        return IO_FORMATS


# EOF
