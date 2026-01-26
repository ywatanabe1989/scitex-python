#!/usr/bin/env python3
"""I/O operations script template."""

TEMPLATE = {
    "name": "I/O Operations Script",
    "description": "Demonstrates stx.io.save() and stx.io.load() for 30+ file formats",
    "filename": "io_script.py",
    "usage": """
Usage:
  python io_script.py

Supported Formats (30+):
  Data:    csv, json, yaml, pkl, npy, npz, mat, hdf5, parquet, feather
  Figures: png, jpg, svg, pdf (with metadata embedding)
  Text:    txt, md, html
  Config:  yaml, toml, ini
""",
    "content": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "{timestamp}"
# File: {filepath}

"""
{docstring}

stx.io Usage Guide
------------------
stx.io.save(obj, path, **kwargs)  # Universal save
stx.io.load(path)                  # Universal load

Supported Formats (30+):
  Data:    csv, json, yaml, pkl, npy, npz, mat, hdf5, parquet, feather
  Figures: png, jpg, svg, pdf (with metadata embedding)
  Text:    txt, md, html
  Config:  yaml, toml, ini

Key Features:
  - Auto-format detection from extension
  - Metadata embedding in images
  - Symlink support for centralized outputs
  - Automatic CSV export for figures
"""

import numpy as np
import pandas as pd
import scitex as stx


@stx.session
def main(
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Demonstrate stx.io save/load operations."""

    # === DataFrame ===
    df = pd.DataFrame({{"x": [1, 2, 3], "y": [4, 5, 6]}})
    stx.io.save(df, "data.csv")       # -> CONFIG.SDIR_OUT/data.csv
    stx.io.save(df, "data.json")
    stx.io.save(df, "data.parquet")

    # === NumPy Arrays ===
    arr = np.random.randn(100, 10)
    stx.io.save(arr, "array.npy")
    stx.io.save(arr, "array.csv")

    # === Dictionaries ===
    config = {{"param1": 100, "param2": "value"}}
    stx.io.save(config, "config.yaml")
    stx.io.save(config, "config.json")

    # === Figures with Metadata ===
    fig, ax = plt.subplots()
    ax.plot_line([1, 2, 3], [1, 4, 9])
    ax.set_xyt("X", "Y", "Example")

    stx.io.save(
        fig,
        "figure.png",
        metadata={{"experiment": "demo"}},  # Embedded in image
        symlink_to="./data",                # Create symlink
        verbose=True,
    )
    # Auto-exports: figure.csv (plotted data)

    # === Loading ===
    df_loaded = stx.io.load(f"{{CONFIG.SDIR_OUT}}/data.csv")
    arr_loaded = stx.io.load(f"{{CONFIG.SDIR_OUT}}/array.npy")
    img, meta = stx.io.load(f"{{CONFIG.SDIR_OUT}}/figure.png")

    logger.info(f"Output directory: {{CONFIG.SDIR_OUT}}")
    logger.info(f"Loaded DataFrame shape: {{df_loaded.shape}}")
    logger.info(f"Loaded array shape: {{arr_loaded.shape}}")
    logger.info(f"Image metadata: {{meta}}")

    return 0


if __name__ == "__main__":
    main()

# EOF
''',
}

# EOF
