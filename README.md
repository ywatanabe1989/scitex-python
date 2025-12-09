<!-- ---
!-- Timestamp: 2025-12-09 04:00:22
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/README.md
!-- --- -->

# SciTeX

A Python framework for scientific research that makes the entire research pipeline more standardized, structured, and reproducible by automating repetitive processes.

Part of the fully open-source SciTeX project: https://scitex.ai

<!-- badges -->
[![PyPI version](https://badge.fury.io/py/scitex.svg)](https://badge.fury.io/py/scitex)
[![Python Versions](https://img.shields.io/pypi/pyversions/scitex.svg)](https://pypi.org/project/scitex/)
[![License](https://img.shields.io/github/license/ywatanabe1989/SciTeX-Code)](https://github.com/ywatanabe1989/SciTeX-Code/blob/main/LICENSE)
[![Tests](https://github.com/ywatanabe1989/SciTeX-Code/actions/workflows/ci.yml/badge.svg)](https://github.com/ywatanabe1989/SciTeX-Code/actions)
[![Coverage](https://codecov.io/gh/ywatanabe1989/SciTeX-Code/branch/main/graph/badge.svg)](https://codecov.io/gh/ywatanabe1989/SciTeX-Code)
[![Stats Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ywatanabe1989/GIST_ID/raw/scitex-stats-coverage.json)](https://github.com/ywatanabe1989/SciTeX-Code/actions/workflows/stats-coverage.yml)
[![Logging Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ywatanabe1989/GIST_ID/raw/scitex-logging-coverage.json)](https://github.com/ywatanabe1989/SciTeX-Code/actions/workflows/logging-coverage.yml)
[![Documentation](https://readthedocs.org/projects/scitex/badge/?version=latest)](https://scitex.readthedocs.io/en/latest/?badge=latest)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

## 📦 Installation

``` bash
pip install scitex # ~600 MB, Core + utilities
pip install scitex[dl,ml,jupyter,neuro,web,gui,scholar,writer,dev] # ~2-5 GB, Complete toolkit
```

### Alial

``` bash
# Ubuntu
sudo apt update
sudo apt-get install ttf-mscorefonts-installer
sudo DEBIAN_FRONTEND=noninteractive \
    apt install -y ttf-mscorefonts-installer
sudo mkdir -p /usr/share/fonts/truetype/custom
sudo cp /mnt/c/Windows/Fonts/arial*.ttf /usr/share/fonts/truetype/custom/
sudo fc-cache -fv
rm ~/.cache/matplotlib -rf

# WSL
mkdir -p ~/.local/share/fonts/windows
cp /mnt/c/Windows/Fonts/arial*.ttf ~/.local/share/fonts/windows/
fc-cache -fv ~/.local/share/fonts/windows
rm ~/.cache/matplotlib -rf
```

``` python
# Check
import matplotlib
print(matplotlib.rcParams['font.family'])

import matplotlib.font_manager as fm
fonts = fm.findSystemFonts()
print("Arial found:", any("Arial" in f or "arial" in f for f in fonts))
[a for a in fonts if "Arial" in a or "arial" in a][:5]

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]  # 念のため

fig, ax = plt.subplots(figsize=(3, 2))
ax.text(0.5, 0.5, "Arial Test", fontsize=32, ha="center", va="center")
ax.set_axis_off()

fig.savefig("arial_test.png", dpi=300)
plt.close(fig)
```

**Optional Groups**:

| Group       | Packages                                                | Size Impact |
|-------------|---------------------------------------------------------|-------------|
| **dl**      | PyTorch, transformers                                   | +2-4 GB     |
| **ml**      | scikit-image, catboost, optuna, OpenAI, Anthropic, Groq | ~200 MB     |
| **jupyter** | JupyterLab, papermill                                   | ~100 MB     |
| **neuro**   | MNE, obspy (EEG/MEG analysis)                           | ~200 MB     |
| **web**     | FastAPI, Flask, Streamlit                               | ~50 MB      |
| **gui**     | Flask, DearPyGui, PyQt6 (multi-backend figure editors)  | ~100 MB     |
| **scholar** | Selenium, PDF tools, paper management                   | ~150 MB     |
| **writer**  | LaTeX compilation tools                                 | ~10 MB      |
| **dev**     | Testing, linting (dev only)                             | ~100 MB     |

## 🚀 Quick Start


### The SciTeX Advantage: **70% Less Code**

Compare these two implementations that produce **identical research outputs**:

#### With SciTeX ([57 Lines of Code]((./examples/demo_session_plt_io.py)))

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-18 09:34:36 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_session_plt_io.py


"""Minimal Demonstration for scitex.{session,io,plt}"""

import numpy as np
import scitex as stx


def demo(filename, verbose=False):
    """Show metadata without QR code (just embedded)."""

    # matplotlib.pyplot wrapper.
    fig, ax = stx.plt.subplots()

    t = np.linspace(0, 2, 1000)
    signal = np.sin(2 * np.pi * 5 * t) * np.exp(-t / 2)

    ax.plot_line(t, signal)  # Original plot for automatic CSV export
    ax.set_xyt(
        "Time (s)",
        "Amplitude",
        "Clean Figure (metadata embedded, no QR overlay)",
    )

    # Saving: stx.io.save(obj, rel_path, **kwargs)
    stx.io.save(
        fig,
        filename,
        metadata={"exp": "s01", "subj": "S001"},  # with meatadata embedding
        symlink_to="./data",  # Symlink for centralized outputs
        verbose=verbose,  # Automatic terminal logging (no manual print())
    )
    fig.close()

    # Loading: stx.io.load(path)
    ldir = __file__.replace(".py", "_out")
    img, meta = stx.io.load(
        f"{ldir}/{filename}",
        verbose=verbose,
    )


@stx.session
def main(filename="demo.jpg", verbose=True):
    """Run demo for scitex.{session,plt,io}."""

    demo(filename, verbose=verbose)

    return 0


if __name__ == "__main__":
    main()
```

#### Without SciTeX ([188 Lines of Code](./examples/demo_session_plt_io_pure_python.py))
<details>
<summary>Click to see the pure Python equivalent requiring 3.3× more code</summary>
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-18 09:34:51 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_session_plt_io_pure_python.py


"""Minimal Demonstration - Pure Python Version"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
import random
import string

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def generate_session_id():
    """Generate unique session ID."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"{timestamp}_{random_suffix}"


def setup_logging(log_dir):
    """Set up logging infrastructure."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    stdout_handler = logging.FileHandler(log_dir / "stdout.log")
    stderr_handler = logging.FileHandler(log_dir / "stderr.log")
    console_handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_plot_data_to_csv(fig, output_path):
    """Extract and save plot data."""
    csv_path = output_path.with_suffix('.csv')
    data_lines = ["ax_00_plot_line_0_line_x,ax_00_plot_line_0_line_y"]
    
    for ax in fig.get_axes():
        for line in ax.get_lines():
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            for x, y in zip(x_data, y_data):
                data_lines.append(f"{x},{y}")
    
    csv_path.write_text('\n'.join(data_lines))
    return csv_path, csv_path.stat().st_size / 1024


def embed_metadata_in_image(image_path, metadata):
    """Embed metadata into image file."""
    img = Image.open(image_path)
    
    if image_path.suffix.lower() in ['.png']:
        pnginfo = PngInfo()
        for key, value in metadata.items():
            pnginfo.add_text(key, str(value))
        img.save(image_path, pnginfo=pnginfo)
    elif image_path.suffix.lower() in ['.jpg', '.jpeg']:
        json_path = image_path.with_suffix(image_path.suffix + '.meta.json')
        json_path.write_text(json.dumps(metadata, indent=2))
        img.save(image_path, quality=95)


def save_figure(fig, output_path, metadata=None, symlink_to=None, logger=None):
    """Save figure with metadata and symlink."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if metadata is None:
        metadata = {}
    metadata['url'] = 'https://scitex.ai'
    
    if logger:
        logger.info(f"📝 Saving figure with metadata to: {output_path}")
        logger.info(f"  • Embedded metadata: {metadata}")
    
    csv_path, csv_size = save_plot_data_to_csv(fig, output_path)
    if logger:
        logger.info(f"✅ Saved to: {csv_path} ({csv_size:.1f} KiB)")
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    embed_metadata_in_image(output_path, metadata)
    
    if symlink_to:
        symlink_dir = Path(symlink_to)
        symlink_dir.mkdir(parents=True, exist_ok=True)
        symlink_path = symlink_dir / output_path.name
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(output_path.resolve())


def demo(output_dir, filename, verbose=False, logger=None):
    """Generate, plot, and save signal."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    t = np.linspace(0, 2, 1000)
    signal = np.sin(2 * np.pi * 5 * t) * np.exp(-t / 2)
    
    ax.plot(t, signal)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Damped Oscillation")
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / filename
    save_figure(fig, output_path, metadata={"exp": "s01", "subj": "S001"},
                symlink_to=output_dir.parent / "data", logger=logger)
    plt.close(fig)
    
    return 0


def main():
    """Run demo - Pure Python Version."""
    parser = argparse.ArgumentParser(description="Run demo - Pure Python Version")
    parser.add_argument('-f', '--filename', default='demo.jpg')
    parser.add_argument('-v', '--verbose', type=bool, default=True)
    args = parser.parse_args()
    
    session_id = generate_session_id()
    script_path = Path(__file__).resolve()
    output_base = script_path.parent / (script_path.stem + "_out")
    running_dir = output_base / "RUNNING" / session_id
    logs_dir = running_dir / "logs"
    config_dir = running_dir / "CONFIGS"
    
    logger = setup_logging(logs_dir)
    
    print("=" * 40)
    print(f"Pure Python Demo")
    print(f"{session_id} (PID: {os.getpid()})")
    print(f"\n{script_path}")
    print(f"\nArguments:")
    print(f"    filename: {args.filename}")
    print(f"    verbose: {args.verbose}")
    print("=" * 40)
    
    config_dir.mkdir(parents=True, exist_ok=True)
    config_data = {
        'ID': session_id,
        'FILE': str(script_path),
        'SDIR_OUT': str(output_base),
        'SDIR_RUN': str(running_dir),
        'PID': os.getpid(),
        'ARGS': vars(args)
    }
    (config_dir / "CONFIG.json").write_text(json.dumps(config_data, indent=2))
    
    try:
        result = demo(output_base, args.filename, args.verbose, logger)
        success_dir = output_base / "FINISHED_SUCCESS" / session_id
        success_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(running_dir), str(success_dir))
        logger.info(f"\n✅ Script completed: {success_dir}")
        return result
    except Exception as e:
        error_dir = output_base / "FINISHED_ERROR" / session_id
        error_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(running_dir), str(error_dir))
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
```

</details>

### What You Get With `@stx.session`

Both implementations produce **identical outputs**, but SciTeX eliminates 131 lines of boilerplate:
```bash
demo_session_plt_io_out/
├── demo.csv              # Auto-extracted plot data
├── demo.jpg              # With embedded metadata
└── FINISHED_SUCCESS/
    └── 2025Y-11M-18D-09h12m03s_HmH5-main/
        ├── CONFIGS/
        │   ├── CONFIG.pkl    # Python object
        │   └── CONFIG.yaml   # Human-readable
        └── logs/
            ├── stderr.log
            └── stdout.log
```

**What SciTeX Automates:**
- ✅ Session ID generation and tracking
- ✅ Output directory management (`RUNNING/` → `FINISHED_SUCCESS/`)
- ✅ Argument parsing with auto-generated help
- ✅ Logging to files and console
- ✅ Config serialization (YAML + pickle)
- ✅ CSV export from matplotlib plots
- ✅ Metadata embedding in images
- ✅ Symlink management for centralized outputs
- ✅ Error handling and directory cleanup
- ✅ Global variable injection (CONFIG, plt, COLORS, logger, rng_manager)

**Research Benefits:**
- 📊 **Figures + data always together** - CSV auto-exported from every plot
- 🔄 **Perfect reproducibility** - Every run tracked with unique session ID
- 🌍 **Universal format** - CSV data readable anywhere
- 📝 **Zero manual work** - Metadata embedded automatically
- 🎯 **3.3× less code** - Focus on research, not infrastructure

### Try It Yourself
```bash
pip install scitex
python ./examples/demo_session_plt_io.py
```

## 📦 Module Overview

SciTeX is organized into focused modules for different aspects of scientific computing:

### 🔧 Core Utilities
| Module                                          | Description                                                         |
|-------------------------------------------------|---------------------------------------------------------------------|
| [`scitex.gen`](./src/scitex/gen#readme)         | Project setup, session management, and experiment tracking          |
| [`scitex.io`](./src/scitex/io#readme)           | Universal I/O for 30+ formats (CSV, JSON, HDF5, Zarr, pickle, etc.) |
| [`scitex.path`](./src/scitex/path#readme)       | Path manipulation and project structure utilities                   |
| [`scitex.logging`](./src/scitex/logging#readme) | Structured logging with color support and context                   |

### 📊 Data Science & Statistics
| Module                                      | Description                                                              |
|---------------------------------------------|--------------------------------------------------------------------------|
| [`scitex.stats`](./src/scitex/stats#readme) | 16 statistical tests, effect sizes, power analysis, multiple corrections |
| [`scitex.plt`](./src/scitex/plt#readme)     | Enhanced matplotlib with auto-export and scientific captions             |
| [`scitex.pd`](./src/scitex/pd#readme)       | Pandas extensions for research workflows                                 |

### 🧠 AI & Machine Learning
| Module                                      | Description                                             |
|---------------------------------------------|---------------------------------------------------------|
| [`scitex.ai`](./src/scitex/ai#readme)       | GenAI (7 providers), classification, training utilities |
| [`scitex.torch`](./src/scitex/torch#readme) | PyTorch training loops, metrics, and utilities          |
| [`scitex.nn`](./src/scitex/nn#readme)       | Custom neural network layers                            |

### 🌊 Signal Processing
| Module                                  | Description                                                   |
|-----------------------------------------|---------------------------------------------------------------|
| [`scitex.dsp`](./src/scitex/dsp#readme) | Filtering, spectral analysis, wavelets, PAC, ripple detection |

### 📚 Literature Management
| Module                                          | Description                                                     |
|-------------------------------------------------|-----------------------------------------------------------------|
| [`scitex.scholar`](./src/scitex/scholar#readme) | Paper search, PDF download, BibTeX enrichment with IF/citations |

### 🌐 Web & Browser
| Module                                          | Description                                                |
|-------------------------------------------------|------------------------------------------------------------|
| [`scitex.browser`](./src/scitex/browser#readme) | Playwright automation with debugging, PDF handling, popups |

### 🗄️ Data Management
| Module                                | Description                         |
|---------------------------------------|-------------------------------------|
| [`scitex.db`](./src/scitex/db#readme) | SQLite3 and PostgreSQL abstractions |

### 🛠️ Utilities
| Module                                                | Description                                         |
|-------------------------------------------------------|-----------------------------------------------------|
| [`scitex.decorators`](./src/scitex/decorators#readme) | Function decorators for caching, timing, validation |
| [`scitex.rng`](./src/scitex/rng#readme)               | Reproducible random number generation               |
| [`scitex.resource`](./src/scitex/resource#readme)     | System resource monitoring (CPU, memory, GPU)       |
| [`scitex.dict`](./src/scitex/dict#readme)             | Dictionary manipulation and nested access           |
| [`scitex.str`](./src/scitex/str#readme)               | String utilities for scientific text processing     |

## 📖 Documentation

### Online Documentation
- **[Read the Docs](https://scitex.readthedocs.io/)**: Complete API reference and guides
- **[Interactive Examples](https://scitex.readthedocs.io/en/latest/examples/index.html)**: Browse all tutorial notebooks
- **[Quick Start Guide](https://scitex.readthedocs.io/en/latest/getting_started.html)**: Get up and running quickly

### Local Resources
- **[Master Tutorial Index](./examples/00_SCITEX_MASTER_INDEX.ipynb)**: Comprehensive guide to all features
- **[Examples Directory](./examples/)**: 25+ Jupyter notebooks covering all modules
- **[Module List](./docs/scitex_modules.csv)**: Complete list of all functions
- **(Experimental) [MCP Servers Documentation](./mcp_servers/README.md)**

### Key Tutorials
1. **[I/O Operations](./examples/01_scitex_io.ipynb)**: Essential file handling (start here!)
2. **[Plotting](./examples/14_scitex_plt.ipynb)**: Publication-ready visualizations
3. **[Statistics](./examples/11_scitex_stats.ipynb)**: Research-grade statistical analysis
4. **[Scholar](./examples/16_scitex_scholar.ipynb)**: Literature management with impact factors
5. **[AI/ML](./examples/16_scitex_ai.ipynb)**: Complete machine learning toolkit

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.


## 📄 License

AGPL-3.0.

## 📧 Contact

Yusuke Watanabe (ywatanabe@scitex.ai)

<!-- EOF -->