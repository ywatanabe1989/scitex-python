<!-- ---
!-- Timestamp: 2025-11-14 09:13:14
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
pip install scitex[dl,ml,jupyter,neuro,web,scholar,writer,dev] # ~2-5 GB, Complete toolkit
```

**Optional Groups**:

| Group       | Packages                                                | Size Impact |
|-------------|---------------------------------------------------------|-------------|
| **dl**      | PyTorch, transformers                                   | +2-4 GB     |
| **ml**      | scikit-image, catboost, optuna, OpenAI, Anthropic, Groq | ~200 MB     |
| **jupyter** | JupyterLab, papermill                                   | ~100 MB     |
| **neuro**   | MNE, obspy (EEG/MEG analysis)                           | ~200 MB     |
| **web**     | FastAPI, Flask, Streamlit                               | ~50 MB      |
| **scholar** | Selenium, PDF tools, paper management                   | ~150 MB     |
| **writer**  | LaTeX compilation tools                                 | ~10 MB      |
| **dev**     | Testing, linting (dev only)                             | ~100 MB     |

## 🚀 Quick Start

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-14 09:09:37 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_session_plt_io.py


"""Minimal Demonstration for scitex.{session,io,plt}"""

import numpy as np
import scitex as stx


def demo_without_qr(filename, verbose=False):
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


@stx.session.session
def main(filename="demo_fig_with_metadata.jpg", verbose=True):
    """Run all demos."""

    demo_without_qr(filename, verbose=verbose)

    return 0


if __name__ == "__main__":
    main()

# (.env-3.11) (wsl) scitex-code $ ./examples/demo_session_plt_io.py -h
# usage: demo_session_plt_io.py [-h] [--filename FILENAME] [--verbose VERBOSE]

# Run all demos.

# options:
#   -h, --help           show this help message and exit
#   --filename FILENAME  (default: demo_fig_with_metadata.jpg)
#   --verbose VERBOSE    (default: True)
# (.env-3.11) (wsl) scitex-code $ ./examples/demo_session_plt_io.py

# ========================================
# SciTeX v2.1.3
# 2025Y-11M-14D-08h56m55s_JDUS (PID: 2374042)

# /home/ywatanabe/proj/scitex-code/./examples/demo_session_plt_io.py

# Arguments:
#     filename: demo_fig_with_metadata.jpg
#     verbose: True
# ========================================

# INFO: Running main with args: {'filename': 'demo_fig_with_metadata.jpg', 'verbose': True}
# (.env-3.11) (wsl) scitex-code $ ./examples/demo_session_plt_io.py -h
# usage: demo_session_plt_io.py [-h] [--filename FILENAME] [--verbose VERBOSE]

# Run all demos.

# options:
#   -h, --help           show this help message and exit
#   --filename FILENAME  (default: demo_fig_with_metadata.jpg)
#   --verbose VERBOSE    (default: True)
# (.env-3.11) (wsl) scitex-code $ ./examples/demo_session_plt_io.py

# ========================================
# SciTeX v2.1.3
# 2025Y-11M-14D-09h08m33s_2DKi (PID: 2396675)

# /home/ywatanabe/proj/scitex-code/./examples/demo_session_plt_io.py

# Arguments:
#     filename: demo_fig_with_metadata.jpg
#     verbose: True
# ========================================

# INFO: Running main with args: {'filename': 'demo_fig_with_metadata.jpg', 'verbose': True}
# INFO: 📝 Saving figure with metadata to: /home/ywatanabe/proj/scitex-code/examples/demo_session_plt_io_out/demo_fig_with_metadata.jpg
# INFO:   • Auto-added URL: https://scitex.ai
# INFO:   • Embedded metadata: {'exp': 's01', 'subj': 'S001', 'url': 'https://scitex.ai'}
# SUCC: Saved to: ./examples/demo_session_plt_io_out/demo_fig_with_metadata.csv (38.0 KiB)
# SUCC: Symlinked: /home/ywatanabe/proj/scitex-code/examples/demo_session_plt_io_out/demo_fig_with_metadata.csv -> /home/ywatanabe/proj/scitex-code/examples/demo_session_plt_io_out/demo_fig_with_metadata.csv/demo_fig_with_metadata.csv
# SUCC: Saved to: ./examples/demo_session_plt_io_out/demo_fig_with_metadata.jpg (241.6 KiB)
# SUCC: Symlinked: /home/ywatanabe/proj/scitex-code/examples/demo_session_plt_io_out/demo_fig_with_metadata.jpg -> /home/ywatanabe/proj/scitex-code/data/demo_fig_with_metadata.jpg
# INFO: ✅ Loading image with metadata from: /home/ywatanabe/proj/scitex-code/./examples/demo_session_plt_io_out/demo_fig_with_metadata.jpg
# INFO:   • Embedded metadata found:
# INFO:     - exp: s01
# INFO:     - subj: S001
# INFO:     - url: https://scitex.ai

# SUCC: Congratulations! The script completed: /home/ywatanabe/proj/scitex-code/examples/demo_session_plt_io_out/FINISHED_SUCCESS/2025Y-11M-14D-09h08m33s_2DKi-main/
# (.env-3.11) (wsl) scitex-code $ ls -al ./data/demo_fig_with_metadata.jpg
# lrwxrwxrwx 1 ywatanabe ywatanabe 62 Nov 14 09:08 ./data/demo_fig_with_metadata.jpg -> ../examples/demo_session_plt_io_out/demo_fig_with_metadata.jpg
# (.env-3.11) (wsl) scitex-code $ tree ./examples/demo_session_plt_io*
# ./examples/demo_session_plt_io_out
# ├── demo_fig_with_metadata.csv
# ├── demo_fig_with_metadata.jpg
# ├── FINISHED_SUCCESS
# │   ├── 2025Y-11M-14D-09h07m28s_j5gY-main
# │   │   ├── CONFIGS
# │   │   │   ├── CONFIG.pkl
# │   │   │   └── CONFIG.yaml
# │   │   └── logs
# │   │       ├── stderr.log
# │   │       └── stdout.log
# │   └── 2025Y-11M-14D-09h08m33s_2DKi-main
# │       ├── CONFIGS
# │       │   ├── CONFIG.pkl
# │       │   └── CONFIG.yaml
# │       └── logs
# │           ├── stderr.log
# │           └── stdout.log
# └── RUNNING
#     └── 2025Y-11M-14D-08h56m37s_jPQu-main
#         └── logs
#             ├── stderr.log
#             └── stdout.log
# ./examples/demo_session_plt_io.py  [error opening dir]

# 11 directories, 13 files
# (.env-3.11) (wsl) scitex-code $ head ./examples/demo_session_plt_io_out/demo_fig_with_metadata.csv
# ax_00_plot_line_0_line_x,ax_00_plot_line_0_line_y
# 0.0,0.0
# 0.06279040531731951,0.002002002002002002
# 0.12520711420365782,0.004004004004004004
# 0.18700423504710992,0.006006006006006006
# 0.24793881631482095,0.008008008008008008
# 0.30777180069339666,0.01001001001001001
# 0.3662689619330633,0.012012012012012012
# 0.4232018207282413,0.014014014014014014
# 0.4783485360573639,0.016016016016016016

# EOF
```

**Benefits:**
- 📊 Figures + data always together
- 🔄 Perfect reproducibility and traceability
- 🌍 Universal CSV format
- 📝 No manual export needed

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

This project is licensed under the MIT License.

## 📧 Contact

Yusuke Watanabe (ywatanabe@scitex.ai)

<!-- EOF -->