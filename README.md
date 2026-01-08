<!-- ---
!-- Timestamp: 2026-01-08 11:41:42
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/README.md
!-- --- -->

# SciTeX

A Python framework for scientific research that makes the entire research pipeline more standardized, structured, and reproducible by automating repetitive processes.

Part of the fully open-source SciTeX project: https://scitex.ai

## 📊 Status

<!-- badges -->
<div align="center">
  <table>
    <tr>
      <td align="center"><strong>Package</strong></td>
      <td>
        <a href="https://badge.fury.io/py/scitex"><img src="https://badge.fury.io/py/scitex.svg" alt="PyPI version"></a>
        <a href="https://pypi.org/project/scitex/"><img src="https://img.shields.io/pypi/pyversions/scitex.svg" alt="Python Versions"></a>
        <a href="https://github.com/ywatanabe1989/SciTeX-Code/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ywatanabe1989/SciTeX-Code" alt="License"></a>
      </td>
    </tr>
    <tr>
      <td align="center"><strong>Install</strong></td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/core.json&label=core" alt="Core Install Time">
        <img src="https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/all.json&label=all" alt="All Install Time">
        <img src="https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/heavy.json&label=heavy" alt="Heavy Install Time">
        <img src="https://img.shields.io/badge/uv-recommended-blue" alt="uv recommended">
      </td>
    </tr>
  </table>
</div>

<details>
<summary><strong>Module Status - Tests & Installation</strong></summary>

### Core Modules

| Module | Description | Install (Time) | Tests |
|--------|-------------|----------------|-------|
| [`io`](./src/scitex/io#readme) | Universal I/O (30+ formats) | `scitex[io]` ![io](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/io.json) | [![io](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-io.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-io.yml) |
| [`path`](./src/scitex/path#readme) | Path utilities | `scitex[path]` ![path](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/path.json) | [![path](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-path.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-path.yml) |
| [`str`](./src/scitex/str#readme) | String processing | `scitex[str]` ![str](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/str.json) | [![str](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-str.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-str.yml) |
| [`dict`](./src/scitex/dict#readme) | Dictionary utilities | core ![dict](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/dict.json) | [![dict](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dict.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dict.yml) |
| [`types`](./src/scitex/types#readme) | Type checking | core ![types](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/types.json) | [![types](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-types.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-types.yml) |
| [`config`](./src/scitex/config#readme) | Configuration management | `scitex[config]` ![config](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/config.json) | [![config](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-config.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-config.yml) |
| [`utils`](./src/scitex/utils#readme) | General utilities | `scitex[utils]` ![utils](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/utils.json) | [![utils](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-utils.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-utils.yml) |
| [`decorators`](./src/scitex/decorators#readme) | Function decorators | `scitex[decorators]` ![decorators](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/decorators.json) | [![decorators](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-decorators.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-decorators.yml) |
| [`logging`](./src/scitex/logging#readme) | Structured logging | `scitex[logging]` ![logging](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/logging.json) | [![logging](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-logging.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-logging.yml) |
| [`gen`](./src/scitex/gen#readme) | Project setup | `scitex[gen]` ![gen](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/gen.json) | [![gen](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-gen.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-gen.yml) |

### Data Science & Statistics

| Module | Description | Install (Time) | Tests |
|--------|-------------|----------------|-------|
| [`stats`](./src/scitex/stats#readme) | Statistical tests & analysis | `scitex[stats]` ![stats](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/stats.json) | [![stats](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-stats.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-stats.yml) |
| [`pd`](./src/scitex/pd#readme) | Pandas extensions | `scitex[pd]` ![pd](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/pd.json) | [![pd](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-pd.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-pd.yml) |
| [`linalg`](./src/scitex/linalg#readme) | Linear algebra | `scitex[linalg]` ![linalg](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/linalg.json) | [![linalg](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-linalg.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-linalg.yml) |
| [`plt`](./src/scitex/plt#readme) | Enhanced matplotlib | `scitex[plt]` ![plt](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/plt.json) | [![plt](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-plt.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-plt.yml) |
| [`dsp`](./src/scitex/dsp#readme) | Signal processing | `scitex[dsp]` ![dsp](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/dsp.json) | [![dsp](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dsp.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dsp.yml) |

### AI & Machine Learning

| Module | Description | Install (Time) | Tests |
|--------|-------------|----------------|-------|
| [`ai`](./src/scitex/ai#readme) | GenAI (7 providers) | `scitex[ai]` ![ai](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/ai.json) | [![ai](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-ai.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-ai.yml) |
| [`nn`](./src/scitex/nn#readme) | Neural network layers | `scitex[nn]` ![nn](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/nn.json) | [![nn](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-nn.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-nn.yml) |
| [`torch`](./src/scitex/torch#readme) | PyTorch utilities | `scitex[torch]` ![torch](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/torch.json) | [![torch](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-torch.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-torch.yml) |

### System & Tools

| Module | Description | Install (Time) | Tests |
|--------|-------------|----------------|-------|
| [`cli`](./src/scitex/cli#readme) | Command-line tools | `scitex[cli]` ![cli](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/cli.json) | [![cli](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-cli.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-cli.yml) |
| [`sh`](./src/scitex/sh#readme) | Shell utilities | `scitex[sh]` ![sh](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/sh.json) | [![sh](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-sh.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-sh.yml) |
| [`git`](./src/scitex/git#readme) | Git operations | `scitex[git]` ![git](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/git.json) | [![git](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-git.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-git.yml) |
| [`session`](./src/scitex/session#readme) | Session management | `scitex[session]` ![session](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/session.json) | [![session](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-session.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-session.yml) |
| [`resource`](./src/scitex/resource#readme) | System monitoring | `scitex[resource]` ![resource](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/resource.json) | [![resource](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-resource.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-resource.yml) |
| [`db`](./src/scitex/db#readme) | Database abstractions | `scitex[db]` ![db](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/db.json) | [![db](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-db.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-db.yml) |

### Research & Publishing

| Module | Description | Install (Time) | Tests |
|--------|-------------|----------------|-------|
| [`writer`](./src/scitex/writer#readme) | Document generation | `scitex[writer]` ![writer](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/writer.json) | [![writer](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-writer.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-writer.yml) |
| [`tex`](./src/scitex/tex#readme) | LaTeX processing | `scitex[tex]` ![tex](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/tex.json) | [![tex](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-tex.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-tex.yml) |
| [`msword`](./src/scitex/msword#readme) | MS Word conversion | `scitex[msword]` ![msword](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/msword.json) | [![msword](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-msword.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-msword.yml) |
| [`scholar`](./src/scitex/scholar#readme) | Literature management | `scitex[scholar]` ![scholar](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/scholar.json) | [![scholar](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-scholar.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-scholar.yml) |
| [`diagram`](./src/scitex/diagram#readme) | Diagram generation | `scitex[diagram]` ![diagram](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/diagram.json) | [![diagram](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-diagram.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-diagram.yml) |

### Web & Automation

| Module | Description | Install (Time) | Tests |
|--------|-------------|----------------|-------|
| [`web`](./src/scitex/web#readme) | Web scraping | `scitex[web]` ![web](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/web.json) | [![web](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-web.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-web.yml) |
| [`browser`](./src/scitex/browser#readme) | Browser automation | `scitex[browser]` ![browser](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/browser.json) | [![browser](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-browser.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-browser.yml) |

### Other Modules

| Module | Description | Install (Time) | Tests |
|--------|-------------|----------------|-------|
| [`audio`](./src/scitex/audio#readme) | Audio processing | `scitex[audio]` ![audio](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/audio.json) | [![audio](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-audio.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-audio.yml) |
| [`capture`](./src/scitex/capture#readme) | Screen capture | `scitex[capture]` ![capture](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/capture.json) | [![capture](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-capture.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-capture.yml) |
| [`repro`](./src/scitex/repro#readme) | Reproducibility | `scitex[repro]` ![repro](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/repro.json) | [![repro](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-repro.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-repro.yml) |
| [`benchmark`](./src/scitex/benchmark#readme) | Performance testing | `scitex[benchmark]` ![benchmark](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/benchmark.json) | [![benchmark](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-benchmark.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-benchmark.yml) |
| [`security`](./src/scitex/security#readme) | Security utilities | `scitex[security]` ![security](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/security.json) | [![security](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-security.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-security.yml) |
| [`dt`](./src/scitex/dt#readme) | Datetime utilities | `scitex[dt]` ![dt](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/dt.json) | [![dt](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dt.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dt.yml) |
| [`dev`](./src/scitex/dev#readme) | Development tools | `scitex[dev]` ![dev](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/dev.json) | [![dev](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dev.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dev.yml) |
| [`schema`](./src/scitex/schema#readme) | Data schemas | `scitex[schema]` ![schema](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/schema.json) | [![schema](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-schema.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-schema.yml) |
| [`bridge`](./src/scitex/bridge#readme) | Module integration | `scitex[bridge]` ![bridge](https://img.shields.io/endpoint?url=https://ywatanabe1989.github.io/scitex-code/badges/bridge.json) | [![bridge](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-bridge.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-bridge.yml) |

</details>

## 📦 Installation

``` bash
uv pip install scitex[all]     # Recommended: Full installation with all modules
uv pip install scitex          # Core only (numpy, pandas, PyYAML, tqdm)
uv pip install scitex[heavy]   # Include heavy deps (torch, mne, optuna, etc.)
```

> **Note**: Heavy dependencies (torch, mne, optuna, catboost, jax, tensorflow, umap-learn)
> are optional and NOT included by default. Install with `scitex[heavy]` if needed.
> Modules gracefully handle missing dependencies with `*_AVAILABLE` flags.

<details>
<summary><strong>Arial Font Setup for Figures</strong></summary>

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

</details>

## Why SciTeX?

SciTeX automates research analysis code.

<!-- **What SciTeX Automates:**
 !-- - ✅ Symlink management for centralized outputs
 !-- - ✅ Error handling and directory cleanup
 !-- - ✅ Global variable injection (CONFIG, plt, COLORS, logger, rng)
 !-- 
 !-- **Research Benefits:**
 !-- - 📊 **Figures + data always together** - CSV auto-exported from every plot
 !-- - 🔄 **Perfect reproducibility** - Every run tracked with unique session ID
 !-- - 🌍 **Universal format** - CSV data readable anywhere
 !-- - 📝 **Zero manual work** - Metadata embedded automatically
 !-- - 🎯 **3.3× less code** - Focus on research, not infrastructure -->


<details>
<summary><strong><code>@scitex.session</code></strong> — Reproducible Experiment Tracking + Auto-CLI</summary>

Standardized outputs with automatic logging. Scripts and outputs closely linked for full traceability.

**Auto-CLI Generation**: Function arguments automatically become argparse options:

```python
# /path/to/script.py
import scitex as stx

@stx.session
def main(
    # arg1,                     # Required: -a ARG1, --arg1 ARG1
    # kwarg1="value1",          # Optional: -k KWARG1, --kwarg1 KWARG1 (default: value1)
    CONFIG=stx.INJECTED,        # Auto-injected from $(pwd)/config/*.yaml files
    plt=stx.INJECTED,           # Pre-configured matplotlib
    COLORS=stx.INJECTED,        # Color palette
    rng=stx.INJECTED,           # Seeded random generator
    logger=stx.INJECTED,        # Session logger
):
    """This docstring becomes --help description."""
    
    stx.io.save(results, "results.csv", symlink_to="./data/")
    # SUCC: Saved to: /path/to/script_out/tmp.txt (4.0 B)
    # SUCC: Symlinked: /path/to/script_out/tmp.txt ->
    # SUCC:            ./data/tmp.txt
    
    return 0
```

```bash
$ python script.py --help
usage: script.py [-a ARG1] [-k KWARG1]
  -a ARG1, --arg1 ARG1     (required)
  -k KWARG1, --kwarg1 KWARG1  (default: value1)

$ python script.py -a myvalue
# Runs with arg1="myvalue", kwarg1="value1"
```

**Output Structure**:
```bash
/path/to/script.py
/path/to/script_out/
├── FINISHED_SUCCESS/
│    └── 2025-01-08_12-30-00_AbC1/ # Session ID allocated
│        ├── results.csv
│        ├── CONFIGS/
│        │   ├── CONFIG.pkl    # Python object
│        │   └── CONFIG.yaml   # Human-readable
│        └── logs/
│            ├── stderr.log    # Standard Errors
│            └── stdout.log    # Standard Outputs
├── FINISHED_FAILED/
│    └── ...
└── RUNNING/
     └── ...
```

</details>

<details>
<summary><strong><code>scitex.plt</code></strong> — Auto-Export Figures with Data</summary>

Figures and data always together. CSV auto-exported from every plot.

```python
import scitex as stx

fig, ax = stx.plt.subplots()
ax.plot_line(x, y)  # Data tracked automatically
stx.io.save(fig, "plot.png")
# Creates: plot.png + plot.csv
```

</details>

<details>
<summary><strong><code>scitex.io</code></strong> — Universal File I/O (30+ formats)</summary>

One API for all formats: CSV, JSON, YAML, pickle, NumPy, HDF5, Zarr, PyTorch, images, PDFs, EEG formats.

```python
import scitex as stx

# Unified API Call for +30 formats
stx.io.save(df, "output.csv")
stx.io.save(arr, "output.npy")
stx.io.save(fig, "output.jpg")

# Round Trip Loading
df = stx.io.load("output.csv")
arr = stx.io.load("output.npy")
```

</details>

<details>
<summary><strong><code>scitex.stats</code></strong> — Publication-Ready Statistics (23 tests)</summary>

Automatic assumption checking, effect sizes, multiple comparison corrections, 9 export formats.

```python
import scitex as stx

result = stx.stats.test_ttest_ind(
    group1, group2,
    return_as="dataframe"  # or "latex", "markdown", "excel"
)
# Includes: p-value, effect size, CI, normality check, power analysis
```

</details>

<details>
<summary><strong><code>scitex.scholar</code></strong> — Literature Management & BibTeX Enrichment</summary>

BibTeX enrichment with abstracts for LLM context, DOI resolution, PDF download, Impact Factor.

```bash
scitex scholar bibtex papers.bib --project myresearch --num-workers 8
```

```bibtex
# Before: Minimal BibTeX
@article{Smith2024,
  title = {Neural Networks},
  author = {Smith, John},
  doi = {10.1038/s41586-024-00001-1}
}

# After: Enriched with abstract for LLM context
@article{Smith2024,
  title = {Neural Networks for Brain Signal Analysis},
  author = {Smith, John and Lee, Alice},
  doi = {10.1038/s41586-024-00001-1},
  journal = {Nature},
  year = {2024},
  abstract = {We present a novel deep learning approach...},  # Rich context for LLMs
  impact_factor = {64.8}
}
```

</details>

<details>
<summary><strong><code>scitex.writer</code></strong> — LaTeX Manuscript Management</summary>

Python interface for LaTeX manuscripts with git-based version control.

```python
from scitex.writer import Writer

writer = Writer("my_paper")
intro = writer.manuscript.contents.introduction

lines = intro.read()
intro.write(lines + ["New paragraph..."])
intro.commit("Update introduction")

result = writer.compile_manuscript()
```

</details>

<details>
<summary><strong><code>scitex.ai</code></strong> — Unified AI/ML Interface (7 providers)</summary>

Single API for OpenAI, Anthropic, Google, Perplexity, DeepSeek, Groq, local models.

```python
from scitex.ai.genai import GenAI

ai = GenAI(provider="openai")
response = ai("Explain this data pattern")

# Switch providers instantly
ai = GenAI(provider="anthropic", model="claude-3-opus-20240229")
```

</details>

## 🖥️ CLI Commands

SciTeX provides a comprehensive command-line interface:

<details>
<summary><strong>CLI Commands</strong></summary>

```bash
# SciTeX Cloud (https://scitex.ai)
scitex cloud login                    # Login to SciTeX Cloud
scitex cloud clone user/project       # Clone from cloud
scitex cloud create my-project        # Create new repository
scitex cloud enrich -i refs.bib -o enriched.bib  # BibTeX enrichment API

# Configuration
scitex config list                    # Show all configured paths
scitex config init                    # Initialize directories

# Research Tools
scitex scholar bibtex papers.bib      # Process BibTeX, download PDFs
scitex scholar single --doi "10.1038/nature12373"

# TIP: Get BibTeX from Scholar QA (https://scholarqa.allen.ai/chat/)
#      Ask questions → Export All Citations → Save as .bib file
scitex stats recommend --data data.csv

# Media
scitex audio speak "Hello world"
scitex capture snap --output screenshot.jpg

# Document Processing
scitex tex compile manuscript.tex
scitex writer compile my_paper

# Utilities
scitex resource usage                 # System resource monitoring
scitex security check --save          # Security audit
scitex web get-urls https://example.com
scitex completion                     # Enable tab completion
```

</details>

## 🔌 MCP Servers

<details>
<summary><strong>Model Context Protocol (MCP) servers for AI agent integration</strong></summary>

SciTeX provides Model Context Protocol (MCP) servers for AI agent integration:

| Server | Description |
|--------|-------------|
| `scitex-audio` | Text-to-speech for agent feedback |
| `scitex-capture` | Screen monitoring and capture |
| `scitex-plt` | Matplotlib figure creation |
| `scitex-stats` | Automated statistical testing |
| `scitex-scholar` | PDF download and metadata enrichment |
| `scitex-diagram` | Mermaid and flowchart creation |
| `scitex-template` | Project scaffolding |
| `scitex-canvas` | Scientific figure canvas |

**Claude Desktop Configuration** (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "scitex-audio": {
      "command": "scitex-audio"
    },
    "scitex-capture": {
      "command": "scitex-capture"
    },
    "scitex-scholar": {
      "command": "scitex-scholar"
    }
  }
}
```

</details>

## 🚀 Quick Start


### Research Analysis with **70% Less Code**

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


<details>
<summary><strong>Equivalent Script without SciTeX ([188 Lines of Code](./examples/demo_session_plt_io_pure_python.py)), requiring 3.3× more code</strong></summary>

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


## 📖 Documentation

<details>
<summary><strong>Resources</strong></summary>

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

</details>

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.


## 📄 License

AGPL-3.0.

## 📧 Contact

Yusuke Watanabe (ywatanabe@scitex.ai)

<!-- EOF -->