<!-- ---
!-- Timestamp: 2026-01-08 09:40:07
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
<img src="https://img.shields.io/badge/core-~3s-brightgreen" alt="Core Install Time">
<img src="https://img.shields.io/badge/heavy-~60s-yellow" alt="Heavy Install Time">
<img src="https://img.shields.io/badge/uv-recommended-blue" alt="uv recommended">
</td>
</tr>
</table>
</div>

<details>
<summary><strong>Module Status (v2.11.0) - Tests & Installation</strong></summary>

### Core Modules

| Module | Tests | Install | Description |
|--------|-------|---------|-------------|
| [`io`](./src/scitex/io#readme) | [![io](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-io.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-io.yml) | `uv pip install scitex[io]` | Universal I/O (30+ formats) |
| [`path`](./src/scitex/path#readme) | [![path](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-path.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-path.yml) | `uv pip install scitex[path]` | Path utilities |
| [`str`](./src/scitex/str#readme) | [![str](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-str.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-str.yml) | `uv pip install scitex[str]` | String processing |
| [`dict`](./src/scitex/dict#readme) | [![dict](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dict.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dict.yml) | core | Dictionary utilities |
| [`types`](./src/scitex/types#readme) | [![types](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-types.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-types.yml) | core | Type checking |
| [`config`](./src/scitex/config#readme) | [![config](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-config.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-config.yml) | `uv pip install scitex[config]` | Configuration management |
| [`utils`](./src/scitex/utils#readme) | [![utils](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-utils.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-utils.yml) | `uv pip install scitex[utils]` | General utilities |
| [`decorators`](./src/scitex/decorators#readme) | [![decorators](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-decorators.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-decorators.yml) | `uv pip install scitex[decorators]` | Function decorators |
| [`logging`](./src/scitex/logging#readme) | [![logging](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-logging.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-logging.yml) | `uv pip install scitex[logging]` | Structured logging |
| [`gen`](./src/scitex/gen#readme) | [![gen](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-gen.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-gen.yml) | `uv pip install scitex[gen]` | Project setup |

### Data Science & Statistics

| Module | Tests | Install | Description |
|--------|-------|---------|-------------|
| [`stats`](./src/scitex/stats#readme) | [![stats](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-stats.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-stats.yml) | `uv pip install scitex[stats]` | Statistical tests & analysis |
| [`pd`](./src/scitex/pd#readme) | [![pd](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-pd.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-pd.yml) | `uv pip install scitex[pd]` | Pandas extensions |
| [`linalg`](./src/scitex/linalg#readme) | [![linalg](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-linalg.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-linalg.yml) | `uv pip install scitex[linalg]` | Linear algebra |
| [`plt`](./src/scitex/plt#readme) | [![plt](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-plt.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-plt.yml) | `uv pip install scitex[plt]` | Enhanced matplotlib |
| [`dsp`](./src/scitex/dsp#readme) | [![dsp](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dsp.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dsp.yml) | `uv pip install scitex[dsp]` | Signal processing |

### AI & Machine Learning

| Module | Tests | Install | Description |
|--------|-------|---------|-------------|
| [`ai`](./src/scitex/ai#readme) | [![ai](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-ai.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-ai.yml) | `uv pip install scitex[ai]` | GenAI (7 providers) |
| [`nn`](./src/scitex/nn#readme) | [![nn](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-nn.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-nn.yml) | `uv pip install scitex[nn]` | Neural network layers |
| [`torch`](./src/scitex/torch#readme) | [![torch](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-torch.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-torch.yml) | `uv pip install scitex[torch]` | PyTorch utilities |

### System & Tools

| Module | Tests | Install | Description |
|--------|-------|---------|-------------|
| [`cli`](./src/scitex/cli#readme) | [![cli](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-cli.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-cli.yml) | `uv pip install scitex[cli]` | Command-line tools |
| [`sh`](./src/scitex/sh#readme) | [![sh](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-sh.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-sh.yml) | `uv pip install scitex[sh]` | Shell utilities |
| [`git`](./src/scitex/git#readme) | [![git](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-git.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-git.yml) | `uv pip install scitex[git]` | Git operations |
| [`session`](./src/scitex/session#readme) | [![session](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-session.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-session.yml) | `uv pip install scitex[session]` | Session management |
| [`resource`](./src/scitex/resource#readme) | [![resource](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-resource.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-resource.yml) | `uv pip install scitex[resource]` | System monitoring |
| [`db`](./src/scitex/db#readme) | [![db](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-db.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-db.yml) | `uv pip install scitex[db]` | Database abstractions |

### Research & Publishing

| Module | Tests | Install | Description |
|--------|-------|---------|-------------|
| [`writer`](./src/scitex/writer#readme) | [![writer](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-writer.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-writer.yml) | `uv pip install scitex[writer]` | Document generation |
| [`tex`](./src/scitex/tex#readme) | [![tex](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-tex.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-tex.yml) | `uv pip install scitex[tex]` | LaTeX processing |
| [`msword`](./src/scitex/msword#readme) | [![msword](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-msword.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-msword.yml) | `uv pip install scitex[msword]` | MS Word conversion |
| [`scholar`](./src/scitex/scholar#readme) | [![scholar](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-scholar.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-scholar.yml) | `uv pip install scitex[scholar]` | Literature management |
| [`diagram`](./src/scitex/diagram#readme) | [![diagram](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-diagram.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-diagram.yml) | `uv pip install scitex[diagram]` | Diagram generation |

### Web & Automation

| Module | Tests | Install | Description |
|--------|-------|---------|-------------|
| [`web`](./src/scitex/web#readme) | [![web](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-web.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-web.yml) | `uv pip install scitex[web]` | Web scraping |
| [`browser`](./src/scitex/browser#readme) | [![browser](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-browser.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-browser.yml) | `uv pip install scitex[browser]` | Browser automation |

### Other Modules

| Module | Tests | Install | Description |
|--------|-------|---------|-------------|
| [`audio`](./src/scitex/audio#readme) | [![audio](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-audio.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-audio.yml) | `uv pip install scitex[audio]` | Audio processing |
| [`capture`](./src/scitex/capture#readme) | [![capture](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-capture.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-capture.yml) | `uv pip install scitex[capture]` | Screen capture |
| [`repro`](./src/scitex/repro#readme) | [![repro](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-repro.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-repro.yml) | `uv pip install scitex[repro]` | Reproducibility |
| [`benchmark`](./src/scitex/benchmark#readme) | [![benchmark](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-benchmark.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-benchmark.yml) | `uv pip install scitex[benchmark]` | Performance testing |
| [`security`](./src/scitex/security#readme) | [![security](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-security.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-security.yml) | `uv pip install scitex[security]` | Security utilities |
| [`dt`](./src/scitex/dt#readme) | [![dt](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dt.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dt.yml) | `uv pip install scitex[dt]` | Datetime utilities |
| [`dev`](./src/scitex/dev#readme) | [![dev](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dev.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-dev.yml) | `uv pip install scitex[dev]` | Development tools |
| [`schema`](./src/scitex/schema#readme) | [![schema](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-schema.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-schema.yml) | `uv pip install scitex[schema]` | Data schemas |
| [`bridge`](./src/scitex/bridge#readme) | [![bridge](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-bridge.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-code/actions/workflows/test-bridge.yml) | `uv pip install scitex[bridge]` | Module integration |

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
<summary><strong>Module Overview</strong></summary>

SciTeX is organized into focused modules for different aspects of scientific computing:

**Modular Installation** (See [./src/scitex](./src/scitex) for all available modules):
``` bash
# Install all modules
uv pip install scitex[all]

# Install only specific modules
uv pip install scitex[ai]
```

<details>
<summary><strong>All available modules, equivalent to scitex[all]</strong></summary>

``` bash
uv pip install ~/proj/scitex-code[\
ai,\
audio,\
benchmark,\
bridge,\
browser,\
capture,\
cli,\
cloud,\
config,\
db,\
decorators,\
diagram,\
dsp,\
devtools,\
dt,\
fig,\
fts,\
gen,\
git,\
io,\
linalg,\
logging,\
msword,\
nn,\
parallel,\
path,\
pd,\
plt,\
repro,\
resource,\
scholar,\
session,\
sh,\
stats,\
str,\
tex,\
torch,\
types,\
utils,\
web,\
writer,\
dev]
```

</details>

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


<details>
<summary><strong>Equivalent without SciTeX ([188 Lines of Code](./examples/demo_session_plt_io_pure_python.py)), requiring 3.3× more code</strong></summary>

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


</details>

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.


## 📄 License

AGPL-3.0.

## 📧 Contact

Yusuke Watanabe (ywatanabe@scitex.ai)

<!-- EOF -->