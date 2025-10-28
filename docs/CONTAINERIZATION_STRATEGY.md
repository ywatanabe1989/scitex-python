# SciTeX Containerization Strategy

## Problem Statement

SciTeX has **two levels of dependencies**:

### 1. Python Dependencies (Heavy!)
From `requirements.txt` (~100+ packages):
- **Scientific Computing**: numpy, scipy, pandas, matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, torch, transformers, optuna
- **Deep Learning**: torch, torchvision, torchaudio, accelerate
- **AI APIs**: openai, anthropic, google-genai, groq
- **Data Processing**: h5py, openpyxl, PyPDF2, pdfplumber
- **Web Scraping**: selenium, bs4, scholarly, pymed
- **Many more...**

Total installation: **~5-10GB** of Python packages!

### 2. System Dependencies (LaTeX & Tools)
- **LaTeX**: pdflatex, xelatex, lualatex, bibtex
- **ImageMagick**: convert, identify
- **Ghostscript**: ps2pdf, gs
- **Mermaid**: mmdc (for diagrams)
- **System tools**: make, git, wget

Total installation: **~2-3GB** of system packages!

## Proposed Solution: Modular Containers

Instead of one massive container, create **separate containers for different features**:

### Container Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User's System                             │
│  - Python 3.8+ (minimal)                                     │
│  - scitex package (pip install scitex)                       │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────┴────────────────┐
        │                                  │
        ▼                                  ▼
┌──────────────────┐            ┌──────────────────┐
│  scitex-core     │            │  scitex-writer   │
│  (Singularity)   │            │  (Singularity)   │
│                  │            │                  │
│  - numpy         │            │  - LaTeX Full    │
│  - scipy         │            │  - ImageMagick   │
│  - pandas        │            │  - Ghostscript   │
│  - matplotlib    │            │  - Mermaid       │
│  - requests      │            │  - bibtex tools  │
│  - minimal deps  │            │                  │
│  ~500MB          │            │  ~2GB            │
└──────────────────┘            └──────────────────┘
        │                                  │
        ▼                                  ▼
┌──────────────────┐            ┌──────────────────┐
│  scitex-scholar  │            │  scitex-ml       │
│  (Singularity)   │            │  (Singularity)   │
│                  │            │                  │
│  - scholarly     │            │  - torch         │
│  - pymed         │            │  - transformers  │
│  - selenium      │            │  - optuna        │
│  - bs4           │            │  - ML tools      │
│  ~1GB            │            │  ~3GB            │
└──────────────────┘            └──────────────────┘
```

### Container Definitions

#### 1. `scitex-core.def` (Minimal, Required)
**Size**: ~500MB
**Purpose**: Core scientific computing
**Use**: Local analysis, data processing

```singularity
Bootstrap: docker
From: python:3.10-slim

%post
    pip install --no-cache-dir \
        numpy scipy pandas matplotlib seaborn \
        requests PyYAML pathlib click

%runscript
    python "$@"
```

#### 2. `scitex-writer.def` (LaTeX Compilation)
**Size**: ~2GB
**Purpose**: LaTeX document compilation
**Use**: scitex.writer module

```singularity
Bootstrap: docker
From: texlive/texlive:latest

%post
    # Install Python
    apt-get update && apt-get install -y python3-pip

    # Install ImageMagick, Ghostscript
    apt-get install -y imagemagick ghostscript

    # Install Mermaid CLI
    apt-get install -y nodejs npm
    npm install -g @mermaid-js/mermaid-cli

    # Python packages for writer
    pip3 install --no-cache-dir PyYAML click

%runscript
    exec "$@"

%apprun pdflatex
    exec pdflatex "$@"

%apprun bibtex
    exec bibtex "$@"

%apprun convert
    exec convert "$@"
```

#### 3. `scitex-scholar.def` (Literature Management)
**Size**: ~1GB
**Purpose**: Paper search, BibTeX enrichment
**Use**: scitex.scholar module

```singularity
Bootstrap: docker
From: python:3.10

%post
    # Install dependencies
    apt-get update && apt-get install -y chromium-driver

    # Python packages
    pip install --no-cache-dir \
        scholarly pymed selenium bs4 \
        requests lxml openpyxl click

%runscript
    python "$@"
```

#### 4. `scitex-ml.def` (Machine Learning, Optional)
**Size**: ~3GB
**Purpose**: ML/DL features
**Use**: Optional advanced features

```singularity
Bootstrap: docker
From: pytorch/pytorch:latest

%post
    pip install --no-cache-dir \
        transformers accelerate optuna \
        scikit-learn pandas numpy

%runscript
    python "$@"
```

## Implementation Strategy

### Phase 1: Container Detection & Auto-Download

```python
# scitex/containers.py

from pathlib import Path
import subprocess
import logging

logger = logging.getLogger(__name__)

CONTAINERS = {
    'core': {
        'url': 'https://github.com/ywatanabe1989/scitex-containers/releases/download/v1.0/scitex-core.sif',
        'size': '500MB',
        'required': True,
    },
    'writer': {
        'url': 'https://github.com/ywatanabe1989/scitex-containers/releases/download/v1.0/scitex-writer.sif',
        'size': '2GB',
        'required': False,  # Only if using scitex.writer
    },
    'scholar': {
        'url': 'https://github.com/ywatanabe1989/scitex-containers/releases/download/v1.0/scitex-scholar.sif',
        'size': '1GB',
        'required': False,  # Only if using scitex.scholar
    },
    'ml': {
        'url': 'https://github.com/ywatanabe1989/scitex-containers/releases/download/v1.0/scitex-ml.sif',
        'size': '3GB',
        'required': False,  # Optional
    },
}

def get_container_dir() -> Path:
    """Get container storage directory."""
    return Path.home() / '.scitex' / 'containers'

def has_container(name: str) -> bool:
    """Check if container exists locally."""
    container_path = get_container_dir() / f"scitex-{name}.sif"
    return container_path.exists()

def download_container(name: str, force: bool = False):
    """Download container if not present."""
    if has_container(name) and not force:
        logger.info(f"Container '{name}' already exists")
        return

    container_dir = get_container_dir()
    container_dir.mkdir(parents=True, exist_ok=True)

    container_path = container_dir / f"scitex-{name}.sif"
    url = CONTAINERS[name]['url']
    size = CONTAINERS[name]['size']

    logger.info(f"Downloading {name} container ({size})...")
    logger.info(f"URL: {url}")

    # Use wget or curl
    subprocess.run([
        'wget', '-O', str(container_path), url
    ], check=True)

    logger.info(f"Downloaded {name} container to {container_path}")

def run_in_container(name: str, command: list) -> subprocess.CompletedProcess:
    """Run command in container."""
    if not has_container(name):
        logger.info(f"Container '{name}' not found, downloading...")
        download_container(name)

    container_path = get_container_dir() / f"scitex-{name}.sif"

    # Run with singularity
    cmd = ['singularity', 'exec', str(container_path)] + command
    return subprocess.run(cmd, capture_output=True, text=True)
```

### Phase 2: Update scitex.writer to Use Container

```python
# scitex/writer/compile.py

from scitex.containers import run_in_container, has_container

def _run_compile(...):
    # Check if container is available
    if has_container('writer'):
        # Use container
        result = run_in_container('writer', [
            '/path/to/compile', 'manuscript'
        ])
    else:
        # Fallback to native (current behavior)
        result = subprocess.run([
            '/path/to/compile', 'manuscript'
        ], ...)
```

### Phase 3: Container Management CLI

```bash
# Download specific container
$ scitex container download writer

# Download all containers
$ scitex container download --all

# List available containers
$ scitex container list

# Check container status
$ scitex container status

# Remove container
$ scitex container remove writer
```

## Advantages

### 1. Modular Installation
Users only download what they need:
- Basic usage: **core only** (~500MB)
- With LaTeX: **core + writer** (~2.5GB)
- Full featured: **all containers** (~6.5GB)

### 2. Easy Updates
Update individual containers:
```bash
$ scitex container update writer
```

### 3. Reproducibility
Lock container versions:
```yaml
# scitex-lock.yaml
containers:
  core: v1.0.0
  writer: v1.2.1
  scholar: v1.1.0
```

### 4. System Independence
Works on:
- Linux (Singularity native)
- Mac (via Singularity on VM)
- Windows (via WSL2 + Singularity)
- HPC clusters (Singularity standard)

## Alternative: Conda Environment File

For users who prefer conda:

```yaml
# environment.yml
name: scitex
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - scipy
  - pandas
  - matplotlib
  # ... minimal deps only
  - pip
  - pip:
    - scitex
```

## Recommended Approach

**Hybrid Strategy**:

1. **Minimal pip install** (no heavy deps):
   ```bash
   pip install scitex-core  # Just core Python code
   ```

2. **Container-based execution** (on-demand):
   - Auto-download containers when needed
   - Cache locally in `~/.scitex/containers/`

3. **Native fallback** (if available):
   - Check for system LaTeX first
   - Use container only if missing

## Implementation Checklist

- [ ] Create container definitions (.def files)
- [ ] Build containers
- [ ] Upload to GitHub Releases
- [ ] Implement container management in Python
- [ ] Update scitex.writer to support containers
- [ ] Add CLI for container management
- [ ] Document container usage
- [ ] Test on Linux/Mac/Windows

## File Locations

```
scitex-code/
├── containers/               # Container definitions
│   ├── scitex-core.def
│   ├── scitex-writer.def
│   ├── scitex-scholar.def
│   └── scitex-ml.def
├── src/scitex/
│   ├── containers.py         # Container management
│   └── writer/
│       └── compile.py        # Updated to use containers
└── requirements-minimal.txt  # Only essential deps
```

## Summary

**Problem**: Heavy dependencies (~10GB Python + LaTeX)
**Solution**: Modular Singularity containers
**Benefit**: Users download only what they need
**Result**: Faster installation, better portability, easier updates

This strategy allows scitex to be both **lightweight** (core only) and **full-featured** (with containers) depending on user needs!
