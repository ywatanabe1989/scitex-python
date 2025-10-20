# SciTeX Installation Guide

## Quick Start

### Recommended Installation (Most Users)
Install SciTeX with commonly used dependencies:
```bash
pip install scitex[recommended]
```

### Full Installation
Install all features (2-5 GB download):
```bash
pip install scitex[all]
```

### Minimal Installation
Core functionality only (200-300 MB):
```bash
pip install scitex
```

## Installation Options

### Feature-Specific Installation

#### Deep Learning
For PyTorch-based deep learning features (2-4 GB):
```bash
pip install scitex[dl]
```

#### Machine Learning
Additional ML tools (scikit-image, catboost, optuna):
```bash
pip install scitex[ml]
```

#### Scholar Module
Paper management and browser automation:
```bash
pip install scitex[scholar]
```

#### AI APIs
OpenAI, Anthropic, Google, Groq clients:
```bash
pip install scitex[ai-apis]
```

#### Neuroscience
EEG/MEG analysis (MNE, pyedflib, etc.):
```bash
pip install scitex[neuro]
```

#### Web Frameworks
FastAPI, Flask, Streamlit:
```bash
pip install scitex[web]
```

#### Jupyter Notebooks
JupyterLab and notebook tools:
```bash
pip install scitex[jupyter]
```

### Combining Multiple Features
```bash
# Deep learning + Scholar
pip install scitex[dl,scholar]

# ML + Jupyter + Web
pip install scitex[ml,jupyter,web]

# Everything except development tools
pip install scitex[all]
```

## Development Installation

For contributing to SciTeX:
```bash
git clone https://github.com/ywatanabe1989/scitex-code.git
cd scitex-code
pip install -e .[dev]
```

## Dependency Size Comparison

| Installation | Download Size | Packages | Use Case |
|-------------|---------------|----------|----------|
| `scitex` | ~200-300 MB | ~30 | Core scientific computing |
| `scitex[recommended]` | ~500-800 MB | ~50 | Most common use cases |
| `scitex[dl]` | ~2-4 GB | +11 | Deep learning with PyTorch |
| `scitex[all]` | ~2-5 GB | ~200 | All features |

## Troubleshooting

### Missing Dependency Errors

If you see an error like:
```
ImportError: 
======================================================================
Optional dependency 'torch' is not installed.

To use this feature, install it with:
  pip install scitex[dl]

Or install all optional dependencies:
  pip install scitex[all]
======================================================================
```

Simply install the suggested extra:
```bash
pip install scitex[dl]
```

### Upgrading

To upgrade SciTeX with all your existing extras:
```bash
pip install --upgrade scitex[all]
```

## What's Included

### Core Installation (`scitex`)
- numpy, scipy, pandas - Scientific computing
- matplotlib, seaborn, plotly - Visualization
- scikit-learn, statsmodels - Basic ML
- h5py, PyYAML, openpyxl - Data formats
- requests, joblib, psutil - Utilities

### Optional Groups
- **dl**: PyTorch, transformers (2-4 GB)
- **ml**: scikit-image, catboost, optuna
- **scholar**: Browser automation, PDF tools
- **ai-apis**: OpenAI, Anthropic, etc.
- **neuro**: MNE, obspy (specialized science)
- **web**: FastAPI, Flask, Streamlit
- **jupyter**: JupyterLab, papermill
- **dev**: Testing, linting, docs (developers only)

## Why Optional Dependencies?

SciTeX has many features, but not everyone needs everything:
- **PyTorch alone is 800-2000 MB** (CPU) or 2-4 GB (GPU)
- Reduces download time by 90%+ for users who don't need deep learning
- Allows faster CI/CD pipelines
- More professional package structure

Choose what you need, install only that!
