#!/usr/bin/env python3
# Timestamp: 2026-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_resources/_modules.py
"""Module-specific documentation resources for AI agents."""

from __future__ import annotations

__all__ = ["register_module_resources"]

MODULE_IO = """\
# stx.io - Universal File I/O
==============================

## Core API
```python
stx.io.save(obj, path, **kwargs)  # Save any object
stx.io.load(path, **kwargs)       # Load any file
```

## Key Features

### 1. Automatic Extension Handling
```python
stx.io.save(df, "data.csv")    # Uses CSV handler
stx.io.save(df, "data.xlsx")   # Uses Excel handler
stx.io.save(arr, "data.npy")   # Uses NumPy handler
```

### 2. Verbose Logging
```python
stx.io.save(df, "data.csv", verbose=True)
# Output: SUCC: Saved to: ./script_out/data.csv (12.5 KiB)
```

### 3. Metadata Embedding (Images)
```python
stx.io.save(fig, "plot.png", metadata={"exp": "exp01"})
img, meta = stx.io.load("plot.png")
print(meta)  # {'exp': 'exp01', 'url': 'https://scitex.ai'}
```

### 4. Symlink Creation
```python
stx.io.save(results, "results.csv", symlink_to="./data")
# Creates: ./data/results.csv -> ./script_out/results.csv
```

### 5. Auto CSV Export for Figures
```python
stx.io.save(fig, "plot.png")
# Creates BOTH: plot.png AND plot.csv (with plotted data)
```

## Common Formats
- `.csv`, `.xlsx`, `.parquet` - DataFrames
- `.npy`, `.npz`, `.h5` - Arrays
- `.pkl`, `.json`, `.yaml` - Objects
- `.png`, `.jpg`, `.pdf`, `.svg` - Figures
"""

MODULE_PLT = """\
# stx.plt - Publication-Ready Figures
======================================

## Basic Usage
```python
fig, ax = stx.plt.subplots()

# Use stx_ prefixed methods for auto CSV export
ax.stx_line(x, y, label="Signal")
ax.stx_scatter(x, y)
ax.stx_bar(categories, values)
ax.stx_errorbar(x, y, yerr=err)

# Set labels with convenience method
ax.set_xyt("X axis", "Y axis", "Title")

# Save -> creates BOTH image AND CSV
stx.io.save(fig, "plot.png")
fig.close()
```

## Tracked Plot Methods (stx_ prefix)
```python
ax.stx_line(x, y)           # Line plot
ax.stx_scatter(x, y)        # Scatter plot
ax.stx_bar(x, height)       # Bar chart
ax.stx_errorbar(x, y, yerr) # Error bars
ax.stx_hist(data, bins=30)  # Histogram
ax.stx_boxplot(data)        # Box plot
ax.stx_violinplot(data)     # Violin plot
ax.stx_imshow(matrix)       # Image/heatmap
```

## Auto CSV Export
When saving with `stx.io.save(fig, "plot.png")`, CSV is auto-created:
```csv
ax_00_stx_line_0_x,ax_00_stx_line_0_y
0.0,0.0
0.1,0.0998
...
```

## Color Palette
```python
@stx.session
def main(COLORS=stx.INJECTED):
    ax.stx_line(x, y1, color=COLORS.blue)
    ax.stx_line(x, y2, color=COLORS.red)
    # Available: blue, red, green, orange, purple, navy, pink, brown, gray
```
"""

MODULE_STATS = """\
# stx.stats - Publication Statistics
=====================================

23 statistical tests with automatic assumption checking, effect sizes,
confidence intervals, and multiple output formats.

## Two-Sample Tests
```python
result = stx.stats.test_ttest_ind(group1, group2)
result = stx.stats.test_mannwhitneyu(group1, group2)  # Non-parametric
```

## Paired Sample Tests
```python
result = stx.stats.test_ttest_rel(before, after)
result = stx.stats.test_wilcoxon(before, after)
```

## Multiple Group Tests
```python
result = stx.stats.test_anova(g1, g2, g3, g4)
result = stx.stats.test_kruskal(g1, g2, g3, g4)  # Non-parametric
```

## Correlation Tests
```python
result = stx.stats.test_pearsonr(x, y)
result = stx.stats.test_spearmanr(x, y)
```

## Output Formats
```python
result = stx.stats.test_ttest_ind(g1, g2, return_as="dataframe")  # Default
result = stx.stats.test_ttest_ind(g1, g2, return_as="latex")      # Papers
result = stx.stats.test_ttest_ind(g1, g2, return_as="markdown")   # Docs
```

## Result Contents
- `statistic`: Test statistic value
- `p_value`: P-value
- `effect_size`: Cohen's d, r, eta², etc.
- `ci_low`, `ci_high`: 95% CI
- `power`: Statistical power
"""

MODULE_SCHOLAR = """\
# stx.scholar - Literature Management
======================================

BibTeX enrichment with abstracts for LLM context, DOI resolution,
PDF download, and impact factors.

## CLI Usage (Recommended)
```bash
scitex scholar bibtex papers.bib --project myresearch --num-workers 8
```

## BibTeX Enrichment

Before:
```bibtex
@article{Smith2024,
  title = {Neural Networks},
  author = {Smith, John},
  doi = {10.1038/s41586-024-00001-1}
}
```

After:
```bibtex
@article{Smith2024,
  title = {Neural Networks for Brain Signal Analysis},
  author = {Smith, John and Lee, Alice},
  doi = {10.1038/s41586-024-00001-1},
  journal = {Nature},
  year = {2024},
  abstract = {We present a novel deep learning approach...},
  impact_factor = {64.8}
}
```

The abstract provides rich context for LLM-based literature review!

## MCP Tools
- `scholar_search_papers`: Search papers
- `scholar_enrich_bibtex`: Add metadata
- `scholar_download_pdf`: Download PDFs
- `scholar_fetch_papers`: Async download
- `scholar_parse_pdf_content`: Extract text

## Tip: Get BibTeX from Scholar QA
1. Go to https://scholarqa.allen.ai/chat/
2. Ask research questions
3. Export All Citations -> Save as .bib
4. Enrich: `scitex scholar bibtex citations.bib`
"""

MODULE_SESSION = """\
# stx.session - Reproducible Experiment Tracking
=================================================

## The @stx.session Decorator
```python
@stx.session
def main(
    input_file="data.csv",       # CLI args (auto-generated)
    n_epochs=100,

    CONFIG=stx.INJECTED,         # Session config
    plt=stx.INJECTED,            # matplotlib
    COLORS=stx.INJECTED,         # Color palette
    rng=stx.INJECTED,            # Random generator
    logger=stx.INJECTED,         # Logger
):
    \"\"\"Docstring becomes --help.\"\"\"
    stx.io.save(results, "output.csv", symlink_to="./data")
    return 0

if __name__ == "__main__":
    main()
```

## Injected Variables

### CONFIG (DotDict)
```python
CONFIG.ID          # "2026Y-01M-20D-09h37m01s_boSr"
CONFIG.FILE        # "/path/to/script.py"
CONFIG.SDIR_OUT    # "/path/to/script_out"
CONFIG.SDIR_RUN    # "/path/to/script_out/RUNNING/<session_id>"
CONFIG.PID         # Process ID
CONFIG.ARGS        # {"input_file": "data.csv", ...}
```

### COLORS
```python
COLORS.blue, COLORS.red, COLORS.green, COLORS.orange
COLORS.purple, COLORS.navy, COLORS.pink, COLORS.brown
```

## YAML Config Loading
Place YAML files in `./config/` directory - auto-loaded into CONFIG:
```yaml
# ./config/experiment.yaml
model:
  hidden_size: 256
  num_layers: 3
training:
  batch_size: 32
  learning_rate: 0.001
```
Access via dot notation:
```python
CONFIG.experiment.model.hidden_size    # 256
CONFIG.experiment.training.batch_size  # 32
```

## Symlinks for Central Navigation
```python
stx.io.save(arr, "output.npy", symlink_to="./data")
# Creates: ./data/output.npy -> ../script_out/output.npy
```
Use `./data` to accumulate outputs from multiple scripts.

## Output Structure
```
script_out/
├── output.npy           # Files at ROOT (not in session dir)
└── FINISHED_SUCCESS/
    └── <session_id>/
        ├── CONFIGS/
        │   ├── CONFIG.pkl    # Pickle snapshot
        │   └── CONFIG.yaml   # Human-readable
        └── logs/
            ├── stdout.log    # print() captured
            └── stderr.log    # errors captured
```

## Best Practices
1. Always return exit status (0 for success)
2. Use stx.io.save() with symlink_to="./data"
3. Use stx.plt for figures (auto CSV export)
"""

MODULE_DOCS = {
    "io": MODULE_IO,
    "plt": MODULE_PLT,
    "stats": MODULE_STATS,
    "scholar": MODULE_SCHOLAR,
    "session": MODULE_SESSION,
}


def register_module_resources(mcp) -> None:
    """Register module documentation resources."""

    @mcp.resource("scitex://module/io")
    def module_io() -> str:
        """stx.io module documentation - Universal File I/O."""
        return MODULE_DOCS["io"]

    @mcp.resource("scitex://module/plt")
    def module_plt() -> str:
        """stx.plt module documentation - Publication-ready figures."""
        return MODULE_DOCS["plt"]

    @mcp.resource("scitex://module/stats")
    def module_stats() -> str:
        """stx.stats module documentation - Statistical tests."""
        return MODULE_DOCS["stats"]

    @mcp.resource("scitex://module/scholar")
    def module_scholar() -> str:
        """stx.scholar module documentation - Literature management."""
        return MODULE_DOCS["scholar"]

    @mcp.resource("scitex://module/session")
    def module_session() -> str:
        """stx.session module documentation - Experiment tracking."""
        return MODULE_DOCS["session"]


# EOF
