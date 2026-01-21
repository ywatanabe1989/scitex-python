#!/usr/bin/env python3
# Timestamp: 2026-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_resources/_cheatsheet.py
"""Core SciTeX cheatsheet resource for AI agents."""

from __future__ import annotations

__all__ = ["register_cheatsheet_resources"]

CHEATSHEET = """\
# SciTeX Cheatsheet for AI Agents
=================================

## Import Pattern (ALWAYS use this)
```python
import scitex as stx
import numpy as np
```

## 1. @stx.session - Reproducible Experiment Tracking

The MOST IMPORTANT pattern. Wrap your main function with @stx.session:

```python
@stx.session
def main(
    # User parameters (become CLI arguments automatically)
    input_file="data.csv",       # --input-file (default: data.csv)
    n_samples=100,               # --n-samples (default: 100)

    # INJECTED parameters (auto-provided by session)
    CONFIG=stx.INJECTED,         # Session config with ID, paths
    plt=stx.INJECTED,            # Pre-configured matplotlib
    COLORS=stx.INJECTED,         # Color palette
    rng=stx.INJECTED,            # Seeded random generator
    logger=stx.INJECTED,         # Session logger
):
    \"\"\"This docstring becomes --help description.\"\"\"

    # Your analysis code here
    data = stx.io.load(input_file)
    results = process(data, n_samples)

    # Save outputs (automatically to session directory)
    stx.io.save(results, "results.csv")
    stx.io.save(fig, "plot.png", symlink_to="./data")

    return 0  # Exit status

if __name__ == "__main__":
    main()  # CLI mode when no args passed
```

## 2. stx.io - Universal File I/O (30+ formats)

ALWAYS use stx.io.save() and stx.io.load() for file operations:

```python
# Saving - extension determines format automatically
stx.io.save(df, "data.csv")           # DataFrame -> CSV
stx.io.save(arr, "data.npy")          # NumPy array
stx.io.save(obj, "data.pkl")          # Any Python object
stx.io.save(fig, "plot.png")          # Figure + auto CSV

# Loading - format auto-detected
data = stx.io.load("data.csv")

# With options
stx.io.save(fig, "plot.png",
    metadata={"exp": "exp01"},  # Embedded metadata
    symlink_to="./data",        # Create symlink
    verbose=True,               # Log messages
)
```

IMPORTANT: stx.io.save shows logging messages:
```
SUCC: Saved to: ./script_out/results.csv (4.0 KiB)
SUCC: Symlinked: ./script_out/results.csv -> ./data/results.csv
```

## 3. stx.plt - Publication-Ready Figures (Auto CSV Export)

```python
fig, ax = stx.plt.subplots()

# Use stx_ prefixed methods for auto CSV export
ax.stx_line(x, y, label="Signal")     # Tracked: exports to CSV
ax.stx_scatter(x, y)                   # Tracked

# Set labels with convenience method
ax.set_xyt("X axis", "Y axis", "Title")

# Save figure -> creates BOTH plot.png AND plot.csv
stx.io.save(fig, "plot.png")
fig.close()
```

## 4. stx.stats - Publication Statistics (23 tests)

```python
result = stx.stats.test_ttest_ind(group1, group2, return_as="dataframe")
# Returns: p-value, Cohen's d, 95% CI, normality check, power analysis

result = stx.stats.test_anova(*groups, return_as="latex")
```

## 5. stx.scholar - Literature Management

```bash
scitex scholar bibtex papers.bib --project myresearch --num-workers 8
# Enriches BibTeX with abstracts, DOIs, impact factors, downloads PDFs
```

## Quick Tips

1. ALWAYS use `import scitex as stx`
2. ALWAYS wrap main functions with `@stx.session`
3. ALWAYS use `stx.io.save()` and `stx.io.load()` for files
4. ALWAYS use `stx.plt.subplots()` for figures
5. ALWAYS use `ax.stx_*` methods for auto CSV export
6. ALWAYS return exit status (0 for success) from main
"""


def register_cheatsheet_resources(mcp) -> None:
    """Register cheatsheet resource."""

    @mcp.resource("scitex://cheatsheet")
    def cheatsheet() -> str:
        """Complete SciTeX quick reference for AI code generation."""
        return CHEATSHEET


# EOF
