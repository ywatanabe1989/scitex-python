#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/template/_templates/plt.py
# ----------------------------------------

"""Template for stx.plt plotting module usage."""

TEMPLATE = {
    "name": "Plotting Module",
    "description": "stx.plt usage for publication-ready figures with automatic CSV export",
    "filename": "plot_script.py",
    "priority": 2,
    "content": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: {timestamp}
# File: {filepath}

"""
stx.plt - Publication-Ready Plotting
====================================

stx.plt wraps matplotlib with:
- Automatic CSV export when saving figures (reproducibility)
- Pre-configured paper-quality styles
- Enhanced axis methods (set_xyt, plot_line, etc.)
- Color palettes (COLORS injection in @stx.session)

Usage Patterns
--------------
1. Direct API: import scitex as stx; stx.plt.subplots()
2. With @stx.session: plt is auto-injected with proper style
3. CLI: scitex plt <recipe.yaml> (for FigRecipe specs)
4. MCP: plt_plot, plt_compose, plt_reproduce tools
"""

import scitex as stx

# ============================================================
# Pattern 1: Direct API Usage
# ============================================================

# Basic figure creation
fig, ax = stx.plt.subplots()
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 5, 3]
ax.plot(x, y, label="Data")
ax.set_xyt("X Axis", "Y Axis", "My Plot")  # Enhanced: xlabel, ylabel, title
ax.legend()

# Save with auto CSV export (creates plot.png + plot.csv)
stx.io.save(fig, "plot.png")

# ============================================================
# Pattern 2: Multiple Subplots
# ============================================================

fig, axes = stx.plt.subplots(2, 2, figsize=(10, 8))

# Flatten for easier iteration
for idx, ax in enumerate(axes.flat):
    ax.plot([1, 2, 3], [idx, idx*2, idx*3])
    ax.set_xyt(f"X{{idx}}", f"Y{{idx}}", f"Panel {{idx}}")

fig.tight_layout()
stx.io.save(fig, "multipanel.png")

# ============================================================
# Pattern 3: With @stx.session (Recommended)
# ============================================================

@stx.session
def main(
    n_points=100,
    CONFIG=stx.INJECTED,   # Session config
    plt=stx.INJECTED,      # Pre-configured matplotlib
    COLORS=stx.INJECTED,   # Color palette
    rng=stx.INJECTED,      # Random generator
    logger=stx.INJECTED,   # Logger
):
    """Example plotting with session management."""
    import numpy as np

    # Generate data
    x = np.linspace(0, 10, n_points)
    y1 = np.sin(x) + rng.normal(0, 0.1, n_points)
    y2 = np.cos(x) + rng.normal(0, 0.1, n_points)

    # Create figure using injected plt
    fig, ax = plt.subplots()

    # Use injected COLORS for consistent styling
    ax.plot(x, y1, color=COLORS[0], label="sin(x)")
    ax.plot(x, y2, color=COLORS[1], label="cos(x)")

    # Enhanced axis methods
    ax.set_xyt("Time (s)", "Amplitude", "Waveform Comparison")
    ax.legend()

    # Save to CONFIG.SDIR_OUT (auto CSV export)
    stx.io.save(fig, CONFIG.SDIR_OUT / "waveforms.png")
    logger.info(f"Saved figure to {{CONFIG.SDIR_OUT}}")

    return 0

# ============================================================
# Pattern 4: Statistical Plots
# ============================================================

def statistical_plots():
    """Common statistical visualization patterns."""
    import numpy as np

    fig, axes = stx.plt.subplots(1, 3, figsize=(12, 4))

    # Box plot
    data = [np.random.randn(100) + i for i in range(3)]
    axes[0].boxplot(data)
    axes[0].set_xyt("Group", "Value", "Box Plot")

    # Histogram
    axes[1].hist(np.random.randn(1000), bins=30, edgecolor='black')
    axes[1].set_xyt("Value", "Count", "Histogram")

    # Scatter with regression
    x = np.random.randn(50)
    y = 2*x + np.random.randn(50)*0.5
    axes[2].scatter(x, y, alpha=0.6)
    axes[2].set_xyt("X", "Y", "Scatter Plot")

    fig.tight_layout()
    stx.io.save(fig, "stats_plots.png")

# ============================================================
# Pattern 5: Heatmaps and Images
# ============================================================

def heatmap_example():
    """Heatmap visualization pattern."""
    import numpy as np

    fig, ax = stx.plt.subplots()

    # Create correlation matrix
    data = np.random.randn(10, 10)
    corr = np.corrcoef(data)

    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xyt("Feature", "Feature", "Correlation Matrix")
    fig.colorbar(im, ax=ax, label="Correlation")

    stx.io.save(fig, "heatmap.png")

# ============================================================
# Key Features Summary
# ============================================================

"""
stx.plt Key Features:
---------------------
1. Enhanced Axis Methods:
   - ax.set_xyt(xlabel, ylabel, title)  # All in one call
   - ax.plot_line(x, y, **kwargs)       # With auto data tracking

2. Auto CSV Export:
   - stx.io.save(fig, "plot.png")       # Creates plot.png + plot.csv
   - CSV contains all plotted data for reproducibility

3. Pre-configured Styles:
   - Publication-quality fonts and sizes
   - Consistent color palettes
   - Grid and tick configurations

4. Color Palettes (via COLORS injection):
   - COLORS[0], COLORS[1], ... for consistent styling
   - Colorblind-friendly by default

5. Figure Recipes (advanced):
   - YAML specifications for reproducible figures
   - CLI: scitex plt recipe.yaml -o output.png
   - MCP: plt_plot, plt_compose, plt_reproduce

CLI Commands:
-------------
scitex plt <recipe.yaml>         # Render from recipe
scitex plt --help               # Show all options

MCP Tools:
----------
plt_plot         # Create figure from specification
plt_compose      # Combine multiple figures
plt_reproduce    # Reproduce from saved recipe
plt_crop         # Crop whitespace
"""

if __name__ == "__main__":
    main()
''',
}

__all__ = ["TEMPLATE"]

# EOF
