#!/usr/bin/env python3
# Timestamp: "2026-02-12 02:25:42 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/stats/_plot_helpers.py

"""Shared plotting helpers for scitex.stats test visualization.

Eliminates boilerplate duplication across all _plot_* functions.
"""

import numpy as np

__all__ = [
    "compose_panels",
    "ensure_figure",
    "stats_text_box",
    "significance_bracket",
    "get_palette",
    "violin_swarm",
    "scatter_regression",
]


def compose_panels(panel_funcs, layout="horizontal", panel_labels=True):
    """Create a composed figure from individual panel plot functions.

    Each panel is created as a separate figure with full SCITEX theme,
    saved as a recipe, then composed via figrecipe.compose().

    Parameters
    ----------
    panel_funcs : list of callable
        Each callable takes a single axes argument and plots on it.
    layout : str or tuple
        'horizontal' for 1xN, 'vertical' for Nx1, or (nrows, ncols).
    panel_labels : bool
        Whether to add A, B, C... labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The composed figure.
    """
    import os
    import shutil
    import tempfile

    import scitex as stx

    _ = stx.plt.load_style()

    tmp = tempfile.mkdtemp()
    panel_paths = []

    try:
        for i, plot_func in enumerate(panel_funcs):
            fig_i, ax_i = stx.plt.subplots()
            plot_func(ax_i)
            path_i = os.path.join(tmp, f"panel_{i}.yaml")
            stx.plt.save(fig_i, path_i, verbose=False, validate=False)
            stx.plt.close(fig_i)
            panel_paths.append(path_i)

        if layout == "horizontal":
            sources = {(0, i): p for i, p in enumerate(panel_paths)}
        elif layout == "vertical":
            sources = {(i, 0): p for i, p in enumerate(panel_paths)}
        else:
            nrows, ncols = layout
            sources = {}
            for i, p in enumerate(panel_paths):
                r, c = divmod(i, ncols)
                sources[(r, c)] = p

        fig, _axes = stx.plt.compose(
            sources=sources,
            panel_labels=panel_labels,
            label_style="uppercase",
        )
        return fig
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ensure_figure(plot, ax, ncols=1, figsize=None):
    """Create figure if needed, standardizing the plot/ax logic.

    Parameters
    ----------
    plot : bool
        Whether plotting was requested.
    ax : matplotlib.axes.Axes or None
        Pre-existing axes, if any.
    ncols : int
        Number of subplot columns (for multi-panel plots).
    figsize : tuple, optional
        Figure size override.

    Returns
    -------
    created : bool
        Whether a new figure was created.
    ax_or_axes : Axes, array of Axes, or None
        The axes to plot on, or None if no plot.
    """
    import scitex as stx

    if ax is not None:
        return False, ax
    if plot:
        kw = {}
        if figsize:
            kw["figsize"] = figsize
        if ncols == 1:
            fig, ax = stx.plt.subplots(**kw)
            return True, ax
        else:
            fig, axes = stx.plt.subplots(1, ncols, **kw)
            return True, axes
    return False, None


def stats_text_box(ax, lines):
    """Draw stats annotation text box at top-left corner.

    Automatically italicizes plain ``$t$``, ``$F$``, ``$p$`` etc. via
    :func:`~scitex.stats._utils._formatters.italicize_stats`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    lines : list of str
        Each line of the annotation (supports matplotlib mathtext).
    """
    from scitex.stats._utils._formatters import italicize_stats

    text = "\n".join(italicize_stats(line) for line in lines)
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        color="black",
        fontsize=6,
    )


def significance_bracket(ax, x1, x2, stars, data_groups):
    """Draw a significance bracket with stars above the data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x1, x2 : float
        X positions of the bracket endpoints.
    stars : str
        Significance stars string (e.g., "***", "ns").
    data_groups : list of array-like
        Data arrays to compute y-range for bracket placement.
    """
    y_max = max(np.max(g) for g in data_groups)
    y_min = min(np.min(g) for g in data_groups)
    y_range = y_max - y_min
    y_pos = y_max + 0.05 * y_range

    ax.plot([x1, x2], [y_pos, y_pos], color="black")
    ax.text(
        (x1 + x2) / 2,
        y_pos + 0.02 * y_range,
        stars,
        ha="center",
        va="bottom",
        color="black",
        fontsize=6,
    )


def get_palette(n_colors):
    """Load SCITEX palette colors from figrecipe style.

    Parameters
    ----------
    n_colors : int
        Number of colors needed.

    Returns
    -------
    colors : list of tuple
        RGB tuples in [0, 1] range.
    """
    try:
        from figrecipe.styles import load_style

        style = load_style()
        palette = style.get("colors", {}).get("palette", [])
        return [tuple(v / 255.0 for v in c) for c in palette[:n_colors]]
    except ImportError:
        return [None] * n_colors


def violin_swarm(ax, groups, positions, var_names):
    """Draw violin + box + jittered scatter for group comparison.

    Delegates to figrecipe's ``violinplot`` with ``kde_extend=True``
    and ``inner="box+swarm"`` for publication-quality rendering with
    smooth KDE tails.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (AxisWrapper or raw matplotlib).
    groups : list of array-like
        Data arrays for each group.
    positions : list of float
        X positions for each group.
    var_names : list of str
        Labels for each group.
    """
    colors = get_palette(len(groups))

    # Raw matplotlib axes — bypasses AxisWrapper postprocessor that
    # would force alpha=1.0 on violin bodies and add a duplicate boxplot.
    mpl_ax = getattr(ax, "_axes_mpl", ax)

    from figrecipe._wrappers._axes_plots import _add_violin_inner_elements
    from figrecipe._wrappers._violin_kde import draw_kde_violins
    from figrecipe.styles._internal import get_style

    style = get_style()
    violin_style = style.get("violinplot", {}) if style else {}

    draw_kde_violins(mpl_ax, groups, positions, colors, violin_style)
    _add_violin_inner_elements(mpl_ax, groups, positions, "box+swarm", violin_style)

    # Axis formatting
    mpl_ax.set_xticks(positions)
    mpl_ax.set_xticklabels(var_names)
    mpl_ax.set_ylabel("Value")
    x_pad = 0.5
    mpl_ax.set_xlim(min(positions) - x_pad, max(positions) + x_pad)


def scatter_regression(ax, x, y):
    """Draw scatter plot with regression line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x, y : array-like
        Data arrays.
    """
    ax.scatter(x, y)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(np.min(x), np.max(x), 100)
    ax.plot(x_line, p(x_line))


# EOF
