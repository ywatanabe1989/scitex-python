#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-19 15:20:00 (ywatanabe)"
# File: ./src/scitex/plt/ax/_style/_style_suptitles.py

"""
Style figure-level titles and labels with proper font sizes.
"""

from typing import Optional


def style_suptitles(
    fig,
    suptitle_font_size_pt: float = 7,
    font_family: str = "DejaVu Sans",
):
    """
    Apply consistent styling to figure-level titles and labels.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or FigWrapper
        The figure to style
    suptitle_font_size_pt : float, optional
        Font size in points for suptitle, supxlabel, supylabel (default: 7)
    font_family : str, optional
        Font family to use (default: "DejaVu Sans")

    Returns
    -------
    fig : matplotlib.figure.Figure or FigWrapper
        The styled figure

    Examples
    --------
    >>> fig, axes = stx.plt.subplots(2, 2, **stx.plt.presets.NATURE_STYLE)
    >>> fig.suptitle("Main Title")
    >>> fig.supxlabel("X Axis Label")
    >>> fig.supylabel("Y Axis Label")
    >>> stx.ax.style_suptitles(fig)

    Notes
    -----
    This function applies font styling to:
    - fig.suptitle() - Main figure title
    - fig.supxlabel() - Figure-level X axis label
    - fig.supylabel() - Figure-level Y axis label

    All are set to the same font size (default 7pt for publication).
    """
    # Unwrap FigWrapper if needed
    if hasattr(fig, "_fig_mpl"):
        fig_mpl = fig._fig_mpl
    else:
        fig_mpl = fig

    # Style suptitle
    if fig_mpl._suptitle is not None:
        fig_mpl._suptitle.set_fontsize(suptitle_font_size_pt)
        fig_mpl._suptitle.set_fontfamily(font_family)

    # Style supxlabel (if it exists)
    if hasattr(fig_mpl, "_supxlabel") and fig_mpl._supxlabel is not None:
        fig_mpl._supxlabel.set_fontsize(suptitle_font_size_pt)
        fig_mpl._supxlabel.set_fontfamily(font_family)

    # Style supylabel (if it exists)
    if hasattr(fig_mpl, "_supylabel") and fig_mpl._supylabel is not None:
        fig_mpl._supylabel.set_fontsize(suptitle_font_size_pt)
        fig_mpl._supylabel.set_fontfamily(font_family)

    return fig


# EOF
