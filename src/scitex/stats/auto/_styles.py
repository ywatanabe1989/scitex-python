#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/stats/auto/_styles.py

"""Journal Style Presets - Publication-ready statistical formatting.

This module defines formatting styles for major journal families:
- APA (American Psychological Association)
- Nature (Nature family journals)
- Cell (Cell Press journals)
- Elsevier (Generic biomedical)

Each style specifies:
- How to format statistics (italic, symbols)
- How to format p-values (thresholds, precision)
- How to format effect sizes
- How to format sample sizes

Supports both LaTeX and HTML output targets.
"""

from typing import Dict, List, Optional

from ._stat_style import OutputTarget, StatStyle
from ._style_definitions import (
    APA_HTML_STYLE,
    APA_LATEX_STYLE,
    CELL_HTML_STYLE,
    CELL_LATEX_STYLE,
    ELSEVIER_HTML_STYLE,
    ELSEVIER_LATEX_STYLE,
    NATURE_HTML_STYLE,
    NATURE_LATEX_STYLE,
    PLAIN_STYLE,
)

# =============================================================================
# Style Registry
# =============================================================================

STAT_STYLES: Dict[str, StatStyle] = {
    "apa_latex": APA_LATEX_STYLE,
    "apa_html": APA_HTML_STYLE,
    "nature_latex": NATURE_LATEX_STYLE,
    "nature_html": NATURE_HTML_STYLE,
    "cell_latex": CELL_LATEX_STYLE,
    "cell_html": CELL_HTML_STYLE,
    "elsevier_latex": ELSEVIER_LATEX_STYLE,
    "elsevier_html": ELSEVIER_HTML_STYLE,
    "plain": PLAIN_STYLE,
}


def get_stat_style(style_id: str) -> StatStyle:
    """Look up a statistical reporting style by its ID.

    Parameters
    ----------
    style_id : str
        Style identifier (e.g., "apa_latex", "nature_html").

    Returns
    -------
    StatStyle
        The requested style, or APA LaTeX as fallback.

    Examples
    --------
    >>> style = get_stat_style("apa_latex")
    >>> style.label
    'APA (LaTeX)'

    >>> style = get_stat_style("unknown")  # Falls back to APA
    >>> style.id
    'apa_latex'
    """
    return STAT_STYLES.get(style_id, APA_LATEX_STYLE)


def list_styles(target: Optional[OutputTarget] = None) -> List[str]:
    """List available style IDs.

    Parameters
    ----------
    target : OutputTarget or None
        If provided, only list styles for this output format.

    Returns
    -------
    list of str
        Style IDs.
    """
    if target is None:
        return list(STAT_STYLES.keys())
    return [sid for sid, style in STAT_STYLES.items() if style.target == target]


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Core types
    "StatStyle",
    "OutputTarget",
    # Registry and helpers
    "STAT_STYLES",
    "get_stat_style",
    "list_styles",
    # Individual styles
    "APA_LATEX_STYLE",
    "APA_HTML_STYLE",
    "NATURE_LATEX_STYLE",
    "NATURE_HTML_STYLE",
    "CELL_LATEX_STYLE",
    "CELL_HTML_STYLE",
    "ELSEVIER_LATEX_STYLE",
    "ELSEVIER_HTML_STYLE",
    "PLAIN_STYLE",
]


# EOF
