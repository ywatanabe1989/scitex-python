#!/usr/bin/env python3
# Timestamp: "2026-01-12 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/_figrecipe_integration.py
"""
Figrecipe integration for statistical annotations on plots.

This module provides functions to convert scitex.stats results to figrecipe
format and annotate plots with statistical significance markers.

Requires figrecipe >= 0.13.0 for full functionality.
"""

from typing import Any, Dict, List, Optional, Union

# Check if figrecipe is available
try:
    from figrecipe._integrations._scitex_stats import (
        SCITEX_STATS_AVAILABLE,
    )
    from figrecipe._integrations._scitex_stats import (
        annotate_from_stats as _fr_annotate,
    )
    from figrecipe._integrations._scitex_stats import (
        from_scitex_stats as _fr_convert,
    )
    from figrecipe._integrations._scitex_stats import (
        load_stats_bundle as _fr_load_bundle,
    )

    FIGRECIPE_AVAILABLE = True
except ImportError:
    FIGRECIPE_AVAILABLE = False
    SCITEX_STATS_AVAILABLE = False


def to_figrecipe(
    stats_result: Union[Dict[str, Any], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Convert scitex.stats result(s) to figrecipe-compatible format.

    This function converts statistical test results from scitex.stats
    into the format expected by figrecipe for plot annotations.

    Parameters
    ----------
    stats_result : dict or list of dict
        Statistical result(s) from scitex.stats functions. Supports:
        - Single comparison dict from a test function
        - List of comparison dicts
        - Flat format: {name, method, p_value, effect_size, ci95}
        - Nested format: {method: {name}, results: {p_value}}

    Returns
    -------
    dict
        Figrecipe-compatible stats dict with 'comparisons' list.
        Each comparison contains: name, p_value, stars, method, effect_size.

    Raises
    ------
    ImportError
        If figrecipe is not installed.

    Examples
    --------
    >>> from scitex import stats
    >>> result = stats.tests.ttest_ind(x, y)
    >>> fr_stats = stats.to_figrecipe(result)
    >>> # Use with figrecipe's annotation
    >>> fig.set_stats(fr_stats)

    >>> # Multiple comparisons
    >>> results = [stats.tests.ttest_ind(a, b), stats.tests.ttest_ind(a, c)]
    >>> fr_stats = stats.to_figrecipe(results)
    """
    if not FIGRECIPE_AVAILABLE:
        raise ImportError(
            "figrecipe >= 0.13.0 is required for stats conversion. "
            "Install with: pip install figrecipe"
        )

    return _fr_convert(stats_result)


def annotate(
    ax,
    stats: Union[Dict[str, Any], List[Dict[str, Any]]],
    positions: Optional[Dict[str, float]] = None,
    style: str = "stars",
    **kwargs,
) -> List[Any]:
    """Add statistical annotations to a plot axes.

    This function adds significance markers (stars, p-values, or brackets)
    to a plot based on statistical test results.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or RecordingAxes
        The axes to annotate.
    stats : dict or list of dict
        Either:
        - Figrecipe stats dict with 'comparisons' list
        - Raw scitex.stats result(s) (will be converted automatically)
    positions : dict, optional
        Mapping of group names to x positions.
        Example: {"Control": 0, "Treatment": 1}
        If None, uses sequential positions (0, 1, 2, ...).
    style : str
        Annotation style:
        - "stars": Show significance stars (*, **, ***)
        - "p_value": Show actual p-value
        - "both": Show both stars and p-value
    **kwargs
        Additional arguments passed to the annotation function:
        - y : float - Y position for annotation
        - fontsize : float - Font size
        - color : str - Text/line color

    Returns
    -------
    list
        List of matplotlib artist objects created.

    Raises
    ------
    ImportError
        If figrecipe is not installed.

    Examples
    --------
    >>> import scitex.plt as splt
    >>> from scitex import stats
    >>>
    >>> # Run statistical test
    >>> result = stats.tests.ttest_ind(control, treatment)
    >>>
    >>> # Create plot
    >>> fig, ax = splt.subplots()
    >>> ax.bar([0, 1], [control.mean(), treatment.mean()])
    >>>
    >>> # Add annotation
    >>> stats.annotate(ax, result, positions={"Control": 0, "Treatment": 1})

    >>> # Or with explicit figrecipe format
    >>> fr_stats = stats.to_figrecipe([result1, result2])
    >>> stats.annotate(ax, fr_stats, style="both")
    """
    if not FIGRECIPE_AVAILABLE:
        raise ImportError(
            "figrecipe >= 0.13.0 is required for plot annotations. "
            "Install with: pip install figrecipe"
        )

    # Convert to figrecipe format if needed
    if isinstance(stats, list) or "comparisons" not in stats:
        stats = _fr_convert(stats)

    # Unwrap scitex AxisWrapper if needed
    if hasattr(ax, "_axis_mpl"):
        ax_mpl = ax._axis_mpl
    elif hasattr(ax, "_ax"):
        ax_mpl = ax._ax
    else:
        ax_mpl = ax

    return _fr_annotate(ax_mpl, stats, positions=positions, style=style, **kwargs)


def load_and_annotate(
    ax,
    path: str,
    positions: Optional[Dict[str, float]] = None,
    style: str = "stars",
    **kwargs,
) -> List[Any]:
    """Load stats from a bundle file and annotate a plot.

    Convenience function that combines loading a .statsz bundle
    and annotating a plot in one step.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or RecordingAxes
        The axes to annotate.
    path : str
        Path to .statsz or .zip bundle file.
    positions : dict, optional
        Mapping of group names to x positions.
    style : str
        Annotation style: "stars", "p_value", or "both".
    **kwargs
        Additional arguments passed to annotate().

    Returns
    -------
    list
        List of matplotlib artist objects created.

    Examples
    --------
    >>> fig, ax = splt.subplots()
    >>> ax.bar([0, 1], [10, 15])
    >>> stats.load_and_annotate(ax, "results.statsz",
    ...                         positions={"Control": 0, "Treatment": 1})
    """
    if not FIGRECIPE_AVAILABLE:
        raise ImportError(
            "figrecipe >= 0.13.0 is required. Install with: pip install figrecipe"
        )

    fr_stats = _fr_load_bundle(path)
    return annotate(ax, fr_stats, positions=positions, style=style, **kwargs)


def check_available() -> bool:
    """Check if figrecipe integration is available.

    Returns
    -------
    bool
        True if figrecipe >= 0.13.0 is installed.
    """
    return FIGRECIPE_AVAILABLE


__all__ = [
    "FIGRECIPE_AVAILABLE",
    "to_figrecipe",
    "annotate",
    "load_and_annotate",
    "check_available",
]
