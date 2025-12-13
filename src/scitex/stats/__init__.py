#!/usr/bin/env python3
"""
SciTeX Stats Module - Professional Statistical Testing Framework

Clean, normalized API for statistical analysis with publication-ready outputs.

Organized submodules:
- auto: Automatic test selection, recommendation, and journal-style formatting
- correct: Multiple comparison correction methods (Bonferroni, FDR, Holm, Sidak)
- effect_sizes: Effect size computations and interpretations
- power: Statistical power analysis
- posthoc: Post-hoc tests (Tukey, Dunnett, Games-Howell)
- tests: Statistical tests (correlation, etc.)
- descriptive: Descriptive statistics
- utils: Formatters and normalizers
- _schema: Core StatResult schema for standardized test results

Quick Start (Auto Selection):
----------------------------
>>> from scitex.stats.auto import StatContext, recommend_tests
>>> ctx = StatContext(
...     n_groups=2,
...     sample_sizes=[30, 32],
...     outcome_type="continuous",
...     design="between",
...     paired=False,
...     has_control_group=False,
...     n_factors=1
... )
>>> tests = recommend_tests(ctx, top_k=3)
>>> print(tests)  # ['brunner_munzel', 'ttest_ind', 'mannwhitneyu']
"""

# Import organized submodules
from . import auto
from . import correct
from . import effect_sizes
from . import power
from . import utils
from . import posthoc
from . import descriptive
from . import tests
from . import _schema

# Export commonly used functions and classes for convenience
from .descriptive import describe
from ._schema import (
    StatResult,
    Position,
    StatStyling,
    StatPositioning,
    create_stat_result,
)

# Export key auto module classes at top level for convenience
from .auto import (
    StatContext,
    TestRule,
    TEST_RULES,
    check_applicable,
    recommend_tests,
    get_menu_items,
    StatStyle,
    get_stat_style,
    format_test_line,
    p_to_stars,
)

# =============================================================================
# .statsz Bundle Support
# =============================================================================

def save_statsz(
    comparisons,
    path,
    metadata=None,
    as_zip=False,
):
    """
    Save statistical results as a .statsz bundle.

    Parameters
    ----------
    comparisons : list of dict or list of StatResult
        List of comparison results. Each should have:
        - name: Comparison name (e.g., "Control vs Treatment")
        - method: Test method (e.g., "t-test")
        - p_value: P-value
        - effect_size: Effect size (optional)
        - ci95: 95% confidence interval (optional)
        - formatted: Star notation (optional)
    path : str or Path
        Output path (e.g., "comparison.statsz.d" or "comparison.statsz").
    metadata : dict, optional
        Additional metadata (n, seed, bootstrap_iters, etc.).
    as_zip : bool, optional
        If True, save as ZIP archive (default: False).

    Returns
    -------
    Path
        Path to saved bundle.

    Examples
    --------
    >>> import scitex.stats as sstats
    >>> comparisons = [
    ...     {
    ...         "name": "Control vs Treatment",
    ...         "method": "t-test",
    ...         "p_value": 0.003,
    ...         "effect_size": 1.21,
    ...         "ci95": [0.5, 1.8],
    ...         "formatted": "**"
    ...     }
    ... ]
    >>> sstats.save_statsz(comparisons, "results.statsz.d")
    """
    from pathlib import Path
    from scitex.io._bundle import save_bundle, BundleType

    p = Path(path)

    # Convert StatResult objects to dicts if needed
    comp_dicts = []
    for comp in comparisons:
        if hasattr(comp, 'to_dict'):
            comp_dicts.append(comp.to_dict())
        elif hasattr(comp, '__dict__'):
            comp_dicts.append(vars(comp))
        else:
            comp_dicts.append(comp)

    # Build spec
    spec = {
        'schema': {'name': 'scitex.stats.stats', 'version': '1.0.0'},
        'comparisons': comp_dicts,
        'metadata': metadata or {},
    }

    bundle_data = {'spec': spec}

    return save_bundle(bundle_data, p, bundle_type=BundleType.STATSZ, as_zip=as_zip)


def load_statsz(path):
    """
    Load a .statsz bundle.

    Parameters
    ----------
    path : str or Path
        Path to .statsz bundle (directory or ZIP).

    Returns
    -------
    dict
        Stats data with:
        - 'comparisons': List of comparison dicts
        - 'metadata': Metadata dict

    Examples
    --------
    >>> stats = scitex.stats.load_statsz("results.statsz.d")
    >>> for comp in stats['comparisons']:
    ...     print(f"{comp['name']}: p={comp['p_value']}")
    """
    from scitex.io._bundle import load_bundle

    bundle = load_bundle(path)

    if bundle['type'] != 'statsz':
        raise ValueError(f"Not a .statsz bundle: {path}")

    spec = bundle.get('spec', {})

    return {
        'comparisons': spec.get('comparisons', []),
        'metadata': spec.get('metadata', {}),
    }


__all__ = [
    # Main submodules
    "auto",
    "correct",
    "effect_sizes",
    "power",
    "utils",
    "posthoc",
    "descriptive",
    "tests",
    "_schema",
    # Schema exports
    "describe",
    "StatResult",
    "Position",
    "StatStyling",
    "StatPositioning",
    "create_stat_result",
    # Auto module convenience exports
    "StatContext",
    "TestRule",
    "TEST_RULES",
    "check_applicable",
    "recommend_tests",
    "get_menu_items",
    "StatStyle",
    "get_stat_style",
    "format_test_line",
    "p_to_stars",
    # .statsz bundle
    "save_statsz",
    "load_statsz",
]

__version__ = "2.1.0"
