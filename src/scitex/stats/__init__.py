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
]

__version__ = "2.1.0"
