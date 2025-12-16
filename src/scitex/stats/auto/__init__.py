#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/__init__.py

"""
Automatic Statistical Test Selection Module.

This module provides intelligent test selection for SciTeX-Viz based on
data context, experimental design, and assumption checks.

Key Features:
- Automatic test recommendation based on data characteristics
- Right-click menu generation with applicable tests
- Journal-style formatting (APA, Nature, Cell, Elsevier)
- Summary statistics computation
- Multiple comparison correction

The Brunner-Munzel test is the recommended default for 2-group comparisons
due to its robustness (no normality or equal variance assumptions).

Quick Start:
-----------
>>> from scitex.stats.auto import StatContext, recommend_tests, get_menu_items
>>>
>>> # Create context from your data
>>> ctx = StatContext(
...     n_groups=2,
...     sample_sizes=[30, 32],
...     outcome_type="continuous",
...     design="between",
...     paired=False,
...     has_control_group=False,
...     n_factors=1
... )
>>>
>>> # Get recommended tests
>>> tests = recommend_tests(ctx, top_k=3)
>>> print(tests)  # ['brunner_munzel', 'ttest_ind', 'mannwhitneyu']
>>>
>>> # Get menu items for UI
>>> items = get_menu_items(ctx)

Submodules:
----------
- _context: StatContext dataclass
- _rules: TestRule and TEST_RULES registry
- _selector: check_applicable, recommend_tests, get_menu_items
- _styles: Journal style presets (StatStyle)
- _formatting: format_test_line, summary statistics
"""

# =============================================================================
# Context
# =============================================================================
from ._context import (
    StatContext,
    OutcomeType,
    DesignType,
)

# =============================================================================
# Rules
# =============================================================================
from ._rules import (
    TestRule,
    TestFamily,
    TEST_RULES,
    get_test_rule,
    list_tests_by_family,
)

# =============================================================================
# Selector
# =============================================================================
from ._selector import (
    check_applicable,
    get_menu_items,
    recommend_tests,
    recommend_effect_sizes,
    recommend_posthoc,
    run_all_applicable_tests,
)

# =============================================================================
# Styles
# =============================================================================
from ._styles import (
    StatStyle,
    OutputTarget,
    STAT_STYLES,
    get_stat_style,
    list_styles,
    # Individual presets
    APA_LATEX_STYLE,
    APA_HTML_STYLE,
    NATURE_LATEX_STYLE,
    NATURE_HTML_STYLE,
    CELL_LATEX_STYLE,
    CELL_HTML_STYLE,
    ELSEVIER_LATEX_STYLE,
    ELSEVIER_HTML_STYLE,
    PLAIN_STYLE,
)

# =============================================================================
# Formatting
# =============================================================================
from ._formatting import (
    # Type definitions
    SummaryStatsDict,
    TestResultDict,
    EffectResultDict,
    CorrectionMethod,
    # Summary statistics
    compute_summary_stats,
    compute_summary_from_groups,
    # Symbol mapping
    get_stat_symbol,
    # Formatting
    format_test_line,
    format_test_line_compact,
    format_for_inspector,
    p_to_stars,
    # Correction
    apply_multiple_correction,
)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Context
    "StatContext",
    "OutcomeType",
    "DesignType",

    # Rules
    "TestRule",
    "TestFamily",
    "TEST_RULES",
    "get_test_rule",
    "list_tests_by_family",

    # Selector
    "check_applicable",
    "get_menu_items",
    "recommend_tests",
    "recommend_effect_sizes",
    "recommend_posthoc",
    "run_all_applicable_tests",

    # Styles
    "StatStyle",
    "OutputTarget",
    "STAT_STYLES",
    "get_stat_style",
    "list_styles",
    "APA_LATEX_STYLE",
    "APA_HTML_STYLE",
    "NATURE_LATEX_STYLE",
    "NATURE_HTML_STYLE",
    "CELL_LATEX_STYLE",
    "CELL_HTML_STYLE",
    "ELSEVIER_LATEX_STYLE",
    "ELSEVIER_HTML_STYLE",
    "PLAIN_STYLE",

    # Formatting
    "SummaryStatsDict",
    "TestResultDict",
    "EffectResultDict",
    "CorrectionMethod",
    "compute_summary_stats",
    "compute_summary_from_groups",
    "get_stat_symbol",
    "format_test_line",
    "format_test_line_compact",
    "format_for_inspector",
    "p_to_stars",
    "apply_multiple_correction",
]

__version__ = "1.0.0"

# EOF
