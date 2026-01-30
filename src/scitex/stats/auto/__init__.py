#!/usr/bin/env python3
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
# Context (public)
# =============================================================================
# Internal context types (hidden with underscore prefix)
from ._context import DesignType as _DesignType
from ._context import OutcomeType as _OutcomeType
from ._context import StatContext

# =============================================================================
# Formatting (public)
# =============================================================================
# Internal formatting (hidden)
from ._formatting import CorrectionMethod as _CorrectionMethod
from ._formatting import EffectResultDict as _EffectResultDict
from ._formatting import SummaryStatsDict as _SummaryStatsDict
from ._formatting import TestResultDict as _TestResultDict
from ._formatting import apply_multiple_correction
from ._formatting import compute_summary_from_groups as _compute_summary_from_groups
from ._formatting import compute_summary_stats as _compute_summary_stats
from ._formatting import format_for_inspector as _format_for_inspector
from ._formatting import format_test_line
from ._formatting import format_test_line_compact as _format_test_line_compact
from ._formatting import get_stat_symbol as _get_stat_symbol
from ._formatting import p_to_stars

# Internal rules (hidden)
# =============================================================================
# Rules (public)
# =============================================================================
from ._rules import TEST_RULES
from ._rules import TestFamily as _TestFamily
from ._rules import TestRule
from ._rules import get_test_rule as _get_test_rule
from ._rules import list_tests_by_family as _list_tests_by_family

# Internal selector functions (hidden)
# =============================================================================
# Selector (public)
# =============================================================================
from ._selector import (
    check_applicable,
    get_menu_items,
    recommend_effect_sizes,
    recommend_posthoc,
    recommend_tests,
)
from ._selector import run_all_applicable_tests as _run_all_applicable_tests

# =============================================================================
# Styles (public)
# =============================================================================
# Internal styles (hidden)
from ._styles import APA_HTML_STYLE as _APA_HTML_STYLE
from ._styles import APA_LATEX_STYLE as _APA_LATEX_STYLE
from ._styles import CELL_HTML_STYLE as _CELL_HTML_STYLE
from ._styles import CELL_LATEX_STYLE as _CELL_LATEX_STYLE
from ._styles import ELSEVIER_HTML_STYLE as _ELSEVIER_HTML_STYLE
from ._styles import ELSEVIER_LATEX_STYLE as _ELSEVIER_LATEX_STYLE
from ._styles import NATURE_HTML_STYLE as _NATURE_HTML_STYLE
from ._styles import NATURE_LATEX_STYLE as _NATURE_LATEX_STYLE
from ._styles import PLAIN_STYLE as _PLAIN_STYLE
from ._styles import STAT_STYLES as _STAT_STYLES
from ._styles import OutputTarget as _OutputTarget
from ._styles import StatStyle, get_stat_style
from ._styles import list_styles as _list_styles

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Context
    "StatContext",
    # Rules
    "TestRule",
    "TEST_RULES",
    # Selector
    "check_applicable",
    "recommend_tests",
    "recommend_effect_sizes",
    "recommend_posthoc",
    "get_menu_items",
    # Styles
    "StatStyle",
    "get_stat_style",
    # Formatting
    "format_test_line",
    "p_to_stars",
    "apply_multiple_correction",
]


# EOF
