#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/stats/_mcp/_handlers/__init__.py

"""Stats MCP handler implementations split into modules."""

from ._corrections import correct_pvalues_handler
from ._descriptive import describe_handler
from ._effect_size import effect_size_handler
from ._format import format_results_handler
from ._normality import normality_test_handler
from ._posthoc import posthoc_test_handler
from ._power import power_analysis_handler
from ._recommend import recommend_tests_handler
from ._run_test import run_test_handler
from ._stars import p_to_stars_handler

__all__ = [
    "recommend_tests_handler",
    "run_test_handler",
    "format_results_handler",
    "power_analysis_handler",
    "correct_pvalues_handler",
    "describe_handler",
    "effect_size_handler",
    "normality_test_handler",
    "posthoc_test_handler",
    "p_to_stars_handler",
]

# EOF
