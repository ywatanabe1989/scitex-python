#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/stats/_mcp/handlers.py

"""Handler implementations for the scitex-stats MCP server.

This module re-exports handlers from the _handlers subpackage for
backwards compatibility.
"""

from __future__ import annotations

from ._handlers import (
    correct_pvalues_handler,
    describe_handler,
    effect_size_handler,
    format_results_handler,
    normality_test_handler,
    p_to_stars_handler,
    posthoc_test_handler,
    power_analysis_handler,
    recommend_tests_handler,
    run_test_handler,
)

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
