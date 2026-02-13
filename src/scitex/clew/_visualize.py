#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/verify/_visualize.py
"""Visualization utilities for verification status.

This module re-exports from the _viz subpackage for backward compatibility.

Provides:
- Terminal output with colored status icons
- Mermaid DAG generation
- Interactive HTML visualization
- PNG/SVG export via scitex.diagram
"""

from ._viz import (
    Colors,
    VerificationLevel,
    format_chain_verification,
    format_list,
    format_run_detailed,
    format_run_verification,
    format_status,
    generate_html_dag,
    generate_mermaid_dag,
    print_verification_summary,
    render_dag,
)

__all__ = [
    "Colors",
    "VerificationLevel",
    "format_run_verification",
    "format_run_detailed",
    "format_chain_verification",
    "format_status",
    "format_list",
    "generate_mermaid_dag",
    "generate_html_dag",
    "render_dag",
    "print_verification_summary",
]


# EOF
