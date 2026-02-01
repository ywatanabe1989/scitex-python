#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/verify/_viz/__init__.py
"""Visualization subpackage for verification status.

Provides multiple visualization backends:
- Terminal: Colored text output with status icons
- Mermaid: Text-based diagrams for docs/GitHub
- HTML: Interactive web visualization
- Plotly: Interactive Python-based visualization (optional)
"""

from ._colors import Colors, VerificationLevel
from ._format import (
    format_chain_verification,
    format_list,
    format_run_detailed,
    format_run_verification,
    format_status,
)
from ._mermaid import generate_html_dag, generate_mermaid_dag, render_dag
from ._utils import print_verification_summary

# Optional Plotly support
try:
    from ._plotly import generate_plotly_dag, render_plotly_dag

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

    def generate_plotly_dag(*args, **kwargs):
        raise ImportError("plotly required: pip install plotly")

    def render_plotly_dag(*args, **kwargs):
        raise ImportError("plotly required: pip install plotly")


__all__ = [
    "Colors",
    "VerificationLevel",
    "format_run_verification",
    "format_chain_verification",
    "format_status",
    "format_list",
    "format_run_detailed",
    "generate_mermaid_dag",
    "generate_html_dag",
    "render_dag",
    "print_verification_summary",
    "generate_plotly_dag",
    "render_plotly_dag",
]


# EOF
