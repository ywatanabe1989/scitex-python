#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/session/_lifecycle/_matplotlib.py
"""Matplotlib configuration for session lifecycle."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any, Dict, Optional, Tuple

import matplotlib

from scitex.logging import getLogger

logger = getLogger(__name__)

# Detect headless/WSL environments
is_headless = False
try:
    # Check for WSL
    if "microsoft" in platform.uname().release.lower() or "WSL" in os.environ.get(
        "WSL_DISTRO_NAME", ""
    ):
        is_headless = True
    # Check for no X11 display
    elif not os.environ.get("DISPLAY"):
        is_headless = True
except Exception:
    # Fallback: if on Linux without DISPLAY, assume headless
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        is_headless = True

if is_headless:
    matplotlib.use("Agg")


from scitex.plt.utils._configure_mpl import configure_mpl


def setup_matplotlib(
    plt=None, agg: bool = False, **mpl_kwargs: Any
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Configure matplotlib settings.

    Parameters
    ----------
    plt : module
        Matplotlib.pyplot module (will be replaced with scitex.plt)
    agg : bool
        Whether to use Agg backend
    **mpl_kwargs : dict
        Additional matplotlib configuration parameters

    Returns
    -------
    tuple
        (plt, COLORS) - Configured scitex.plt module and color cycle
    """
    if plt is not None:
        plt.close("all")
        _, COLORS = configure_mpl(plt, **mpl_kwargs)
        COLORS["gray"] = COLORS["grey"]

        # Note: Backend is now set early in module initialization
        # to avoid tkinter threading issues in headless/WSL environments.
        # The 'agg' parameter is kept for backwards compatibility but has
        # no effect since backend must be set before pyplot import.
        if agg and not is_headless:
            logger.warning(
                "agg=True specified but backend was already set to Agg "
                "during module initialization for headless environment"
            )

        # Replace matplotlib.pyplot with scitex.plt to get wrapped functions
        import scitex.plt as stx_plt

        return stx_plt, COLORS
    return plt, None


# EOF
