#!/usr/bin/env python3
# Timestamp: "2025-08-21 20:36:45 (ywatanabe)"
# File: /home/ywatanabe/proj/SciTeX-Code/src/scitex/session/__init__.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""Experiment session management for SciTeX.

This module provides session lifecycle management functionality that was previously
in scitex.session.start and scitex.session.close, now as a dedicated session management system.

Usage:
    # Session management (replaces scitex.session.start/close)
    import sys
    import matplotlib.pyplot as plt
    from scitex import session

    # Start a session
    CONFIG, sys.stdout, sys.stderr, plt, COLORS, rng = session.start(sys, plt)

    # Your experiment code here

    # Close the session
    session.close(CONFIG)

    # Session manager for advanced use cases
    manager = session.SessionManager()
    active_sessions = manager.get_active_sessions()

    # Using INJECTED sentinel for decorator parameters
    @stx.session
    def main(CONFIG=stx.session.INJECTED, plt=stx.session.INJECTED):
        ...
"""


# Sentinel object for decorator-injected parameters
class _InjectedSentinel:
    """Sentinel value indicating a parameter will be injected by a decorator."""

    def __repr__(self):
        return "<INJECTED>"


INJECTED = _InjectedSentinel()


# Import session management functionality
from ._decorator import run, session
from ._lifecycle import close, running2finished, start
from ._manager import SessionManager

# Export public API
__all__ = [
    # Sentinel for injected parameters
    "INJECTED",
    # Session lifecycle (main functions)
    "start",
    "close",
    "running2finished",
    # Session decorator (new simplified API)
    "session",
    "run",
    # Advanced session management
    "SessionManager",
]

# EOF
