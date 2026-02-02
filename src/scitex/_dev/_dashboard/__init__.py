#!/usr/bin/env python3
# Timestamp: 2026-02-02
# File: scitex/_dev/_dashboard/__init__.py

"""Flask dashboard for scitex version management."""

from ._app import create_app, run_dashboard

__all__ = ["create_app", "run_dashboard"]

# EOF
