#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ~/proj/scitex-code/src/scitex/security/__init__.py

"""
SciTeX Security Module

Reusable security utilities for the SciTeX ecosystem.
Handles GitHub security alerts, secret scanning, and vulnerability management.

Usage:
    from scitex.security import check_github_alerts

    alerts = check_github_alerts()
    if alerts:
        print(f"Found {len(alerts)} security alerts!")
"""

from .github import (
    GitHubSecurityError,
    check_github_alerts,
    format_alerts_report,
    get_latest_alerts_file,
    save_alerts_to_file,
)

__all__ = [
    "check_github_alerts",
    "save_alerts_to_file",
    "get_latest_alerts_file",
    "format_alerts_report",
    "GitHubSecurityError",
]
