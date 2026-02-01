#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/verify/_viz/_colors.py
"""Color constants and status icons for verification visualization."""

from __future__ import annotations

from .._chain import VerificationStatus


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class VerificationLevel:
    """Verification level for display badges."""

    CACHE = "cache"  # verified-by-cache (fast hash comparison) - 🟢
    SCRATCH = "scratch"  # verified-from-scratch (re-executed) - 🟢🟢


def status_icon(
    status: VerificationStatus,
    level: str = VerificationLevel.CACHE,
) -> str:
    """
    Get colored icon for verification status.

    Parameters
    ----------
    status : VerificationStatus
        The verification status
    level : str
        Verification level: 'cache' (🟢) or 'scratch' (🟢🟢)

    Returns
    -------
    str
        Colored status icon
    """
    if level == VerificationLevel.SCRATCH and status == VerificationStatus.VERIFIED:
        return f"{Colors.GREEN}●●{Colors.RESET}"

    icons = {
        VerificationStatus.VERIFIED: f"{Colors.GREEN}●{Colors.RESET}",
        VerificationStatus.MISMATCH: f"{Colors.RED}●{Colors.RESET}",
        VerificationStatus.MISSING: f"{Colors.YELLOW}○{Colors.RESET}",
        VerificationStatus.UNKNOWN: f"{Colors.CYAN}?{Colors.RESET}",
    }
    return icons.get(status, "?")


def status_text(status: VerificationStatus) -> str:
    """
    Get colored text for verification status.

    Parameters
    ----------
    status : VerificationStatus
        The verification status

    Returns
    -------
    str
        Colored status text
    """
    texts = {
        VerificationStatus.VERIFIED: f"{Colors.GREEN}verified{Colors.RESET}",
        VerificationStatus.MISMATCH: f"{Colors.RED}mismatch{Colors.RESET}",
        VerificationStatus.MISSING: f"{Colors.YELLOW}missing{Colors.RESET}",
        VerificationStatus.UNKNOWN: f"{Colors.CYAN}unknown{Colors.RESET}",
    }
    return texts.get(status, "unknown")


# EOF
