#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: src/scitex/scholar/jobs/_errors.py
# ----------------------------------------

"""Error categorization and structured error handling for Scholar jobs.

Provides error type detection, screenshot collection, and structured
error information for job failure diagnosis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ErrorType(str, Enum):
    """Categorized error types for failure diagnosis."""

    BOT_DETECTION = "bot_detection"
    TIMEOUT = "timeout"
    AUTH_REQUIRED = "auth_required"
    AUTH_EXPIRED = "auth_expired"
    CAPTCHA = "captcha"
    RATE_LIMIT = "rate_limit"
    NOT_FOUND = "not_found"
    PAYWALL = "paywall"
    NETWORK = "network"
    PDF_INVALID = "pdf_invalid"
    BROWSER_CRASH = "browser_crash"
    UNKNOWN = "unknown"


# Patterns for error categorization
_ERROR_PATTERNS = {
    ErrorType.BOT_DETECTION: [
        r"cloudflare",
        r"bot.?detect",
        r"access.?denied",
        r"forbidden",
        r"blocked",
        r"security.?check",
        r"unusual.?traffic",
        r"automated.?access",
        r"robot",
        r"captcha.*required",
    ],
    ErrorType.TIMEOUT: [
        r"timeout",
        r"timed?.?out",
        r"deadline.?exceeded",
        r"connection.?timed",
    ],
    ErrorType.AUTH_REQUIRED: [
        r"login.?required",
        r"authentication.?required",
        r"sign.?in",
        r"unauthorized",
        r"401",
        r"access.?restricted",
    ],
    ErrorType.AUTH_EXPIRED: [
        r"session.?expired",
        r"token.?expired",
        r"cookie.?expired",
        r"re-?authenticate",
    ],
    ErrorType.CAPTCHA: [
        r"captcha",
        r"recaptcha",
        r"hcaptcha",
        r"challenge",
        r"verify.?human",
    ],
    ErrorType.RATE_LIMIT: [
        r"rate.?limit",
        r"too.?many.?requests",
        r"429",
        r"throttl",
        r"slow.?down",
    ],
    ErrorType.NOT_FOUND: [
        r"not.?found",
        r"404",
        r"does.?not.?exist",
        r"no.?such",
        r"invalid.?doi",
    ],
    ErrorType.PAYWALL: [
        r"paywall",
        r"subscription.?required",
        r"purchase",
        r"buy.?access",
        r"institutional.?access",
    ],
    ErrorType.NETWORK: [
        r"network.?error",
        r"connection.?refused",
        r"dns",
        r"unreachable",
        r"connection.?reset",
        r"econnreset",
        r"socket",
    ],
    ErrorType.PDF_INVALID: [
        r"invalid.?pdf",
        r"corrupted",
        r"not.?a.?pdf",
        r"empty.?file",
        r"file.?too.?small",
    ],
    ErrorType.BROWSER_CRASH: [
        r"page.?crash",
        r"browser.?crash",
        r"target.?closed",
        r"context.?closed",
        r"execution.?context",
    ],
}


@dataclass
class StructuredError:
    """Structured error information for job failures."""

    error_type: ErrorType = ErrorType.UNKNOWN
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    screenshot_paths: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    recoverable: bool = True
    suggested_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "details": self.details,
            "screenshot_paths": self.screenshot_paths,
            "timestamp": self.timestamp,
            "recoverable": self.recoverable,
            "suggested_action": self.suggested_action,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredError:
        """Create from dictionary."""
        error_type = data.get("error_type", "unknown")
        if isinstance(error_type, str):
            try:
                error_type = ErrorType(error_type)
            except ValueError:
                error_type = ErrorType.UNKNOWN

        return cls(
            error_type=error_type,
            message=data.get("message", ""),
            details=data.get("details", {}),
            screenshot_paths=data.get("screenshot_paths", []),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            recoverable=data.get("recoverable", True),
            suggested_action=data.get("suggested_action", ""),
        )


def categorize_error(error_message: str) -> ErrorType:
    """Categorize error message into an ErrorType.

    Args:
        error_message: The error message string to categorize

    Returns
    -------
        ErrorType enum value
    """
    if not error_message:
        return ErrorType.UNKNOWN

    error_lower = error_message.lower()

    for error_type, patterns in _ERROR_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, error_lower):
                return error_type

    return ErrorType.UNKNOWN


def get_suggested_action(error_type: ErrorType) -> str:
    """Get suggested action for an error type.

    Args:
        error_type: The categorized error type

    Returns
    -------
        Human-readable suggestion for resolving the error
    """
    suggestions = {
        ErrorType.BOT_DETECTION: "Try using interactive browser mode or wait before retrying",
        ErrorType.TIMEOUT: "Increase timeout or check network connectivity",
        ErrorType.AUTH_REQUIRED: "Re-authenticate using: scitex scholar auth openathens",
        ErrorType.AUTH_EXPIRED: "Session expired. Re-authenticate using: scitex scholar auth openathens",
        ErrorType.CAPTCHA: "Use interactive browser mode to solve CAPTCHA manually",
        ErrorType.RATE_LIMIT: "Wait 5-10 minutes before retrying, reduce parallel workers",
        ErrorType.NOT_FOUND: "Verify DOI/title is correct, try searching with different terms",
        ErrorType.PAYWALL: "Check institutional access or try open access repositories",
        ErrorType.NETWORK: "Check internet connection and try again",
        ErrorType.PDF_INVALID: "PDF may be corrupted at source, try alternative download",
        ErrorType.BROWSER_CRASH: "Restart browser or reduce concurrent downloads",
        ErrorType.UNKNOWN: "Check logs and screenshots for more details",
    }
    return suggestions.get(error_type, suggestions[ErrorType.UNKNOWN])


def is_recoverable(error_type: ErrorType) -> bool:
    """Check if an error type is potentially recoverable with retry.

    Args:
        error_type: The categorized error type

    Returns
    -------
        True if retrying might succeed
    """
    non_recoverable = {
        ErrorType.NOT_FOUND,
        ErrorType.PAYWALL,
        ErrorType.PDF_INVALID,
    }
    return error_type not in non_recoverable


def collect_recent_screenshots(
    since_timestamp: str | None = None,
    screenshot_dir: Path | None = None,
    max_screenshots: int = 10,
    level_filter: list[str] | None = None,
) -> list[str]:
    """Collect recent screenshots from browser logger directory.

    Args:
        since_timestamp: Only collect screenshots after this ISO timestamp
        screenshot_dir: Override default screenshot directory
        max_screenshots: Maximum number of screenshots to collect
        level_filter: Only include screenshots with these levels (e.g., ["ERROR", "WARNING"])

    Returns
    -------
        List of screenshot paths (most recent first)
    """
    if screenshot_dir is None:
        from scitex.config import get_paths

        screenshot_dir = get_paths().resolve("browser_screenshots")

    if not screenshot_dir.exists():
        return []

    # Find all PNG files
    screenshots = list(screenshot_dir.glob("**/*.png"))

    # Filter by timestamp if provided
    if since_timestamp:
        try:
            since_dt = datetime.fromisoformat(since_timestamp)
            screenshots = [
                s for s in screenshots if s.stat().st_mtime >= since_dt.timestamp()
            ]
        except (ValueError, OSError):
            pass

    # Filter by level if provided
    if level_filter:
        level_patterns = [f"-{level.upper()}-" for level in level_filter]
        screenshots = [
            s
            for s in screenshots
            if any(pattern in s.name for pattern in level_patterns)
        ]

    # Sort by modification time (newest first)
    screenshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Limit results
    screenshots = screenshots[:max_screenshots]

    return [str(s) for s in screenshots]


def create_structured_error(
    error: str | Exception,
    doi_or_title: str | None = None,
    url: str | None = None,
    since_timestamp: str | None = None,
    include_screenshots: bool = True,
) -> StructuredError:
    """Create a structured error from an exception or error message.

    Args:
        error: Exception or error message string
        doi_or_title: The DOI or title being processed
        url: The URL being accessed when error occurred
        since_timestamp: Collect screenshots since this timestamp
        include_screenshots: Whether to collect screenshots

    Returns
    -------
        StructuredError with categorization and details
    """
    error_message = str(error)
    error_type = categorize_error(error_message)

    details = {}
    if doi_or_title:
        details["doi_or_title"] = doi_or_title
    if url:
        details["url"] = url
    if isinstance(error, Exception):
        details["exception_type"] = type(error).__name__

    screenshot_paths = []
    if include_screenshots:
        screenshot_paths = collect_recent_screenshots(
            since_timestamp=since_timestamp,
            level_filter=["ERROR", "WARNING", "FAIL"],
            max_screenshots=5,
        )

    return StructuredError(
        error_type=error_type,
        message=error_message,
        details=details,
        screenshot_paths=screenshot_paths,
        recoverable=is_recoverable(error_type),
        suggested_action=get_suggested_action(error_type),
    )


# EOF
