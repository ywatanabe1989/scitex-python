#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/stats/_mcp/_handlers/_stars.py

"""P-value to stars conversion handler."""

from __future__ import annotations

from datetime import datetime

__all__ = ["p_to_stars_handler"]


async def p_to_stars_handler(
    p_value: float,
    thresholds: list[float] | None = None,
) -> dict:
    """Convert p-value to significance stars."""
    try:
        thresh = thresholds or [0.001, 0.01, 0.05]

        if p_value < thresh[0]:
            stars = "***"
            significance = f"p < {thresh[0]}"
        elif p_value < thresh[1]:
            stars = "**"
            significance = f"p < {thresh[1]}"
        elif p_value < thresh[2]:
            stars = "*"
            significance = f"p < {thresh[2]}"
        else:
            stars = "ns"
            significance = f"p >= {thresh[2]} (not significant)"

        return {
            "success": True,
            "p_value": p_value,
            "stars": stars,
            "significance": significance,
            "thresholds": thresh,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# EOF
