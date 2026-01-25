#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/stats/_mcp/_handlers/_format.py

"""Results formatting handler."""

from __future__ import annotations

import asyncio
from datetime import datetime

__all__ = ["format_results_handler"]


async def format_results_handler(
    test_name: str,
    statistic: float,
    p_value: float,
    df: float | None = None,
    effect_size: float | None = None,
    effect_size_name: str | None = None,
    style: str = "apa",
    ci_lower: float | None = None,
    ci_upper: float | None = None,
) -> dict:
    """Format statistical results in journal style."""
    try:
        loop = asyncio.get_event_loop()

        def do_format():
            from scitex.stats.auto import format_test_line, p_to_stars
            from scitex.stats.auto._formatting import EffectResultDict, TestResultDict

            # Build test result dict
            test_result: TestResultDict = {
                "test_name": test_name,
                "stat": statistic,
                "p_raw": p_value,
            }
            if df is not None:
                test_result["df"] = df

            # Build effect result if provided
            effects = None
            if effect_size is not None:
                effects = [
                    EffectResultDict(
                        name=effect_size_name or "d",
                        label=effect_size_name or "Cohen's d",
                        value=effect_size,
                        ci_lower=ci_lower,
                        ci_upper=ci_upper,
                    )
                ]

            # Map style names
            style_map = {
                "apa": "apa_latex",
                "nature": "nature",
                "science": "science",
                "brief": "brief",
            }
            style_id = style_map.get(style, "apa_latex")

            # Format the line
            formatted = format_test_line(
                test_result,
                effects=effects,
                style=style_id,
                include_n=False,
            )

            # Get stars representation
            stars = p_to_stars(p_value)

            return {
                "formatted": formatted,
                "stars": stars,
            }

        result = await loop.run_in_executor(None, do_format)

        return {
            "success": True,
            "style": style,
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# EOF
