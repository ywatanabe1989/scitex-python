#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/stats/_mcp/_handlers/_normality.py

"""Normality test handler."""

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np

__all__ = ["normality_test_handler"]


async def normality_test_handler(
    data: list[float],
    method: str = "shapiro",
) -> dict:
    """Test whether data follows a normal distribution."""
    try:
        from scipy import stats as scipy_stats

        loop = asyncio.get_event_loop()

        def do_normality():
            arr = np.array(data, dtype=float)
            arr = arr[~np.isnan(arr)]

            if len(arr) < 3:
                return {"error": "Need at least 3 data points"}

            result = {}

            if method == "shapiro":
                stat, p_value = scipy_stats.shapiro(arr)
                result = {
                    "test": "Shapiro-Wilk",
                    "statistic": float(stat),
                    "statistic_name": "W",
                    "p_value": float(p_value),
                }

            elif method == "dagostino":
                if len(arr) < 8:
                    return {"error": "D'Agostino test requires at least 8 samples"}
                stat, p_value = scipy_stats.normaltest(arr)
                result = {
                    "test": "D'Agostino-Pearson",
                    "statistic": float(stat),
                    "statistic_name": "K2",
                    "p_value": float(p_value),
                }

            elif method == "anderson":
                res = scipy_stats.anderson(arr, dist="norm")
                # Use 5% significance level
                idx = 2  # Index for 5% level
                result = {
                    "test": "Anderson-Darling",
                    "statistic": float(res.statistic),
                    "statistic_name": "A2",
                    "critical_value_5pct": float(res.critical_values[idx]),
                    "normal": bool(res.statistic < res.critical_values[idx]),
                }

            elif method == "lilliefors":
                try:
                    from statsmodels.stats.diagnostic import lilliefors

                    stat, p_value = lilliefors(arr, dist="norm")
                    result = {
                        "test": "Lilliefors",
                        "statistic": float(stat),
                        "statistic_name": "D",
                        "p_value": float(p_value),
                    }
                except ImportError:
                    return {"error": "statsmodels required for Lilliefors test"}

            else:
                raise ValueError(f"Unknown method: {method}")

            # Add interpretation
            if "p_value" in result:
                result["is_normal"] = result["p_value"] >= 0.05
                result["interpretation"] = (
                    "Data appears normally distributed (p >= 0.05)"
                    if result["is_normal"]
                    else "Data deviates from normal distribution (p < 0.05)"
                )

            return result

        result = await loop.run_in_executor(None, do_normality)

        return {
            "success": True,
            "method": method,
            "n": len(data),
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# EOF
