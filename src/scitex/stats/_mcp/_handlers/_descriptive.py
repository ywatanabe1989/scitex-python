#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/stats/_mcp/_handlers/_descriptive.py

"""Descriptive statistics handler."""

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np

__all__ = ["describe_handler"]


async def describe_handler(
    data: list[float],
    percentiles: list[float] | None = None,
) -> dict:
    """Calculate descriptive statistics for data."""
    try:
        loop = asyncio.get_event_loop()

        def do_describe():
            arr = np.array(data, dtype=float)
            arr = arr[~np.isnan(arr)]  # Remove NaN

            if len(arr) == 0:
                return {"error": "No valid data points"}

            percs = percentiles or [25, 50, 75]
            percentile_values = np.percentile(arr, percs)

            result = {
                "n": int(len(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "var": float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "sem": (
                    float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
                    if len(arr) > 1
                    else 0.0
                ),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "range": float(np.max(arr) - np.min(arr)),
                "median": float(np.median(arr)),
                "percentiles": {
                    str(int(p)): float(v) for p, v in zip(percs, percentile_values)
                },
                "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            }

            # Add skewness and kurtosis if scipy available
            try:
                from scipy import stats as scipy_stats

                result["skewness"] = float(scipy_stats.skew(arr))
                result["kurtosis"] = float(scipy_stats.kurtosis(arr))
            except ImportError:
                pass

            return result

        result = await loop.run_in_executor(None, do_describe)

        return {
            "success": True,
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# EOF
