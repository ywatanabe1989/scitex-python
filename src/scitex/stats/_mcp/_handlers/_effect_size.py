#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/stats/_mcp/_handlers/_effect_size.py

"""Effect size calculation handler."""

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np

__all__ = ["effect_size_handler"]


async def effect_size_handler(
    group1: list[float],
    group2: list[float],
    measure: str = "cohens_d",
    pooled: bool = True,
) -> dict:
    """Calculate effect size between groups."""
    try:
        from scitex.stats.effect_sizes import (
            cliffs_delta,
            cohens_d,
            interpret_cliffs_delta,
            interpret_cohens_d,
        )

        loop = asyncio.get_event_loop()

        def do_effect_size():
            g1 = np.array(group1, dtype=float)
            g2 = np.array(group2, dtype=float)

            result = {}

            if measure == "cohens_d":
                d = cohens_d(g1, g2)
                result = {
                    "measure": "Cohen's d",
                    "value": float(d),
                    "interpretation": interpret_cohens_d(d),
                }

            elif measure == "hedges_g":
                # Hedges' g is Cohen's d with bias correction
                d = cohens_d(g1, g2)
                n1, n2 = len(g1), len(g2)
                correction = 1 - (3 / (4 * (n1 + n2) - 9))
                g = d * correction
                result = {
                    "measure": "Hedges' g",
                    "value": float(g),
                    "interpretation": interpret_cohens_d(g),  # Same thresholds
                }

            elif measure == "glass_delta":
                # Glass's delta uses only control group std
                mean_diff = np.mean(g1) - np.mean(g2)
                delta = mean_diff / np.std(g2, ddof=1)
                result = {
                    "measure": "Glass's delta",
                    "value": float(delta),
                    "interpretation": interpret_cohens_d(delta),
                }

            elif measure == "cliffs_delta":
                delta = cliffs_delta(g1, g2)
                result = {
                    "measure": "Cliff's delta",
                    "value": float(delta),
                    "interpretation": interpret_cliffs_delta(delta),
                }

            else:
                raise ValueError(f"Unknown measure: {measure}")

            # Add confidence interval approximation for Cohen's d
            if measure in ["cohens_d", "hedges_g", "glass_delta"]:
                n1, n2 = len(g1), len(g2)
                se = np.sqrt(
                    (n1 + n2) / (n1 * n2) + result["value"] ** 2 / (2 * (n1 + n2))
                )
                result["ci_lower"] = float(result["value"] - 1.96 * se)
                result["ci_upper"] = float(result["value"] + 1.96 * se)

            return result

        result = await loop.run_in_executor(None, do_effect_size)

        return {
            "success": True,
            "group1_n": len(group1),
            "group2_n": len(group2),
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# EOF
