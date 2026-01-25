#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/stats/_mcp/_handlers/_power.py

"""Power analysis handler."""

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np

__all__ = ["power_analysis_handler"]


async def power_analysis_handler(
    test_type: str = "ttest",
    effect_size: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    n: int | None = None,
    n_groups: int = 2,
    ratio: float = 1.0,
) -> dict:
    """Calculate statistical power or required sample size."""
    try:
        loop = asyncio.get_event_loop()

        def do_power():
            from scitex.stats.power._power import power_ttest, sample_size_ttest

            result = {}

            if test_type == "ttest":
                if n is not None and effect_size is not None:
                    # Calculate power given n and effect size
                    calculated_power = power_ttest(
                        effect_size=effect_size,
                        n1=n,
                        n2=int(n * ratio),
                        alpha=alpha,
                        test_type="two-sample",
                    )
                    result = {
                        "mode": "power_calculation",
                        "power": calculated_power,
                        "n1": n,
                        "n2": int(n * ratio),
                        "effect_size": effect_size,
                        "alpha": alpha,
                    }
                elif effect_size is not None:
                    # Calculate required sample size
                    n1, n2 = sample_size_ttest(
                        effect_size=effect_size,
                        power=power,
                        alpha=alpha,
                        ratio=ratio,
                    )
                    result = {
                        "mode": "sample_size_calculation",
                        "required_n1": n1,
                        "required_n2": n2,
                        "total_n": n1 + n2,
                        "effect_size": effect_size,
                        "target_power": power,
                        "alpha": alpha,
                    }
                else:
                    raise ValueError("Either n or effect_size must be provided")

            elif test_type == "anova":
                result = _power_anova(effect_size, alpha, power, n, n_groups)

            elif test_type == "correlation":
                result = _power_correlation(effect_size, alpha, power, n)

            elif test_type == "chi2":
                result = _power_chi2(effect_size, alpha, power, n, n_groups)

            else:
                raise ValueError(f"Unknown test_type: {test_type}")

            return result

        result = await loop.run_in_executor(None, do_power)

        return {
            "success": True,
            "test_type": test_type,
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def _power_anova(
    effect_size: float | None,
    alpha: float,
    power: float,
    n: int | None,
    n_groups: int,
) -> dict:
    """ANOVA power calculation."""
    from scipy import stats as scipy_stats

    if effect_size is None:
        raise ValueError("effect_size required for ANOVA power")

    if n is not None:
        df1 = n_groups - 1
        df2 = n_groups * n - n_groups
        nc = effect_size**2 * n * n_groups
        f_crit = scipy_stats.f.ppf(1 - alpha, df1, df2)
        power_val = 1 - scipy_stats.ncf.cdf(f_crit, df1, df2, nc)
        return {
            "mode": "power_calculation",
            "power": power_val,
            "n_per_group": n,
            "n_groups": n_groups,
            "effect_size_f": effect_size,
            "alpha": alpha,
        }
    else:
        # Binary search for n
        n_min, n_max = 2, 1000
        while n_max - n_min > 1:
            n_mid = (n_min + n_max) // 2
            df1 = n_groups - 1
            df2 = n_groups * n_mid - n_groups
            nc = effect_size**2 * n_mid * n_groups
            f_crit = scipy_stats.f.ppf(1 - alpha, df1, df2)
            power_val = 1 - scipy_stats.ncf.cdf(f_crit, df1, df2, nc)
            if power_val < power:
                n_min = n_mid
            else:
                n_max = n_mid

        return {
            "mode": "sample_size_calculation",
            "required_n_per_group": n_max,
            "total_n": n_max * n_groups,
            "n_groups": n_groups,
            "effect_size_f": effect_size,
            "target_power": power,
            "alpha": alpha,
        }


def _power_correlation(
    effect_size: float | None,
    alpha: float,
    power: float,
    n: int | None,
) -> dict:
    """Correlation power calculation."""
    from scipy import stats as scipy_stats

    if effect_size is None:
        raise ValueError("effect_size (r) required for correlation power")

    if n is not None:
        # Calculate power
        z = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
        se = 1 / np.sqrt(n - 3)
        z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
        power_val = (
            1
            - scipy_stats.norm.cdf(z_crit - z / se)
            + scipy_stats.norm.cdf(-z_crit - z / se)
        )
        return {
            "mode": "power_calculation",
            "power": power_val,
            "n": n,
            "effect_size_r": effect_size,
            "alpha": alpha,
        }
    else:
        # Calculate required n
        z = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
        z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
        z_power = scipy_stats.norm.ppf(power)
        required_n = int(np.ceil(((z_crit + z_power) / z) ** 2 + 3))
        return {
            "mode": "sample_size_calculation",
            "required_n": required_n,
            "effect_size_r": effect_size,
            "target_power": power,
            "alpha": alpha,
        }


def _power_chi2(
    effect_size: float | None,
    alpha: float,
    power: float,
    n: int | None,
    n_groups: int,
) -> dict:
    """Chi-square power calculation."""
    from scipy import stats as scipy_stats

    if effect_size is None:
        raise ValueError("effect_size (w) required for chi2 power")

    df = n_groups - 1  # Simplified: using n_groups as number of cells

    if n is not None:
        nc = effect_size**2 * n
        chi2_crit = scipy_stats.chi2.ppf(1 - alpha, df)
        power_val = 1 - scipy_stats.ncx2.cdf(chi2_crit, df, nc)
        return {
            "mode": "power_calculation",
            "power": power_val,
            "n": n,
            "df": df,
            "effect_size_w": effect_size,
            "alpha": alpha,
        }
    else:
        # Binary search for n
        n_min, n_max = 10, 10000
        while n_max - n_min > 1:
            n_mid = (n_min + n_max) // 2
            nc = effect_size**2 * n_mid
            chi2_crit = scipy_stats.chi2.ppf(1 - alpha, df)
            power_val = 1 - scipy_stats.ncx2.cdf(chi2_crit, df, nc)
            if power_val < power:
                n_min = n_mid
            else:
                n_max = n_mid

        return {
            "mode": "sample_size_calculation",
            "required_n": n_max,
            "df": df,
            "effect_size_w": effect_size,
            "target_power": power,
            "alpha": alpha,
        }


# EOF
