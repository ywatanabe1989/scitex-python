#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_summary.py

"""
Summary Statistics - Per-group summary computation.

This module provides functions for computing descriptive statistics
for individual groups or samples.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

import numpy as np


class SummaryStatsDict(TypedDict, total=False):
    """Summary statistics for a single group."""

    group: str
    n: int
    mean: Optional[float]
    sd: Optional[float]
    sem: Optional[float]
    median: Optional[float]
    iqr: Optional[float]
    q1: Optional[float]
    q3: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]


def compute_summary_stats(
    y: np.ndarray,
    group: np.ndarray,
) -> List[SummaryStatsDict]:
    """
    Compute per-group summary statistics.

    Parameters
    ----------
    y : np.ndarray
        Outcome values.
    group : np.ndarray
        Group labels for each observation.

    Returns
    -------
    list of SummaryStatsDict
        Summary statistics for each group.

    Examples
    --------
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> group = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
    >>> stats = compute_summary_stats(y, group)
    >>> stats[0]['group']
    'A'
    >>> stats[0]['n']
    3
    """
    y = np.asarray(y, dtype=float)
    group = np.asarray(group)

    stats_list: List[SummaryStatsDict] = []

    for group_value in np.unique(group):
        mask = group == group_value
        vals = y[mask]
        vals = vals[~np.isnan(vals)]  # Remove NaN

        if vals.size == 0:
            continue

        n = int(vals.size)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else 0.0
        sem = float(sd / np.sqrt(n)) if n > 1 else 0.0

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1

        stats_list.append(
            SummaryStatsDict(
                group=str(group_value),
                n=n,
                mean=mean,
                sd=sd,
                sem=sem,
                median=float(med),
                iqr=float(iqr),
                q1=float(q1),
                q3=float(q3),
                minimum=float(vals.min()),
                maximum=float(vals.max()),
            )
        )

    return stats_list


def compute_summary_from_groups(
    groups: List[np.ndarray],
    group_names: Optional[List[str]] = None,
) -> List[SummaryStatsDict]:
    """
    Compute summary statistics from a list of group arrays.

    Parameters
    ----------
    groups : list of np.ndarray
        List of arrays, one per group.
    group_names : list of str, optional
        Names for each group.

    Returns
    -------
    list of SummaryStatsDict
        Summary statistics for each group.
    """
    if group_names is None:
        group_names = [f"Group_{i+1}" for i in range(len(groups))]

    stats_list: List[SummaryStatsDict] = []

    for name, vals in zip(group_names, groups):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]

        if vals.size == 0:
            continue

        n = int(vals.size)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else 0.0
        sem = float(sd / np.sqrt(n)) if n > 1 else 0.0

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1

        stats_list.append(
            SummaryStatsDict(
                group=name,
                n=n,
                mean=mean,
                sd=sd,
                sem=sem,
                median=float(med),
                iqr=float(iqr),
                q1=float(q1),
                q3=float(q3),
                minimum=float(vals.min()),
                maximum=float(vals.max()),
            )
        )

    return stats_list


# =============================================================================


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "compute_summary_stats",
    "compute_summary_from_groups",
]

# EOF
