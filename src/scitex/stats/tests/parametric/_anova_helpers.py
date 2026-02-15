#!/usr/bin/env python3
"""Shared helper functions for ANOVA test modules."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats


def partial_eta_squared(ss_effect: float, ss_error: float) -> float:
    """Compute partial eta-squared."""
    return ss_effect / (ss_effect + ss_error)


def interpret_eta_squared(eta2: float) -> str:
    """Interpret eta-squared effect size."""
    if eta2 < 0.01:
        return "negligible"
    elif eta2 < 0.06:
        return "small"
    elif eta2 < 0.14:
        return "medium"
    else:
        return "large"


def mauchly_sphericity(data: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute Mauchly's test of sphericity.

    Parameters
    ----------
    data : array, shape (n_subjects, n_conditions)
        Data matrix

    Returns
    -------
    W : float
        Mauchly's W statistic
    chi2 : float
        Chi-square statistic
    pvalue : float
        p-value

    Notes
    -----
    Tests whether the variances of differences between conditions are equal.
    If p < 0.05, sphericity is violated.
    """
    n, k = data.shape

    # Compute difference matrix
    diffs = []
    for i in range(k):
        for j in range(i + 1, k):
            diffs.append(data[:, i] - data[:, j])

    diff_matrix = np.array(diffs).T  # shape: (n_subjects, n_pairs)

    # Covariance matrix of differences
    S = np.cov(diff_matrix, rowvar=False)

    # Mauchly's W statistic
    W = np.linalg.det(S) / (np.trace(S) / S.shape[0]) ** S.shape[0]

    # Chi-square approximation
    df = k * (k - 1) / 2 - 1
    chi2 = -(n - 1 - (2 * k**2 - 3 * k + 3) / (6 * (k - 1))) * np.log(W)
    pvalue = 1 - stats.chi2.cdf(chi2, df)

    return float(W), float(chi2), float(pvalue)


def greenhouse_geisser_epsilon(data: np.ndarray) -> float:
    """
    Compute Greenhouse-Geisser epsilon correction factor.

    Parameters
    ----------
    data : array, shape (n_subjects, n_conditions)
        Data matrix

    Returns
    -------
    epsilon : float
        GG epsilon (between 1/(k-1) and 1.0)

    Notes
    -----
    Used to correct degrees of freedom when sphericity is violated.
    epsilon = 1.0 indicates perfect sphericity.
    """
    n, k = data.shape

    # Compute covariance matrix
    centered = data - data.mean(axis=1, keepdims=True)
    S = np.dot(centered.T, centered) / (n - 1)

    # Compute epsilon
    trace_S = np.trace(S)
    trace_S2 = np.trace(np.dot(S, S))

    numerator = (k * trace_S) ** 2
    denominator = (k - 1) * (k * trace_S2 - trace_S**2)

    if denominator == 0:
        return 1.0

    epsilon = numerator / denominator

    # Bound epsilon
    epsilon = max(1.0 / (k - 1), min(epsilon, 1.0))

    return float(epsilon)


# EOF
