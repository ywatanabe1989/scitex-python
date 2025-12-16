#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_context.py

"""
Statistical Context - Data structure for automatic test selection.

This module defines StatContext which captures all relevant information
about data and experimental design needed to determine which statistical
tests are applicable.

The StatContext is built from figure/axis metadata or directly from data,
and is used by the test selection engine to filter applicable tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Dict, Any

import numpy as np


# =============================================================================
# Type Aliases
# =============================================================================

OutcomeType = Literal["continuous", "ordinal", "binary", "categorical"]
DesignType = Literal["between", "within", "mixed"]


# =============================================================================
# StatContext
# =============================================================================


@dataclass
class StatContext:
    """
    Statistical context for determining which tests are applicable.

    This dataclass captures all the information needed to decide which
    statistical tests can be applied to the current data/figure context.
    It is used by check_applicable() to filter the test registry.

    Parameters
    ----------
    n_groups : int
        Number of groups/levels to compare (e.g., 2 for A vs B).
    sample_sizes : list of int
        Sample sizes per group in the same order as the groups.
    outcome_type : OutcomeType
        Type of outcome variable:
        - "continuous": numeric, interval/ratio scale
        - "ordinal": ordered categories, ranks
        - "binary": 0/1 or yes/no
        - "categorical": nominal with >= 2 categories
    design : DesignType
        Overall experimental design:
        - "between": independent groups
        - "within": repeated measures / paired
        - "mixed": mixed design (some within, some between)
    paired : bool or None
        Whether the comparison is explicitly paired.
        If None, inferred from design.
    has_control_group : bool
        Whether a control group is identifiable (for Dunnett etc.).
    n_factors : int
        Number of factors; 1 for one-way, 2 for two-way, etc.
    normality_ok : bool or None
        Result of normality check if available.
        True = normal, False = non-normal, None = unknown.
    variance_homogeneity_ok : bool or None
        Result of homogeneity test (e.g., Levene).
        True = homogeneous, False = heteroscedastic, None = unknown.
    missing_allowed : bool
        Whether missing data is allowed for the chosen method.
    group_names : list of str, optional
        Names of the groups for display purposes.
    control_group_name : str, optional
        Name of the control group if has_control_group is True.

    Examples
    --------
    >>> # Two-group independent comparison
    >>> ctx = StatContext(
    ...     n_groups=2,
    ...     sample_sizes=[30, 32],
    ...     outcome_type="continuous",
    ...     design="between",
    ...     paired=False,
    ...     has_control_group=False,
    ...     n_factors=1
    ... )

    >>> # Three-group repeated measures
    >>> ctx = StatContext(
    ...     n_groups=3,
    ...     sample_sizes=[20, 20, 20],
    ...     outcome_type="continuous",
    ...     design="within",
    ...     paired=True,
    ...     has_control_group=True,
    ...     n_factors=1,
    ...     control_group_name="baseline"
    ... )

    >>> # Check if data appears normal
    >>> ctx.normality_ok = True
    >>> ctx.variance_homogeneity_ok = True
    """

    # Core group information
    n_groups: int
    sample_sizes: List[int]

    # Data characteristics
    outcome_type: OutcomeType

    # Experimental design
    design: DesignType
    paired: Optional[bool] = None

    # Control group (for Dunnett, etc.)
    has_control_group: bool = False

    # Factor structure (for ANOVA designs)
    n_factors: int = 1

    # Assumption checks (None = not tested)
    normality_ok: Optional[bool] = None
    variance_homogeneity_ok: Optional[bool] = None

    # Missing data handling
    missing_allowed: bool = False

    # Group metadata
    group_names: Optional[List[str]] = None
    control_group_name: Optional[str] = None

    def __post_init__(self):
        """Validate and set defaults."""
        # Ensure sample_sizes matches n_groups
        if len(self.sample_sizes) != self.n_groups:
            raise ValueError(
                f"sample_sizes length ({len(self.sample_sizes)}) must match "
                f"n_groups ({self.n_groups})"
            )

        # Infer paired from design if not specified
        if self.paired is None:
            if self.design == "within":
                self.paired = True
            elif self.design == "between":
                self.paired = False
            # mixed design: leave as None, tests must handle both

        # Default group names if not provided
        if self.group_names is None:
            self.group_names = [f"Group_{i+1}" for i in range(self.n_groups)]

    @property
    def n_total(self) -> int:
        """Total sample size across all groups."""
        return sum(self.sample_sizes)

    @property
    def min_n_per_group(self) -> int:
        """Minimum sample size per group."""
        return min(self.sample_sizes) if self.sample_sizes else 0

    @property
    def effective_paired(self) -> Optional[bool]:
        """
        Effective paired status, considering design.

        Returns
        -------
        bool or None
            True if paired/within, False if unpaired/between, None if unknown.
        """
        if self.paired is not None:
            return self.paired
        if self.design == "within":
            return True
        if self.design == "between":
            return False
        return None  # mixed or unknown

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_groups": self.n_groups,
            "sample_sizes": self.sample_sizes,
            "outcome_type": self.outcome_type,
            "design": self.design,
            "paired": self.paired,
            "has_control_group": self.has_control_group,
            "n_factors": self.n_factors,
            "normality_ok": self.normality_ok,
            "variance_homogeneity_ok": self.variance_homogeneity_ok,
            "missing_allowed": self.missing_allowed,
            "group_names": self.group_names,
            "control_group_name": self.control_group_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatContext":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_data(
        cls,
        y: np.ndarray,
        group: np.ndarray,
        design: DesignType = "between",
        outcome_type: Optional[OutcomeType] = None,
        **kwargs,
    ) -> "StatContext":
        """
        Create StatContext from data arrays.

        Parameters
        ----------
        y : np.ndarray
            Outcome values.
        group : np.ndarray
            Group labels for each observation.
        design : DesignType
            Experimental design.
        outcome_type : OutcomeType, optional
            Type of outcome. If None, inferred from data.
        **kwargs
            Additional arguments for StatContext.

        Returns
        -------
        StatContext
            Context built from the data.

        Examples
        --------
        >>> y = np.array([1, 2, 3, 4, 5, 6])
        >>> group = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        >>> ctx = StatContext.from_data(y, group)
        >>> ctx.n_groups
        2
        """
        y = np.asarray(y)
        group = np.asarray(group)

        # Get unique groups and their sizes
        unique_groups = np.unique(group)
        n_groups = len(unique_groups)
        sample_sizes = [int(np.sum(group == g)) for g in unique_groups]
        group_names = [str(g) for g in unique_groups]

        # Infer outcome type if not provided
        if outcome_type is None:
            outcome_type = _infer_outcome_type(y)

        return cls(
            n_groups=n_groups,
            sample_sizes=sample_sizes,
            outcome_type=outcome_type,
            design=design,
            group_names=group_names,
            **kwargs,
        )


def _infer_outcome_type(y: np.ndarray) -> OutcomeType:
    """
    Infer outcome type from data.

    Parameters
    ----------
    y : np.ndarray
        Outcome values.

    Returns
    -------
    OutcomeType
        Inferred outcome type.
    """
    y = np.asarray(y)

    # Check for categorical first (object dtype like strings)
    if y.dtype == object:
        unique = np.unique(y)
        if len(unique) <= 10:
            return "categorical"
        return "categorical"  # Object arrays are always categorical

    # For numeric arrays, filter out NaN
    try:
        unique = np.unique(y[~np.isnan(y)])
    except TypeError:
        # If isnan fails (e.g., for complex types), use all values
        unique = np.unique(y)

    # Check for binary
    if len(unique) == 2:
        if set(unique).issubset({0, 1}):
            return "binary"

    # Check for categorical (few unique values)
    if len(unique) <= 10 and y.dtype == object:
        return "categorical"

    # Check for ordinal (integer-like with few values)
    if len(unique) <= 20 and np.allclose(y, y.astype(int), equal_nan=True):
        return "ordinal"

    # Default to continuous
    return "continuous"


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "StatContext",
    "OutcomeType",
    "DesignType",
]

# EOF
