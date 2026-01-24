#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/stats/auto/_stat_style.py

"""StatStyle dataclass for statistical reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

OutputTarget = Literal["latex", "html", "plain"]

__all__ = ["StatStyle", "OutputTarget"]


@dataclass
class StatStyle:
    """Style configuration for statistical reporting.

    Defines how to format statistical results for a specific journal
    or output format.

    Parameters
    ----------
    id : str
        Unique identifier for this style.
    label : str
        Human-readable label (e.g., "APA (LaTeX)").
    target : OutputTarget
        Output format: "latex", "html", or "plain".
    stat_symbol_format : dict
        Maps statistic symbols to their formatted versions.
    p_format : str
        Format string for p-values.
    alpha_thresholds : list of (float, str)
        P-value thresholds for stars.
    effect_label_format : dict
        Maps effect size names to their formatted labels.
    n_format : str
        Format string for sample sizes.
    decimal_places_p : int
        Decimal places for p-values.
    decimal_places_stat : int
        Decimal places for test statistics.
    decimal_places_effect : int
        Decimal places for effect sizes.
    """

    id: str
    label: str
    target: OutputTarget

    stat_symbol_format: Dict[str, str] = field(default_factory=dict)
    p_format: str = "p = {p:.3f}"
    alpha_thresholds: List[Tuple[float, str]] = field(default_factory=list)
    effect_label_format: Dict[str, str] = field(default_factory=dict)
    n_format: str = "n_{%s} = %d"

    decimal_places_p: int = 3
    decimal_places_stat: int = 2
    decimal_places_effect: int = 2

    def format_stat(
        self,
        symbol: str,
        value: float,
        df: Optional[float] = None,
    ) -> str:
        """Format a test statistic.

        Parameters
        ----------
        symbol : str
            Statistic symbol (e.g., "t", "F", "chi2").
        value : float
            Statistic value.
        df : float, optional
            Degrees of freedom.

        Returns
        -------
        str
            Formatted statistic string.
        """
        fmt_symbol = self.stat_symbol_format.get(symbol, symbol)
        dp = self.decimal_places_stat

        if df is not None:
            return f"{fmt_symbol}({df:.1f}) = {value:.{dp}f}"
        return f"{fmt_symbol} = {value:.{dp}f}"

    def format_p(self, p_value: float) -> str:
        """Format a p-value.

        Parameters
        ----------
        p_value : float
            P-value to format.

        Returns
        -------
        str
            Formatted p-value string.
        """
        p_symbol = self.stat_symbol_format.get("p", "p")
        dp = self.decimal_places_p

        if p_value < 0.001:
            return f"{p_symbol} < 0.001"
        if p_value < 0.0001:
            return f"{p_symbol} < 0.0001"
        return f"{p_symbol} = {p_value:.{dp}f}"

    def format_effect(self, name: str, value: float) -> str:
        """Format an effect size.

        Parameters
        ----------
        name : str
            Effect size name (e.g., "cohens_d_ind").
        value : float
            Effect size value.

        Returns
        -------
        str
            Formatted effect size string.
        """
        label = self.effect_label_format.get(name, name)
        dp = self.decimal_places_effect
        return f"{label} = {value:.{dp}f}"

    def format_n(self, group: str, n: int) -> str:
        """Format a sample size.

        Parameters
        ----------
        group : str
            Group name/label.
        n : int
            Sample size.

        Returns
        -------
        str
            Formatted sample size string.
        """
        if "%s" in self.n_format:
            return self.n_format % (group, n)
        return self.n_format % n

    def p_to_stars(self, p_value: float) -> str:
        """Convert p-value to significance stars.

        Parameters
        ----------
        p_value : float
            P-value.

        Returns
        -------
        str
            Stars string ("***", "**", "*", or "ns").
        """
        if p_value is None:
            return "ns"

        for threshold, stars in self.alpha_thresholds:
            if p_value < threshold:
                return stars
        return "ns"


# EOF
