#!/usr/bin/env python3
"""Plot function for two-way ANOVA results."""

from __future__ import annotations

import matplotlib.pyplot as plt  # noqa: STX-I001
import numpy as np

from scitex.stats._utils._formatters import fmt_stat, fmt_sym


def _plot_anova_2way(results_dict):
    """Create visualization for two-way ANOVA."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    cell_means = results_dict["cell_means"]
    a_levels = results_dict["a_levels"]
    b_levels = results_dict["b_levels"]
    a_marginal = results_dict["a_marginal_means"]
    b_marginal = results_dict["b_marginal_means"]
    factor_a_name = results_dict["factor_a_name"]
    factor_b_name = results_dict["factor_b_name"]
    effects = results_dict["effects"]

    # Panel 1: Interaction plot (A on x-axis, lines for B)
    ax = axes[0, 0]
    x_pos = np.arange(len(a_levels))

    for bi, b_level in enumerate(b_levels):
        ax.plot(
            x_pos,
            cell_means[:, bi],
            marker="o",
            label=str(b_level),
            linewidth=2,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(level) for level in a_levels])
    ax.set_xlabel(factor_a_name, fontsize=12)
    ax.set_ylabel("Mean", fontsize=12)
    ax.set_title("Two-way ANOVA", fontsize=12, fontweight="bold")
    ax.legend(title=factor_b_name, loc="best")
    ax.grid(True, alpha=0.3)

    # Stats text box - top-left corner (Interaction effect)
    effect_ab = effects[2]
    df_ab = f"{effect_ab['df_effect']}, {effect_ab['df_error']}"
    text_str = (
        f"{fmt_stat('F', effect_ab['statistic'], df=df_ab)}\n"
        f"{fmt_stat('p', effect_ab['pvalue'], fmt='.4f', stars=effect_ab['stars'])}\n"
        f"{fmt_sym('eta_p^2')} = {effect_ab['effect_size']:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        text_str,
        transform=ax.transAxes,
        verticalalignment="top",
        color="black",
        fontsize=6,
    )

    # Panel 2: Interaction plot (B on x-axis, lines for A)
    ax = axes[0, 1]
    x_pos = np.arange(len(b_levels))

    for ai, a_level in enumerate(a_levels):
        ax.plot(
            x_pos,
            cell_means[ai, :],
            marker="s",
            label=str(a_level),
            linewidth=2,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(level) for level in b_levels])
    ax.set_xlabel(factor_b_name, fontsize=12)
    ax.set_ylabel("Mean", fontsize=12)
    ax.set_title(
        f"Interaction Plot: {factor_b_name} × {factor_a_name}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(title=factor_a_name, loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 3: Marginal means for Factor A
    ax = axes[1, 0]
    ax.bar(
        range(len(a_levels)),
        a_marginal,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )

    ax.set_xticks(range(len(a_levels)))
    ax.set_xticklabels([str(level) for level in a_levels])
    ax.set_xlabel(factor_a_name, fontsize=12)
    ax.set_ylabel("Marginal Mean", fontsize=12)
    ax.set_title(f"Main Effect: {factor_a_name}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Stats text box for Factor A
    effect_a = effects[0]
    df_a = f"{effect_a['df_effect']}, {effect_a['df_error']}"
    text_str_a = (
        f"{fmt_stat('F', effect_a['statistic'], df=df_a)}\n"
        f"{fmt_stat('p', effect_a['pvalue'], fmt='.4f', stars=effect_a['stars'])}\n"
        f"{fmt_sym('eta_p^2')} = {effect_a['effect_size']:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        text_str_a,
        transform=ax.transAxes,
        verticalalignment="top",
        color="black",
        fontsize=6,
    )

    # Panel 4: Marginal means for Factor B
    ax = axes[1, 1]
    ax.bar(
        range(len(b_levels)),
        b_marginal,
        color="coral",
        alpha=0.7,
        edgecolor="black",
    )

    ax.set_xticks(range(len(b_levels)))
    ax.set_xticklabels([str(level) for level in b_levels])
    ax.set_xlabel(factor_b_name, fontsize=12)
    ax.set_ylabel("Marginal Mean", fontsize=12)
    ax.set_title(f"Main Effect: {factor_b_name}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Stats text box for Factor B
    effect_b = effects[1]
    df_b = f"{effect_b['df_effect']}, {effect_b['df_error']}"
    text_str_b = (
        f"{fmt_stat('F', effect_b['statistic'], df=df_b)}\n"
        f"{fmt_stat('p', effect_b['pvalue'], fmt='.4f', stars=effect_b['stars'])}\n"
        f"{fmt_sym('eta_p^2')} = {effect_b['effect_size']:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        text_str_b,
        transform=ax.transAxes,
        verticalalignment="top",
        color="black",
        fontsize=6,
    )

    plt.tight_layout()
    return fig


# EOF
