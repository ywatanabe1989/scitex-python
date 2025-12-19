#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/examples/fig/11_stats_annotation_binding.py

"""
Example 11: Stats to Annotation Binding

Demonstrates:
- stats/stats.json drives visual annotations
- result_id links statistical results to visual elements
- Significance stars (*, **, ***) derived from p-value
- Annotation updates automatically when stats change
"""

import json

import numpy as np
from scipy import stats as scipy_stats

import scitex as stx
from scitex import INJECTED
from scitex.fig import Figz


def get_sig_stars(p_value: float) -> str:
    """Convert p-value to significance stars."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "n.s."


@stx.session(verbose=True, agg=True)
def main(plt=INJECTED, CONFIG=INJECTED, logger=INJECTED):
    """Demonstrate stats-to-annotation binding."""
    logger.info("Example 11: Stats Annotation Binding Demo")

    out_dir = CONFIG["SDIR_OUT"]

    # Generate sample data (two groups)
    np.random.seed(42)
    group_ctrl = np.random.normal(5.0, 1.0, 30)
    group_treat = np.random.normal(7.0, 1.2, 30)

    # Compute statistics
    t_stat, p_value = scipy_stats.ttest_ind(group_ctrl, group_treat)
    effect_size = (np.mean(group_treat) - np.mean(group_ctrl)) / np.std(group_ctrl)

    logger.info(f"Statistical test: t={t_stat:.3f}, p={p_value:.6f}")
    logger.info(f"Effect size (Cohen's d): {effect_size:.3f}")
    logger.info(f"Significance: {get_sig_stars(p_value)}")

    # Create figure
    fig = Figz(
        out_dir / "stats_binding.stx.d",
        name="Stats Binding Demo",
        size_mm={"width": 120, "height": 80},
    )

    # Create bar plot with error bars
    fig_a, ax = plt.subplots(figsize=(4, 3))

    means = [np.mean(group_ctrl), np.mean(group_treat)]
    stds = [np.std(group_ctrl), np.std(group_treat)]
    x_pos = [0, 1]
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=["#4A90D9", "#D94A4A"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Control", "Treatment"])
    ax.set_ylabel("Value")
    ax.set_title("Group Comparison")

    # Add significance bracket manually
    y_max = max(means) + max(stds) + 0.5
    ax.plot([0, 0, 1, 1], [y_max, y_max + 0.3, y_max + 0.3, y_max], "k-", lw=1.2)
    ax.text(
        0.5, y_max + 0.4, get_sig_stars(p_value), ha="center", va="bottom", fontsize=14
    )

    fig.add_element(
        "plot_A",
        "plot",
        fig_a,
        position={"x_mm": 10, "y_mm": 5},
        size={"width_mm": 100, "height_mm": 70},
    )
    plt.close(fig_a)

    fig.set_panel_info(
        "plot_A", panel_letter="A", description="Group comparison with stats"
    )

    # Save figure first to create directory structure
    fig.save()

    # Write stats.json with result_id binding
    stats_data = {
        "schema": {"name": "scitex.stats", "version": "1.0.0"},
        "tests": [
            {
                "result_id": "ttest_ctrl_vs_treat",
                "test_type": "independent_t_test",
                "groups": ["Control", "Treatment"],
                "statistic": round(t_stat, 4),
                "p_value": round(p_value, 6),
                "effect_size": round(effect_size, 4),
                "effect_type": "cohens_d",
                "n_samples": [len(group_ctrl), len(group_treat)],
                "significance": get_sig_stars(p_value),
            }
        ],
        "bindings": [
            {
                "result_id": "ttest_ctrl_vs_treat",
                "target_element": "plot_A",
                "annotation_type": "significance_bracket",
                "position": {"x1": 0, "x2": 1, "y": "auto"},
            }
        ],
    }

    stats_path = fig.path / "stats" / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=2)

    logger.info(f"\nSaved stats to: {stats_path}")

    # === Demo: Show binding ===
    logger.info("\n" + "=" * 60)
    logger.info("STATS → ANNOTATION BINDING")
    logger.info("=" * 60)

    logger.info("\nstats/stats.json contents:")
    logger.info(f"  result_id: {stats_data['tests'][0]['result_id']}")
    logger.info(f"  test_type: {stats_data['tests'][0]['test_type']}")
    logger.info(f"  p_value: {stats_data['tests'][0]['p_value']}")
    logger.info(f"  significance: {stats_data['tests'][0]['significance']}")

    logger.info("\nBinding:")
    binding = stats_data["bindings"][0]
    logger.info(f"  result_id: {binding['result_id']}")
    logger.info(f"  target_element: {binding['target_element']}")
    logger.info(f"  annotation_type: {binding['annotation_type']}")

    logger.info("\n" + "-" * 40)
    logger.info("How it works:")
    logger.info("  1. stats.json stores result_id with p-value")
    logger.info("  2. Binding links result_id to target element")
    logger.info("  3. Renderer reads binding and adds annotation")
    logger.info("  4. Change data → recompute stats → stars update")

    # Simulate data change
    logger.info("\n" + "-" * 40)
    logger.info("Simulation: If p-value were different...")
    for sim_p in [0.04, 0.008, 0.0005]:
        logger.info(f"  p={sim_p:.4f} → {get_sig_stars(sim_p)}")

    logger.info("\n" + "=" * 60)
    logger.success("Example 11 completed!")


if __name__ == "__main__":
    main()

# EOF
