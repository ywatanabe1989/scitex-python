#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 20:47:36 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/plt/demo_seaborn_wrappers.py


"""Demo: Seaborn wrapper methods (ax.sns_xxx)."""

# Imports
import scitex as stx
from scitex.plt.styles.presets import SCITEX_STYLE

from demo_seaborn_wrappers import demo_sns_boxplot
from demo_seaborn_wrappers import demo_sns_violinplot
from demo_seaborn_wrappers import demo_sns_scatterplot
from demo_seaborn_wrappers import demo_sns_lineplot
from demo_seaborn_wrappers import demo_sns_histplot
from demo_seaborn_wrappers import demo_sns_kdeplot
from demo_seaborn_wrappers import demo_sns_barplot
from demo_seaborn_wrappers import demo_sns_stripplot
from demo_seaborn_wrappers import demo_sns_swarmplot
from demo_seaborn_wrappers import demo_sns_heatmap


# Demo registry: (function, filename)
DEMOS = [
    (demo_sns_boxplot, "./png/01_sns_boxplot.png"),
    (demo_sns_violinplot, "./png/02_sns_violinplot.png"),
    (demo_sns_scatterplot, "./png/03_sns_scatterplot.png"),
    (demo_sns_lineplot, "./png/04_sns_lineplot.png"),
    (demo_sns_histplot, "./png/05_sns_histplot.png"),
    (demo_sns_kdeplot, "./png/06_sns_kdeplot.png"),
    (demo_sns_barplot, "./png/07_sns_barplot.png"),
    (demo_sns_stripplot, "./png/08_sns_stripplot.png"),
    (demo_sns_swarmplot, "./png/09_sns_swarmplot.png"),
    (demo_sns_heatmap, "./png/10_sns_heatmap.png"),
]


@stx.session
def main(
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Demo: Seaborn wrapper methods (10 types)."""
    STYLE = SCITEX_STYLE.copy()

    logger.info("=" * 70)
    logger.info("Demo: Seaborn Wrapper Methods (ax.sns_xxx)")
    logger.info("=" * 70)

    for idx, (demo_func, filename) in enumerate(DEMOS, 1):
        logger.info(f"\n[{idx:02d}] {demo_func.__doc__}")

        fig, ax = stx.plt.subplots(**STYLE)
        demo_func(fig, ax, stx)
        stx.io.save(fig, filename)
        fig.close()

    logger.info("\n" + "=" * 70)
    logger.info("All demos completed")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    main()

# EOF
