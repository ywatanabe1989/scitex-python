#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-08 15:41:14 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/plt/demo_matplotlib_basic.py


"""Demo: Pure Matplotlib methods through SciTeX wrapper."""

# Imports
import scitex as stx
from scitex.plt.styles.presets import SCITEX_STYLE

from demo_matplotlib_basic import demo_plot
from demo_matplotlib_basic import demo_step
from demo_matplotlib_basic import demo_stem
from demo_matplotlib_basic import demo_scatter
from demo_matplotlib_basic import demo_bar
from demo_matplotlib_basic import demo_barh
from demo_matplotlib_basic import demo_hist
from demo_matplotlib_basic import demo_hist2d
from demo_matplotlib_basic import demo_hexbin
from demo_matplotlib_basic import demo_boxplot
from demo_matplotlib_basic import demo_violinplot
from demo_matplotlib_basic import demo_fill_between
from demo_matplotlib_basic import demo_fill_betweenx
from demo_matplotlib_basic import demo_errorbar
from demo_matplotlib_basic import demo_contour
from demo_matplotlib_basic import demo_contourf
from demo_matplotlib_basic import demo_imshow
from demo_matplotlib_basic import demo_matshow
from demo_matplotlib_basic import demo_pie
from demo_matplotlib_basic import demo_quiver
from demo_matplotlib_basic import demo_streamplot


# Demo registry: (function, filename)
DEMOS = [
    (demo_plot, "./png/01_plot.png"),
    (demo_step, "./png/02_step.png"),
    (demo_stem, "./png/03_stem.png"),
    (demo_scatter, "./png/04_scatter.png"),
    (demo_bar, "./png/05_bar.png"),
    (demo_barh, "./png/06_barh.png"),
    (demo_hist, "./png/07_hist.png"),
    (demo_hist2d, "./png/08_hist2d.png"),
    (demo_hexbin, "./png/09_hexbin.png"),
    (demo_boxplot, "./png/10_boxplot.png"),
    (demo_violinplot, "./png/11_violinplot.png"),
    (demo_fill_between, "./png/12_fill_between.png"),
    (demo_fill_betweenx, "./png/13_fill_betweenx.png"),
    (demo_errorbar, "./png/14_errorbar.png"),
    (demo_contour, "./png/15_contour.png"),
    (demo_contourf, "./png/16_contourf.png"),
    (demo_imshow, "./png/17_imshow.png"),
    (demo_matshow, "./png/18_matshow.png"),
    (demo_pie, "./png/19_pie.png"),
    (demo_quiver, "./png/20_quiver.png"),
    (demo_streamplot, "./png/21_streamplot.png"),
]

# Flat design
DEMOS = [(demo[0], demo[1].replace("./png/", "./")) for demo in DEMOS]


@stx.session
def main(
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Demo: Pure Matplotlib methods through SciTeX wrapper (21 types)."""

    STYLE = SCITEX_STYLE.copy()

    logger.info("=" * 70)
    logger.info("Demo: Pure Matplotlib Methods through SciTeX")
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
