#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 20:47:26 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/plt/demo_scitex_wrappers.py


"""Demo: SciTeX custom wrapper methods (ax.plot_xxx)."""

# Imports
import scitex as stx
from scitex.plt.styles.presets import SCITEX_STYLE

from demo_scitex_wrappers import demo_plot_line
from demo_scitex_wrappers import demo_plot_shaded_line
from demo_scitex_wrappers import demo_plot_mean_std
from demo_scitex_wrappers import demo_plot_mean_ci
from demo_scitex_wrappers import demo_plot_median_iqr
from demo_scitex_wrappers import demo_plot_kde
from demo_scitex_wrappers import demo_plot_ecdf
from demo_scitex_wrappers import demo_plot_box
from demo_scitex_wrappers import demo_plot_violin
from demo_scitex_wrappers import demo_plot_bar
from demo_scitex_wrappers import demo_plot_barh
from demo_scitex_wrappers import demo_plot_scatter
from demo_scitex_wrappers import demo_plot_errorbar
from demo_scitex_wrappers import demo_plot_fill_between
from demo_scitex_wrappers import demo_plot_fillv
from demo_scitex_wrappers import demo_plot_contour
from demo_scitex_wrappers import demo_plot_imshow
from demo_scitex_wrappers import demo_plot_image
from demo_scitex_wrappers import demo_plot_heatmap
from demo_scitex_wrappers import demo_plot_conf_mat
from demo_scitex_wrappers import demo_plot_boxplot
from demo_scitex_wrappers import demo_plot_violinplot
from demo_scitex_wrappers import demo_plot_raster
from demo_scitex_wrappers import demo_plot_joyplot
from demo_scitex_wrappers import demo_plot_rectangle


# Demo registry: (function, filename)
DEMOS = [
    (demo_plot_line, "./png/01_plot_line.png"),
    (demo_plot_shaded_line, "./png/02_plot_shaded_line.png"),
    (demo_plot_mean_std, "./png/03_plot_mean_std.png"),
    (demo_plot_mean_ci, "./png/04_plot_mean_ci.png"),
    (demo_plot_median_iqr, "./png/05_plot_median_iqr.png"),
    (demo_plot_kde, "./png/06_plot_kde.png"),
    (demo_plot_ecdf, "./png/07_plot_ecdf.png"),
    (demo_plot_box, "./png/08_plot_box.png"),
    (demo_plot_violin, "./png/09_plot_violin.png"),
    (demo_plot_bar, "./png/10_plot_bar.png"),
    (demo_plot_barh, "./png/11_plot_barh.png"),
    (demo_plot_scatter, "./png/12_plot_scatter.png"),
    (demo_plot_errorbar, "./png/13_plot_errorbar.png"),
    (demo_plot_fill_between, "./png/14_plot_fill_between.png"),
    (demo_plot_fillv, "./png/15_plot_fillv.png"),
    (demo_plot_contour, "./png/16_plot_contour.png"),
    (demo_plot_imshow, "./png/17_plot_imshow.png"),
    (demo_plot_image, "./png/18_plot_image.png"),
    (demo_plot_heatmap, "./png/19_plot_heatmap.png"),
    (demo_plot_conf_mat, "./png/20_plot_conf_mat.png"),
    (demo_plot_boxplot, "./png/21_plot_boxplot.png"),
    (demo_plot_violinplot, "./png/22_plot_violinplot.png"),
    (demo_plot_raster, "./png/23_plot_raster.png"),
    (demo_plot_joyplot, "./png/24_plot_joyplot.png"),
    (demo_plot_rectangle, "./png/25_plot_rectangle.png"),
]


@stx.session
def main(
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Demo: SciTeX custom wrapper methods (25 types)."""
    STYLE = SCITEX_STYLE.copy()

    logger.info("=" * 70)
    logger.info("Demo: SciTeX Custom Wrapper Methods (ax.plot_xxx)")
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
