#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-02 05:54:57 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/plt/demo_scitex_wrappers.py


"""Demo: SciTeX custom wrapper methods (ax.stx_xxx)."""

# Imports
import scitex as stx
from scitex.plt.styles.presets import SCITEX_STYLE

from demo_scitex_wrappers import demo_stx_line
from demo_scitex_wrappers import demo_stx_shaded_line
from demo_scitex_wrappers import demo_stx_mean_std
from demo_scitex_wrappers import demo_stx_mean_ci
from demo_scitex_wrappers import demo_stx_median_iqr
from demo_scitex_wrappers import demo_stx_kde
from demo_scitex_wrappers import demo_stx_ecdf
from demo_scitex_wrappers import demo_stx_box
from demo_scitex_wrappers import demo_stx_violin
from demo_scitex_wrappers import demo_stx_bar
from demo_scitex_wrappers import demo_stx_barh
from demo_scitex_wrappers import demo_stx_scatter
from demo_scitex_wrappers import demo_stx_errorbar
from demo_scitex_wrappers import demo_stx_fill_between
from demo_scitex_wrappers import demo_stx_fillv
from demo_scitex_wrappers import demo_stx_contour
from demo_scitex_wrappers import demo_stx_imshow
from demo_scitex_wrappers import demo_stx_image
from demo_scitex_wrappers import demo_stx_heatmap
from demo_scitex_wrappers import demo_stx_conf_mat
from demo_scitex_wrappers import demo_stx_boxplot
from demo_scitex_wrappers import demo_stx_violinplot
from demo_scitex_wrappers import demo_stx_raster
from demo_scitex_wrappers import demo_stx_joyplot
from demo_scitex_wrappers import demo_stx_rectangle


# Demo registry: (function, filename)
DEMOS = [
    (demo_stx_line, "./png/01_stx_line.png"),
    (demo_stx_shaded_line, "./png/02_stx_shaded_line.png"),
    (demo_stx_mean_std, "./png/03_stx_mean_std.png"),
    (demo_stx_mean_ci, "./png/04_stx_mean_ci.png"),
    (demo_stx_median_iqr, "./png/05_stx_median_iqr.png"),
    (demo_stx_kde, "./png/06_stx_kde.png"),
    (demo_stx_ecdf, "./png/07_stx_ecdf.png"),
    (demo_stx_box, "./png/08_stx_box.png"),
    (demo_stx_violin, "./png/09_stx_violin.png"),
    (demo_stx_bar, "./png/10_stx_bar.png"),
    (demo_stx_barh, "./png/11_stx_barh.png"),
    (demo_stx_scatter, "./png/12_stx_scatter.png"),
    (demo_stx_errorbar, "./png/13_stx_errorbar.png"),
    (demo_stx_fill_between, "./png/14_stx_fill_between.png"),
    (demo_stx_fillv, "./png/15_stx_fillv.png"),
    (demo_stx_contour, "./png/16_stx_contour.png"),
    (demo_stx_imshow, "./png/17_stx_imshow.png"),
    (demo_stx_image, "./png/18_stx_image.png"),
    (demo_stx_heatmap, "./png/19_stx_heatmap.png"),
    (demo_stx_conf_mat, "./png/20_stx_conf_mat.png"),
    (demo_stx_boxplot, "./png/21_stx_boxplot.png"),
    (demo_stx_violinplot, "./png/22_stx_violinplot.png"),
    (demo_stx_raster, "./png/23_stx_raster.png"),
    (demo_stx_joyplot, "./png/24_stx_joyplot.png"),
    (demo_stx_rectangle, "./png/25_stx_rectangle.png"),
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
    logger.info("Demo: SciTeX Custom Wrapper Methods (ax.stx_xxx)")
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
