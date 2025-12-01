<!-- ---
!-- Timestamp: 2025-12-01 09:37:28
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/TODO.md
!-- --- -->

## Update scitex.{plt,vis}

### Demo Scripts
- `examples/demo_matplotlib_basic.py` - Pure matplotlib methods
- `examples/demo_scitex_wrappers.py` - SciTeX custom wrappers (plot_xxx)
- `examples/demo_seaborn_wrappers.py` - Seaborn wrappers (sns_xxx)

### scitex.stats integration
When available, stats information should be incorporated into metadata of figures (json and embedded info).
Please see ~/proj/scitex-cloud/apps/vis_app/ for our intention.

---

## Matplotlib Basic (21 types)
Script: `examples/demo_matplotlib_basic.py`
Output: `examples/demo_matplotlib_basic_out/`

| #  | File                 | Method             | Status |
|----|----------------------|--------------------|--------|
| 01 | 01_plot.png          | ax.plot()          | [ ]    |
| 02 | 02_step.png          | ax.step()          | [ ]    |
| 03 | 03_stem.png          | ax.stem()          | [ ]    |
| 04 | 04_scatter.png       | ax.scatter()       | [ ]    |
| 05 | 05_bar.png           | ax.bar()           | [ ]    |
| 06 | 06_barh.png          | ax.barh()          | [ ]    |
| 07 | 07_hist.png          | ax.hist()          | [ ]    |
| 08 | 08_hist2d.png        | ax.hist2d()        | [ ]    |
| 09 | 09_hexbin.png        | ax.hexbin()        | [ ]    |
| 10 | 10_boxplot.png       | ax.boxplot()       | [ ]    |
| 11 | 11_violinplot.png    | ax.violinplot()    | [ ]    |
| 12 | 12_fill_between.png  | ax.fill_between()  | [ ]    |
| 13 | 13_fill_betweenx.png | ax.fill_betweenx() | [ ]    |
| 14 | 14_errorbar.png      | ax.errorbar()      | [ ]    |
| 15 | 15_contour.png       | ax.contour()       | [ ]    |
| 16 | 16_contourf.png      | ax.contourf()      | [ ]    |
| 17 | 17_imshow.png        | ax.imshow()        | [ ]    |
| 18 | 18_matshow.png       | ax.matshow()       | [ ]    |
| 19 | 19_pie.png           | ax.pie()           | [ ]    |
| 20 | 20_quiver.png        | ax.quiver()        | [ ]    |
| 21 | 21_streamplot.png    | ax.streamplot()    | [ ]    |

---

## SciTeX Wrappers (25 types)
Script: `examples/demo_scitex_wrappers.py`
Output: `examples/demo_scitex_wrappers_out/`

| #  | File                     | Method                 | Status |
|----|--------------------------|------------------------|--------|
| 01 | 01_plot_line.png         | ax.plot_line()         | [ ]    |
| 02 | 02_plot_shaded_line.png  | ax.plot_shaded_line()  | [ ]    |
| 03 | 03_plot_mean_std.png     | ax.plot_mean_std()     | [ ]    |
| 04 | 04_plot_mean_ci.png      | ax.plot_mean_ci()      | [ ]    |
| 05 | 05_plot_median_iqr.png   | ax.plot_median_iqr()   | [ ]    |
| 06 | 06_plot_kde.png          | ax.plot_kde()          | [ ]    |
| 07 | 07_plot_ecdf.png         | ax.plot_ecdf()         | [ ]    |
| 08 | 08_plot_box.png          | ax.plot_box()          | [ ]    |
| 09 | 09_plot_violin.png       | ax.plot_violin()       | [ ]    |
| 10 | 10_plot_bar.png          | ax.plot_bar()          | [ ]    |
| 11 | 11_plot_barh.png         | ax.plot_barh()         | [ ]    |
| 12 | 12_plot_scatter.png      | ax.plot_scatter()      | [ ]    |
| 13 | 13_plot_errorbar.png     | ax.plot_errorbar()     | [ ]    |
| 14 | 14_plot_fill_between.png | ax.plot_fill_between() | [ ]    |
| 15 | 15_plot_fillv.png        | ax.plot_fillv()        | [ ]    |
| 16 | 16_plot_contour.png      | ax.plot_contour()      | [ ]    |
| 17 | 17_plot_imshow.png       | ax.plot_imshow()       | [ ]    |
| 18 | 18_plot_image.png        | ax.plot_image()        | [ ]    |
| 19 | 19_plot_heatmap.png      | ax.plot_heatmap()      | [ ]    |
| 20 | 20_plot_conf_mat.png     | ax.plot_conf_mat()     | [ ]    |
| 21 | 21_plot_boxplot.png      | ax.plot_boxplot()      | [ ]    |
| 22 | 22_plot_violinplot.png   | ax.plot_violinplot()   | [ ]    |
| 23 | 23_plot_raster.png       | ax.plot_raster()       | [ ]    |
| 24 | 24_plot_joyplot.png      | ax.plot_joyplot()      | [ ]    |
| 25 | 25_plot_rectangle.png    | ax.plot_rectangle()    | [ ]    |

---

## Seaborn Wrappers (10 types)
Script: `examples/demo_seaborn_wrappers.py`
Output: `examples/demo_seaborn_wrappers_out/`

| #  | File                   | Method               | Status |
|----|------------------------|----------------------|--------|
| 01 | 01_sns_boxplot.png     | ax.sns_boxplot()     | [ ]    |
| 02 | 02_sns_violinplot.png  | ax.sns_violinplot()  | [ ]    |
| 03 | 03_sns_scatterplot.png | ax.sns_scatterplot() | [ ]    |
| 04 | 04_sns_lineplot.png    | ax.sns_lineplot()    | [ ]    |
| 05 | 05_sns_histplot.png    | ax.sns_histplot()    | [ ]    |
| 06 | 06_sns_kdeplot.png     | ax.sns_kdeplot()     | [ ]    |
| 07 | 07_sns_barplot.png     | ax.sns_barplot()     | [ ]    |
| 08 | 08_sns_stripplot.png   | ax.sns_stripplot()   | [ ]    |
| 09 | 09_sns_swarmplot.png   | ax.sns_swarmplot()   | [ ]    |
| 10 | 10_sns_heatmap.png     | ax.sns_heatmap()     | [ ]    |

---

## Summary
- **Total**: 56 plot types
  - Matplotlib Basic: 21
  - SciTeX Wrappers: 25
  - Seaborn Wrappers: 10

<!-- EOF -->