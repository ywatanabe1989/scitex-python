<!-- ---
!-- Timestamp: 2025-12-01 09:34:16
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/TODO.md
!-- --- -->

## Update scitex.{plt,vis}

### /home/ywatanabe/proj/scitex-code/examples/demo_plot_all_types_publication.py
Demo plotting script as default behavior gallery
- Do not udpate this demo plotting script. 
- Instead, revise the source code.

### Produced Figures
You can see ~/proj/scitex-code/examples/demo_plot_all_types_publication_out/publication/

## scitex.stats integration
When available, stats information should be incorporated into metadata of figures (json and embedded info)
Please consider how to integrate scitex.stats into scitex.{plt,vis} in an standardized manner
Please see ~/proj/scitex-cloud/apps/vis_app/ for our intention
We will use this scitex.plt module as a backend with mm-level of adjustment for high quality figures and enjoy GUI adjustment in the django app, allowing style changes by redrawing figures using scitex.{plt,vis}

## Redraw based on codebase update
Run this to override the figures and metadata
- Plotting script: /home/ywatanabe/proj/scitex-code/examples/demo_plot_all_types_publication.py
- Output: /home/ywatanabe/proj/scitex-code/examples/demo_plot_all_types_publication_out/publication/

### PNG Files in publication directory (29 total):

/home/ywatanabe/proj/scitex-code/examples/demo_plot_all_types.py
/home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers.py
/home/ywatanabe/proj/scitex-code/examples/demo_seaborn_wrappers.py

#### 01_matplotlib_basic (11 files):
/home/ywatanabe/proj/scitex-code/examples/demo_plot_all_types.py
- [x] 01_plot.png
- [x] 02_scatter.png
  - [x] Fit line should be black and "--"
  - [x] Add R2 and p value
- [ ] 03_bar.png
  - [ ] error bars should be in black 
  - [ ] Do not make bold the xlabel
  - [ ] Allow colors for each bar based on groups (more control)
- [ ] 04_hist.png
  - [x] Make the KDE in black
  - [ ] in dotted "--"
- [ ] 05_boxplot.png
  - [ ] Make no fliers
  - [ ] Add legends
  - [ ] midian line as black
- [ ] 06_errorbar.png
  - [ ] 0.08 mm for symbol size
- [ ] 07_barh.png
  - [ ] Same for the vertial bar
- [ ] 08_fill_between.png
  - [ ] The color of the mean (median) should follow our color theme
- [ ] 09_imshow.png
  - [ ] Maybe this does not need x,y,z if the intent is to show image
  - [ ] We have heatmap and make distinguishes between this
- [x] 10_contour.png
- [ ] 11_violinplot.png
  - [ ] May better to make error bars and vertical lines to be black
  - [ ] Legend needed

#### 02_custom_scitex (7 files):
/home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers.py

- [ ] 01_plot_heatmap.png
  - [ ] Adjust inner labels based on size of the axis
  - [ ] The length of colorbar should align with the height of the y axis
- [x] 02_plot_line.png
- [x] 03_plot_shaded_line.png
- [ ] 04_plot_violin.png
  - [ ] Align the ticks and violin plots
  - [ ] Remove borders of violins
  - [ ] Add legend all the time
  - [ ] Make boxplots more match to violin
- [ ] 05_plot_ecdf.png
  - [ ] Use line
  - [ ] Maybe symbol needs to be removed at least when sample number is large
- [ ] 06_plot_box.png
  - [ ] Width of borders should be 0.2mm
  - [ ] No fliers
- [ ] 07_plot_mean_std.png
  - [ ] use n=10 to show SD

#### 03_functional (1 file):
- [ ] 01_plot_kde.png

#### 04_seaborn (7 files):
/home/ywatanabe/proj/scitex-code/examples/demo_seaborn_wrappers.py
- [ ] 01_sns_boxplot.png
- [ ] 02_sns_violinplot.png
- [ ] 03_sns_scatterplot.png
- [ ] 04_sns_lineplot.png
- [ ] 05_sns_histplot.png
- [ ] 06_sns_barplot.png
- [ ] 07_sns_stripplot.png

#### 05_multi_panel (2 files):
- [ ] 01_2x2_scitex.png
- [ ] 02_1x3_varied_widths.png

#### Root (1 file):
- [ ] style_override.png

<!-- EOF -->