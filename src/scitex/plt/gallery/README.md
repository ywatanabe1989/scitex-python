<!-- ---
!-- Timestamp: 2025-12-08 23:59:46
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/gallery/README.md
!-- --- -->

# SciTeX Plot Gallery

Generate sample plots for all 46 plot types available in `scitex.plt`.

## Usage

```python
import scitex as stx

# Generate all plots
stx.plt.gallery.generate()

# Generate specific category
stx.plt.gallery.generate(category="statistical")

# Generate single plot
stx.plt.gallery.generate(plot_type="stx_mean_std")

# List available plots
stx.plt.gallery.list_plots()
```

## Signature

```python
def generate(
    output_dir="./gallery",
    category=None,
    plot_type=None,
    figsize=(4, 3),
    dpi=150,
    save_csv=True,
    save_png=True,
    verbose=True,
) -> dict:

def list_plots(category=None) -> list:
```

## Output

Each plot generates PNG and CSV files organized by category:

```
gallery/<category>/<plot_name>.png
gallery/<category>/<plot_name>.csv
```

## Categories

```python
fig, ax = stx.plt.subplots()
ax.<method>()
```

| Category     | Methods                                                                                                                                                                                            |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| line         | `plot`, `step`, `stx_line`, `stx_shaded_line`                                                                                                                                                      |
| scatter      | `scatter`, `stx_scatter`                                                                                                                                                                           |
| bar          | `bar`, `barh`, `stx_bar`, `stx_barh`                                                                                                                                                               |
| statistical  | `stx_mean_std`, `stx_mean_ci`, `stx_median_iqr`, `errorbar`, `stx_errorbar`                                                                                                                        |
| distribution | `hist`, `hist2d`, `stx_kde`, `stx_ecdf`                                                                                                                                                            |
| categorical  | `boxplot`, `stx_box`, `stx_boxplot`, `stx_violin`, `stx_violinplot`, `sns_stripplot`, `sns_swarmplot`                                                                                              |
| heatmap      | `imshow`, `stx_imshow`, `stx_heatmap`, `stx_image`, `stx_conf_mat`                                                                                                                                 |
| contour      | `contour`, `contourf`, `stx_contour`, `stx_fillv`, `stx_fill_between`                                                                                                                              |
| special      | `stx_raster`, `stx_rectangle`, `stx_joyplot`, `stx_scatter_hist`                                                                                                                                   |
| seaborn      | `sns_barplot`, `sns_boxplot`, `sns_heatmap`, `sns_histplot`, `sns_jointplot`, `sns_kdeplot`, `sns_lineplot`, `sns_pairplot`, `sns_scatterplot`, `sns_stripplot`, `sns_swarmplot`, `sns_violinplot` |

<!-- EOF -->