#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-09-21 01:52:22 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/plt/_subplots/_export_as_csv.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
import pandas as pd
from scitex.pd import to_xyz

from scitex import logging

logger = logging.getLogger(__name__)

# Global warning registry to track which warnings have been shown
_warning_registry = set()

# Mapping of matplotlib/seaborn methods to their scitex equivalents
_METHOD_ALTERNATIVES = {
    # Matplotlib methods
    "imshow": "plot_imshow",
    "plot": "plot",  # already tracked
    "scatter": "plot_scatter",  # already tracked
    "bar": "plot_bar",  # already tracked
    "barh": "plot_barh",  # already tracked
    "hist": "hist",  # already tracked
    "boxplot": "stx_box or plot_boxplot",
    "violinplot": "stx_violin or plot_violinplot",
    "fill_between": "plot_fill_between",
    "errorbar": "plot_errorbar",
    "contour": "plot_contour",
    "heatmap": "stx_heatmap",
    # Seaborn methods (accessed via ax.sns_*)
    "scatterplot": "sns_scatterplot",
    "lineplot": "sns_lineplot",
    "barplot": "sns_barplot",
    "boxplot_sns": "sns_boxplot",
    "violinplot_sns": "sns_violinplot",
    "stripplot": "sns_stripplot",
    "swarmplot": "sns_swarmplot",
    "histplot": "sns_histplot",
    "kdeplot": "sns_kdeplot",
    "heatmap_sns": "sns_heatmap",
    "jointplot": "sns_jointplot",
    "pairplot": "sns_pairplot",
}


def _warn_once(message, category=UserWarning):
    """Show a warning only once per runtime.

    Args:
        message: Warning message to display
        category: Warning category (default: UserWarning)
    """
    if message not in _warning_registry:
        _warning_registry.add(message)
        logger.warning(message)


from ._export_as_csv_formatters import (
    # Standard matplotlib formatters
    _format_annotate,
    _format_bar,
    _format_barh,
    _format_boxplot,
    _format_contour,
    _format_contourf,
    _format_errorbar,
    _format_eventplot,
    _format_fill,
    _format_fill_between,
    _format_stackplot,
    _format_pcolormesh,
    _format_hexbin,
    _format_hist,
    _format_hist2d,
    _format_imshow,
    _format_imshow2d,
    _format_matshow,
    _format_pie,
    _format_plot,
    _format_quiver,
    _format_scatter,
    _format_stem,
    _format_step,
    _format_streamplot,
    _format_text,
    _format_violin,
    _format_violinplot,
    # Custom scitex formatters
    _format_plot_box,
    _format_plot_conf_mat,
    _format_stx_contour,
    _format_plot_ecdf,
    _format_plot_fillv,
    _format_plot_heatmap,
    _format_plot_image,
    _format_plot_imshow,
    _format_stx_imshow,
    _format_plot_joyplot,
    _format_plot_kde,
    _format_plot_line,
    _format_plot_mean_ci,
    _format_plot_mean_std,
    _format_plot_median_iqr,
    _format_plot_raster,
    _format_plot_rectangle,
    _format_plot_scatter,
    _format_plot_scatter_hist,
    _format_plot_shaded_line,
    _format_plot_violin,
    # stx_ aliases formatters
    _format_stx_scatter,
    _format_stx_bar,
    _format_stx_barh,
    _format_stx_errorbar,
    # Seaborn formatters
    _format_sns_barplot,
    _format_sns_boxplot,
    _format_sns_heatmap,
    _format_sns_histplot,
    _format_sns_jointplot,
    _format_sns_kdeplot,
    _format_sns_lineplot,
    _format_sns_pairplot,
    _format_sns_scatterplot,
    _format_sns_stripplot,
    _format_sns_swarmplot,
    _format_sns_violinplot,
)

# Registry mapping method names to their formatter functions
_FORMATTER_REGISTRY = {
    # Standard matplotlib methods
    "annotate": _format_annotate,
    "bar": _format_bar,
    "barh": _format_barh,
    "boxplot": _format_boxplot,
    "contour": _format_contour,
    "contourf": _format_contourf,
    "errorbar": _format_errorbar,
    "eventplot": _format_eventplot,
    "fill": _format_fill,
    "fill_between": _format_fill_between,
    "stackplot": _format_stackplot,
    "pcolormesh": _format_pcolormesh,
    "pcolor": _format_pcolormesh,
    "hexbin": _format_hexbin,
    "hist": _format_hist,
    "hist2d": _format_hist2d,
    "imshow": _format_imshow,
    "imshow2d": _format_imshow2d,
    "matshow": _format_matshow,
    "pie": _format_pie,
    "plot": _format_plot,
    "quiver": _format_quiver,
    "scatter": _format_scatter,
    "stem": _format_stem,
    "step": _format_step,
    "streamplot": _format_streamplot,
    "text": _format_text,
    "violin": _format_violin,
    "violinplot": _format_violinplot,
    # Custom scitex methods
    "stx_box": _format_plot_box,
    "stx_conf_mat": _format_plot_conf_mat,
    "stx_contour": _format_stx_contour,
    "stx_ecdf": _format_plot_ecdf,
    "stx_fillv": _format_plot_fillv,
    "stx_heatmap": _format_plot_heatmap,
    "stx_image": _format_plot_image,
    "plot_imshow": _format_plot_imshow,
    "stx_imshow": _format_stx_imshow,
    "stx_joyplot": _format_plot_joyplot,
    "stx_kde": _format_plot_kde,
    "stx_line": _format_plot_line,
    "stx_mean_ci": _format_plot_mean_ci,
    "stx_mean_std": _format_plot_mean_std,
    "stx_median_iqr": _format_plot_median_iqr,
    "stx_raster": _format_plot_raster,
    "stx_rectangle": _format_plot_rectangle,
    "plot_scatter": _format_plot_scatter,
    "stx_scatter_hist": _format_plot_scatter_hist,
    "stx_shaded_line": _format_plot_shaded_line,
    "stx_violin": _format_plot_violin,
    # stx_ aliases
    "stx_scatter": _format_stx_scatter,
    "stx_bar": _format_stx_bar,
    "stx_barh": _format_stx_barh,
    "stx_errorbar": _format_stx_errorbar,
    # Seaborn methods (sns_ prefix)
    "sns_barplot": _format_sns_barplot,
    "sns_boxplot": _format_sns_boxplot,
    "sns_heatmap": _format_sns_heatmap,
    "sns_histplot": _format_sns_histplot,
    "sns_jointplot": _format_sns_jointplot,
    "sns_kdeplot": _format_sns_kdeplot,
    "sns_lineplot": _format_sns_lineplot,
    "sns_pairplot": _format_sns_pairplot,
    "sns_scatterplot": _format_sns_scatterplot,
    "sns_stripplot": _format_sns_stripplot,
    "sns_swarmplot": _format_sns_swarmplot,
    "sns_violinplot": _format_sns_violinplot,
}


def _to_numpy(data):
    """Convert various data types to numpy array.

    Handles torch tensors, pandas Series/DataFrame, and other array-like objects.

    Parameters
    ----------
    data : array-like
        Data to convert to numpy array

    Returns
    -------
    numpy.ndarray
        Data as numpy array
    """
    if hasattr(data, "numpy"):  # torch tensor
        return data.detach().numpy() if hasattr(data, "detach") else data.numpy()
    elif hasattr(data, "values"):  # pandas series/dataframe
        return data.values
    else:
        return np.asarray(data)


def export_as_csv(history_records):
    """Convert plotting history records to a combined DataFrame suitable for CSV export.

    Args:
        history_records (dict): Dictionary of plotting records.

    Returns:
        pd.DataFrame: Combined DataFrame containing all plotting data.

    Raises:
        ValueError: If no plotting records are found or they cannot be combined.
    """
    if len(history_records) <= 0:
        logger.warning("Plotting records not found. Cannot export empty data.")
        return pd.DataFrame()  # Return empty DataFrame instead of None

    dfs = []
    failed_methods = set()  # Track failed methods for helpful warnings

    for record_index, record in enumerate(list(history_records.values())):
        try:
            formatted_df = format_record(record, record_index=record_index)
            if formatted_df is not None and not formatted_df.empty:
                dfs.append(formatted_df)
            else:
                # Track the method that failed to format
                method_name = record[1] if len(record) > 1 else "unknown"
                failed_methods.add(method_name)
        except Exception as e:
            method_name = record[1] if len(record) > 1 else "unknown"
            failed_methods.add(method_name)

    # If no valid dataframes were created, provide helpful suggestions
    if not dfs and failed_methods:
        for method in failed_methods:
            if method in _METHOD_ALTERNATIVES:
                alternative = _METHOD_ALTERNATIVES[method]
                message = (
                    f"Matplotlib method '{method}()' does not support full data tracking for CSV export. "
                    f"Consider using 'ax.{alternative}()' instead for better data export support."
                )
            else:
                message = (
                    f"Method '{method}()' does not support data tracking for CSV export. "
                    f"Consider using scitex plot methods (e.g., stx_image, plot_imshow) for data export support."
                )
            _warn_once(message)
        return pd.DataFrame()

    try:
        # Reset index for each dataframe to avoid alignment issues
        dfs_reset = [df.reset_index(drop=True) for df in dfs]
        df = pd.concat(dfs_reset, axis=1)
        return df
    except Exception as e:
        logger.warning(f"Failed to combine plotting records: {e}")
        # Return a DataFrame with metadata about what records were attempted
        meta_df = pd.DataFrame(
            {
                "record_id": [r[0] for r in history_records.values()],
                "method": [r[1] for r in history_records.values()],
                "has_data": [
                    "Yes" if r[2] and r[2] != {} else "No"
                    for r in history_records.values()
                ],
            }
        )
        return meta_df


def format_record(record, record_index=0):
    """Route record to the appropriate formatting function based on plot method.

    Args:
        record (tuple): Plotting record tuple (id, method, tracked_dict, kwargs).
        record_index (int): Index of this record in the history (used as fallback
            for trace_id when user doesn't provide an explicit id= kwarg).

    Returns:
        pd.DataFrame: Formatted data for the plot record.
    """
    id, method, tracked_dict, kwargs = record

    # Basic Matplotlib functions
    if method == "plot":
        return _format_plot(id, tracked_dict, kwargs)
    elif method == "scatter":
        return _format_scatter(id, tracked_dict, kwargs)
    elif method == "bar":
        return _format_bar(id, tracked_dict, kwargs)
    elif method == "barh":
        return _format_barh(id, tracked_dict, kwargs)
    elif method == "hist":
        return _format_hist(id, tracked_dict, kwargs)
    elif method == "boxplot":
        return _format_boxplot(id, tracked_dict, kwargs)
    elif method == "contour":
        return _format_contour(id, tracked_dict, kwargs)
    elif method == "contourf":
        return _format_contourf(id, tracked_dict, kwargs)
    elif method == "errorbar":
        return _format_errorbar(id, tracked_dict, kwargs)
    elif method == "eventplot":
        return _format_eventplot(id, tracked_dict, kwargs)
    elif method == "fill":
        return _format_fill(id, tracked_dict, kwargs)
    elif method == "fill_between":
        return _format_fill_between(id, tracked_dict, kwargs)
    elif method == "stackplot":
        return _format_stackplot(id, tracked_dict, kwargs)
    elif method == "pcolormesh":
        return _format_pcolormesh(id, tracked_dict, kwargs)
    elif method == "pcolor":
        return _format_pcolormesh(id, tracked_dict, kwargs)
    elif method == "hexbin":
        return _format_hexbin(id, tracked_dict, kwargs)
    elif method == "hist2d":
        return _format_hist2d(id, tracked_dict, kwargs)
    elif method == "imshow":
        return _format_imshow(id, tracked_dict, kwargs)
    elif method == "imshow2d":
        return _format_imshow2d(id, tracked_dict, kwargs)
    elif method == "matshow":
        return _format_matshow(id, tracked_dict, kwargs)
    elif method == "pie":
        return _format_pie(id, tracked_dict, kwargs)
    elif method == "quiver":
        return _format_quiver(id, tracked_dict, kwargs)
    elif method == "stem":
        return _format_stem(id, tracked_dict, kwargs)
    elif method == "step":
        return _format_step(id, tracked_dict, kwargs)
    elif method == "streamplot":
        return _format_streamplot(id, tracked_dict, kwargs)
    elif method == "violin":
        return _format_violin(id, tracked_dict, kwargs)
    elif method == "violinplot":
        return _format_violinplot(id, tracked_dict, kwargs)
    elif method == "text":
        return _format_text(id, tracked_dict, kwargs)
    elif method == "annotate":
        return _format_annotate(id, tracked_dict, kwargs)

    # Custom plotting functions
    elif method == "stx_box":
        return _format_plot_box(id, tracked_dict, kwargs)
    elif method == "stx_conf_mat":
        return _format_plot_conf_mat(id, tracked_dict, kwargs)
    elif method == "stx_contour":
        return _format_stx_contour(id, tracked_dict, kwargs)
    elif method == "stx_ecdf":
        return _format_plot_ecdf(id, tracked_dict, kwargs)
    elif method == "stx_fillv":
        return _format_plot_fillv(id, tracked_dict, kwargs)
    elif method == "stx_heatmap":
        return _format_plot_heatmap(id, tracked_dict, kwargs)
    elif method == "stx_image":
        return _format_plot_image(id, tracked_dict, kwargs)
    elif method == "plot_imshow":
        return _format_plot_imshow(id, tracked_dict, kwargs)
    elif method == "stx_imshow":
        return _format_stx_imshow(id, tracked_dict, kwargs)
    elif method == "stx_joyplot":
        return _format_plot_joyplot(id, tracked_dict, kwargs)
    elif method == "stx_kde":
        return _format_plot_kde(id, tracked_dict, kwargs)
    elif method == "stx_line":
        return _format_plot_line(id, tracked_dict, kwargs)
    elif method == "stx_mean_ci":
        return _format_plot_mean_ci(id, tracked_dict, kwargs)
    elif method == "stx_mean_std":
        return _format_plot_mean_std(id, tracked_dict, kwargs)
    elif method == "stx_median_iqr":
        return _format_plot_median_iqr(id, tracked_dict, kwargs)
    elif method == "stx_raster":
        return _format_plot_raster(id, tracked_dict, kwargs)
    elif method == "stx_rectangle":
        return _format_plot_rectangle(id, tracked_dict, kwargs)
    elif method == "plot_scatter":
        return _format_plot_scatter(id, tracked_dict, kwargs)
    elif method == "stx_scatter_hist":
        return _format_plot_scatter_hist(id, tracked_dict, kwargs)
    elif method == "stx_shaded_line":
        return _format_plot_shaded_line(id, tracked_dict, kwargs)
    elif method == "stx_violin":
        return _format_plot_violin(id, tracked_dict, kwargs)

    # stx_ aliases
    elif method == "stx_scatter":
        return _format_stx_scatter(id, tracked_dict, kwargs)
    elif method == "stx_bar":
        return _format_stx_bar(id, tracked_dict, kwargs)
    elif method == "stx_barh":
        return _format_stx_barh(id, tracked_dict, kwargs)
    elif method == "stx_errorbar":
        return _format_stx_errorbar(id, tracked_dict, kwargs)

    # Seaborn functions (sns_ prefix)
    elif method == "sns_barplot":
        return _format_sns_barplot(id, tracked_dict, kwargs)
    elif method == "sns_boxplot":
        return _format_sns_boxplot(id, tracked_dict, kwargs)
    elif method == "sns_heatmap":
        return _format_sns_heatmap(id, tracked_dict, kwargs)
    elif method == "sns_histplot":
        return _format_sns_histplot(id, tracked_dict, kwargs)
    elif method == "sns_jointplot":
        return _format_sns_jointplot(id, tracked_dict, kwargs)
    elif method == "sns_kdeplot":
        return _format_sns_kdeplot(id, tracked_dict, kwargs)
    elif method == "sns_lineplot":
        return _format_sns_lineplot(id, tracked_dict, kwargs)
    elif method == "sns_pairplot":
        return _format_sns_pairplot(id, tracked_dict, kwargs)
    elif method == "sns_scatterplot":
        return _format_sns_scatterplot(id, tracked_dict, kwargs)
    elif method == "sns_stripplot":
        return _format_sns_stripplot(id, tracked_dict, kwargs)
    elif method == "sns_swarmplot":
        return _format_sns_swarmplot(id, tracked_dict, kwargs)
    elif method == "sns_violinplot":
        return _format_sns_violinplot(id, tracked_dict, kwargs)
    else:
        # Unknown or unimplemented method
        raise NotImplementedError(
            f"CSV export for plot method '{method}' is not yet implemented in the scitex.plt module. "
            f"Check the feature-request-export-as-csv-functions.md for implementation status."
        )


# EOF
