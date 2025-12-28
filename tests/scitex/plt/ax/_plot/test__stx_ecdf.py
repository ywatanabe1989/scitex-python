# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/ax/_plot/_stx_ecdf.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-01 14:00:00 (ywatanabe)"
# # File: ./src/scitex/plt/ax/_plot/_plot_ecdf.py
# 
# """Empirical Cumulative Distribution Function (ECDF) plotting."""
# 
# from typing import Any, Tuple, Union
# 
# import numpy as np
# import pandas as pd
# from matplotlib.axes import Axes
# 
# from scitex import logging
# from scitex.pd._force_df import force_df as scitex_pd_force_df
# from ....plt.utils import assert_valid_axis, mm_to_pt
# 
# logger = logging.getLogger(__name__)
# 
# 
# # Default line width (0.2mm for publication)
# DEFAULT_LINE_WIDTH_MM = 0.2
# 
# 
# def stx_ecdf(
#     axis: Union[Axes, "AxisWrapper"],
#     values_1d: np.ndarray,
#     **kwargs: Any,
# ) -> Tuple[Union[Axes, "AxisWrapper"], pd.DataFrame]:
#     """Plot Empirical Cumulative Distribution Function (ECDF).
# 
#     The ECDF shows the proportion of data points less than or equal to each
#     value, representing the empirical estimate of the cumulative distribution
#     function.
# 
#     Parameters
#     ----------
#     axis : matplotlib.axes.Axes or AxisWrapper
#         Matplotlib axis or scitex axis wrapper to plot on.
#     values_1d : array-like, shape (n_samples,)
#         1D array of values to compute and plot ECDF for. NaN values are automatically ignored.
#     **kwargs : dict
#         Additional arguments passed to plot function.
# 
#     Returns
#     -------
#     axis : matplotlib.axes.Axes or AxisWrapper
#         The axes with the ECDF plot.
#     df : pd.DataFrame
#         DataFrame containing ECDF data with columns:
#         - x: sorted data values
#         - y: cumulative percentages (0-100)
#         - n: total number of data points
#         - x_step, y_step: step plot coordinates
# 
#     Examples
#     --------
#     >>> import numpy as np
#     >>> import scitex as stx
#     >>> data = np.random.randn(100)
#     >>> fig, ax = stx.plt.subplots()
#     >>> ax, df = stx.plt.ax.stx_ecdf(ax, data)
#     """
#     assert_valid_axis(
#         axis, "First argument must be a matplotlib axis or scitex axis wrapper"
#     )
# 
#     # Flatten and remove NaN values
#     values_1d = np.hstack(values_1d)
# 
#     # Warnings
#     if np.isnan(values_1d).any():
#         logger.warning("NaN values are ignored for ECDF plot.")
#     values_1d = values_1d[~np.isnan(values_1d)]
#     nn = len(values_1d)
# 
#     # Sort the data and compute the ECDF values
#     data_sorted = np.sort(values_1d)
#     ecdf_perc = 100 * np.arange(1, len(data_sorted) + 1) / len(data_sorted)
# 
#     # Create the pseudo x-axis for step plotting
#     x_step = np.repeat(data_sorted, 2)[1:]
#     y_step = np.repeat(ecdf_perc, 2)[:-1]
# 
#     # Apply default linewidth if not specified
#     if "linewidth" not in kwargs and "lw" not in kwargs:
#         kwargs["linewidth"] = mm_to_pt(DEFAULT_LINE_WIDTH_MM)
# 
#     # Add sample size to label if provided
#     if "label" in kwargs and kwargs["label"]:
#         kwargs["label"] = f"{kwargs['label']} ($n$={nn})"
# 
#     # Plot the ECDF using steps (no markers - clean line only)
#     axis.plot(x_step, y_step, drawstyle="steps-post", **kwargs)
# 
#     # Set ylim (xlim is auto-scaled based on data)
#     axis.set_ylim(0, 100)
# 
#     # Create a DataFrame to hold the ECDF data
#     df = scitex_pd_force_df(
#         {
#             "x": data_sorted,
#             "y": ecdf_perc,
#             "n": nn,
#             "x_step": x_step,
#             "y_step": y_step,
#         }
#     )
# 
#     return axis, df
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/ax/_plot/_stx_ecdf.py
# --------------------------------------------------------------------------------
