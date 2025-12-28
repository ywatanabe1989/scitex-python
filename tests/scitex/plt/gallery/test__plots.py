# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/gallery/_plots.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-08 23:30:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/gallery/_plots.py
# 
# """Plot definitions for gallery generation."""
# 
# import numpy as np
# 
# 
# # =============================================================================
# # Line Plots
# # =============================================================================
# def plot_plot(fig, ax, stx):
#     """ax.plot() - Basic line plot."""
#     time_sec = np.linspace(0, 2 * np.pi, 100)
#     ax.plot(time_sec, np.sin(time_sec), "b-", label="sin", id="sine_wave")
#     ax.plot(time_sec, np.cos(time_sec), "r--", label="cos", id="cosine_wave")
#     ax.set_xyt(x="Time [s]", y="Amplitude [a.u.]", t="ax.plot()")
#     ax.legend(frameon=False, fontsize=6)
#     return fig, ax
# 
# 
# def plot_step(fig, ax, stx):
#     """ax.step() - Step plot."""
#     time_ms = np.arange(20)
#     voltage_mv = np.random.randint(0, 5, 20)
#     ax.step(time_ms, voltage_mv, where="mid", id="digital_signal")
#     ax.set_xyt(x="Time [ms]", y="Voltage [mV]", t="ax.step()")
#     return fig, ax
# 
# 
# def plot_stx_line(fig, ax, stx):
#     """ax.stx_line() - Simple line from 1D array."""
#     np.random.seed(42)
#     signal_amplitude = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(
#         0, 0.1, 100
#     )
#     ax.stx_line(signal_amplitude, id="eeg_signal", label="Signal")
#     ax.set_xyt(x="Sample", y="Amplitude [μV]", t="ax.stx_line()")
#     ax.legend(frameon=False, fontsize=6)
#     return fig, ax
# 
# 
# def plot_stx_shaded_line(fig, ax, stx):
#     """ax.stx_shaded_line() - Line with shaded region."""
#     np.random.seed(42)
#     time_sec = np.linspace(0, 10, 100)
#     mean_value = np.sin(time_sec)
#     lower_bound = mean_value - 0.3
#     upper_bound = mean_value + 0.3
#     ax.stx_shaded_line(
#         time_sec, mean_value, lower_bound, upper_bound, id="confidence_band"
#     )
#     ax.set_xyt(x="Time [s]", y="Response [a.u.]", t="ax.stx_shaded_line()")
#     return fig, ax
# 
# 
# # =============================================================================
# # Statistical Summaries
# # =============================================================================
# def plot_stx_mean_std(fig, ax, stx):
#     """ax.stx_mean_std() - Mean ± standard deviation."""
#     np.random.seed(42)
#     n_trials, n_timepoints = 50, 100
#     trials_data = np.sin(np.linspace(0, 2 * np.pi, n_timepoints)) + np.random.normal(
#         0, 0.3, (n_trials, n_timepoints)
#     )
#     ax.stx_mean_std(trials_data, id="trial_average")
#     ax.set_xyt(x="Time [ms]", y="Amplitude [μV]", t="ax.stx_mean_std()")
#     return fig, ax
# 
# 
# def plot_stx_mean_ci(fig, ax, stx):
#     """ax.stx_mean_ci() - Mean with confidence interval."""
#     np.random.seed(42)
#     n_subjects, n_timepoints = 30, 100
#     subjects_data = np.sin(np.linspace(0, 2 * np.pi, n_timepoints)) + np.random.normal(
#         0, 0.4, (n_subjects, n_timepoints)
#     )
#     ax.stx_mean_ci(subjects_data, id="group_average")
#     ax.set_xyt(x="Time [ms]", y="BOLD [%]", t="ax.stx_mean_ci()")
#     return fig, ax
# 
# 
# def plot_stx_median_iqr(fig, ax, stx):
#     """ax.stx_median_iqr() - Median with interquartile range."""
#     np.random.seed(42)
#     n_samples, n_timepoints = 40, 100
#     samples_data = np.exp(np.linspace(0, 2, n_timepoints)) + np.random.exponential(
#         0.5, (n_samples, n_timepoints)
#     )
#     ax.stx_median_iqr(samples_data, id="reaction_times")
#     ax.set_xyt(x="Trial", y="RT [ms]", t="ax.stx_median_iqr()")
#     return fig, ax
# 
# 
# def plot_errorbar(fig, ax, stx):
#     """ax.errorbar() - Error bars."""
#     conditions = np.arange(5)
#     accuracy_mean = [0.75, 0.82, 0.88, 0.91, 0.85]
#     accuracy_std = [0.08, 0.06, 0.05, 0.04, 0.07]
#     ax.errorbar(
#         conditions, accuracy_mean, yerr=accuracy_std, fmt="o-", capsize=3, id="accuracy"
#     )
#     ax.set_xyt(x="Condition", y="Accuracy", t="ax.errorbar()")
#     return fig, ax
# 
# 
# def plot_stx_errorbar(fig, ax, stx):
#     """ax.stx_errorbar() - Error bars with x, y, yerr."""
#     np.random.seed(42)
#     x = np.arange(5)
#     y = np.array([0.7, 0.8, 0.85, 0.9, 0.82])
#     yerr = np.array([0.05, 0.04, 0.03, 0.04, 0.06])
#     ax.stx_errorbar(x, y, yerr=yerr, id="group_accuracy", fmt="o-", capsize=3)
#     ax.set_xyt(x="Condition", y="Accuracy", t="ax.stx_errorbar()")
#     return fig, ax
# 
# 
# # =============================================================================
# # Distributions
# # =============================================================================
# def plot_hist(fig, ax, stx):
#     """ax.hist() - Histogram."""
#     np.random.seed(42)
#     reaction_times = np.random.normal(450, 80, 1000)
#     ax.hist(reaction_times, bins=30, alpha=0.7, edgecolor="black", id="rt_distribution")
#     ax.set_xyt(x="Reaction Time [ms]", y="Count", t="ax.hist()")
#     return fig, ax
# 
# 
# def plot_hist2d(fig, ax, stx):
#     """ax.hist2d() - 2D histogram."""
#     np.random.seed(42)
#     x_position = np.random.normal(0, 1, 5000)
#     y_position = np.random.normal(0, 1, 5000)
#     ax.hist2d(x_position, y_position, bins=30, cmap="viridis", id="position_density")
#     ax.set_xyt(x="X Position [cm]", y="Y Position [cm]", t="ax.hist2d()")
#     return fig, ax
# 
# 
# def plot_stx_kde(fig, ax, stx):
#     """ax.stx_kde() - Kernel density estimation."""
#     np.random.seed(42)
#     scores = np.concatenate(
#         [np.random.normal(70, 10, 500), np.random.normal(85, 8, 300)]
#     )
#     ax.stx_kde(scores, id="score_distribution")
#     ax.set_xyt(x="Score", y="Density", t="ax.stx_kde()")
#     return fig, ax
# 
# 
# def plot_stx_ecdf(fig, ax, stx):
#     """ax.stx_ecdf() - Empirical CDF."""
#     np.random.seed(42)
#     latencies = np.random.exponential(50, 500)
#     ax.stx_ecdf(latencies, id="latency_cdf")
#     ax.set_xyt(x="Latency [ms]", y="Cumulative Probability", t="ax.stx_ecdf()")
#     return fig, ax
# 
# 
# def plot_stx_joyplot(fig, ax, stx):
#     """ax.stx_joyplot() - Ridge/joy plot."""
#     np.random.seed(42)
#     n_groups = 5
#     data_groups = [np.random.normal(i * 0.5, 0.8, 200) for i in range(n_groups)]
#     ax.stx_joyplot(data_groups, id="distributions_by_group")
#     ax.set_xyt(x="Value", y="Group", t="ax.stx_joyplot()")
#     return fig, ax
# 
# 
# # =============================================================================
# # Categorical
# # =============================================================================
# def plot_bar(fig, ax, stx):
#     """ax.bar() - Bar chart."""
#     categories = ["A", "B", "C", "D"]
#     values = [23, 45, 56, 78]
#     ax.bar(categories, values, id="category_counts")
#     ax.set_xyt(x="Category", y="Count", t="ax.bar()")
#     return fig, ax
# 
# 
# def plot_barh(fig, ax, stx):
#     """ax.barh() - Horizontal bar chart."""
#     methods = ["Method A", "Method B", "Method C", "Method D"]
#     accuracy = [0.85, 0.92, 0.78, 0.88]
#     ax.barh(methods, accuracy, id="method_comparison")
#     ax.set_xyt(x="Accuracy", y="Method", t="ax.barh()")
#     return fig, ax
# 
# 
# def plot_stx_bar(fig, ax, stx):
#     """ax.stx_bar() - Bar chart with x and height."""
#     x = ["A", "B", "C", "D"]
#     height = [0.7, 0.8, 0.85, 0.9]
#     ax.stx_bar(x, height, id="condition_means")
#     ax.set_xyt(x="Condition", y="Performance", t="ax.stx_bar()")
#     return fig, ax
# 
# 
# def plot_stx_barh(fig, ax, stx):
#     """ax.stx_barh() - Horizontal bar with y and width."""
#     y = ["Method A", "Method B", "Method C", "Method D"]
#     width = [0.75, 0.82, 0.88, 0.91]
#     ax.stx_barh(y, width, id="method_performance")
#     ax.set_xyt(x="Accuracy", y="Method", t="ax.stx_barh()")
#     return fig, ax
# 
# 
# def plot_boxplot(fig, ax, stx):
#     """ax.boxplot() - Box plot."""
#     np.random.seed(42)
#     group_a = np.random.normal(100, 15, 50)
#     group_b = np.random.normal(110, 20, 50)
#     group_c = np.random.normal(95, 10, 50)
#     ax.boxplot(
#         [group_a, group_b, group_c],
#         labels=["Control", "Treatment A", "Treatment B"],
#         id="treatment_comparison",
#     )
#     ax.set_xyt(x="Group", y="Response", t="ax.boxplot()")
#     return fig, ax
# 
# 
# def plot_violinplot(fig, ax, stx):
#     """ax.violinplot() - Violin plot."""
#     np.random.seed(42)
#     data = [np.random.normal(0, std, 100) for std in [1, 2, 1.5, 2.5]]
#     ax.violinplot(data, showmeans=True, showmedians=True, id="distribution_comparison")
#     ax.set_xyt(x="Condition", y="Value", t="ax.violinplot()")
#     return fig, ax
# 
# 
# def plot_stx_box(fig, ax, stx):
#     """ax.stx_box() - Box plot from 2D array."""
#     np.random.seed(42)
#     n_subjects, n_conditions = 30, 4
#     data = np.random.normal([50, 55, 48, 60], 10, (n_subjects, n_conditions))
#     ax.stx_box(data, id="condition_distributions")
#     ax.set_xyt(x="Condition", y="Score", t="ax.stx_box()")
#     return fig, ax
# 
# 
# def plot_stx_violin(fig, ax, stx):
#     """ax.stx_violin() - Violin plot from list of arrays."""
#     np.random.seed(42)
#     data = [
#         np.random.normal(0, 1, 100),
#         np.random.normal(0.5, 1.2, 100),
#         np.random.normal(1, 0.8, 100),
#     ]
#     ax.stx_violin(data, id="effect_distributions")
#     ax.set_xyt(x="Condition", y="Effect Size", t="ax.stx_violin()")
#     return fig, ax
# 
# 
# def plot_stx_boxplot(fig, ax, stx):
#     """ax.stx_boxplot() - Enhanced box plot."""
#     np.random.seed(42)
#     data = [np.random.exponential(scale, 100) for scale in [1, 1.5, 2, 1.2]]
#     ax.stx_boxplot(data, id="skewed_distributions")
#     ax.set_xyt(x="Group", y="Value", t="ax.stx_boxplot()")
#     return fig, ax
# 
# 
# def plot_stx_violinplot(fig, ax, stx):
#     """ax.stx_violinplot() - Enhanced violin plot."""
#     np.random.seed(42)
#     data = [
#         np.concatenate([np.random.normal(-1, 0.5, 50), np.random.normal(1, 0.5, 50)])
#         for _ in range(3)
#     ]
#     ax.stx_violinplot(data, id="bimodal_distributions")
#     ax.set_xyt(x="Condition", y="Value", t="ax.stx_violinplot()")
#     return fig, ax
# 
# 
# # =============================================================================
# # Scatter & Points
# # =============================================================================
# def plot_scatter(fig, ax, stx):
#     """ax.scatter() - Scatter plot."""
#     np.random.seed(42)
#     age = np.random.uniform(20, 80, 100)
#     response_time = 200 + age * 2 + np.random.normal(0, 30, 100)
#     ax.scatter(age, response_time, alpha=0.6, id="age_rt_correlation")
#     ax.set_xyt(x="Age [years]", y="Response Time [ms]", t="ax.scatter()")
#     return fig, ax
# 
# 
# def plot_stx_scatter(fig, ax, stx):
#     """ax.stx_scatter() - Enhanced scatter plot."""
#     np.random.seed(42)
#     x_feature = np.random.normal(0, 1, 150)
#     y_feature = 0.7 * x_feature + np.random.normal(0, 0.5, 150)
#     ax.stx_scatter(x_feature, y_feature, id="feature_correlation")
#     ax.set_xyt(x="Feature X", y="Feature Y", t="ax.stx_scatter()")
#     return fig, ax
# 
# 
# def plot_stem(fig, ax, stx):
#     """ax.stem() - Stem plot."""
#     frequencies = np.arange(0, 50, 5)
#     power_spectrum = np.abs(np.random.normal(0, 1, 10)) ** 2
#     ax.stem(frequencies, power_spectrum, id="frequency_spectrum")
#     ax.set_xyt(x="Frequency [Hz]", y="Power [dB]", t="ax.stem()")
#     return fig, ax
# 
# 
# def plot_hexbin(fig, ax, stx):
#     """ax.hexbin() - Hexagonal binning."""
#     np.random.seed(42)
#     x_coord = np.random.normal(0, 2, 10000)
#     y_coord = np.random.normal(0, 2, 10000)
#     ax.hexbin(x_coord, y_coord, gridsize=20, cmap="YlOrRd", id="density_map")
#     ax.set_xyt(x="X", y="Y", t="ax.hexbin()")
#     return fig, ax
# 
# 
# # =============================================================================
# # Area & Fill
# # =============================================================================
# def plot_fill_between(fig, ax, stx):
#     """ax.fill_between() - Filled area between curves."""
#     time = np.linspace(0, 10, 100)
#     upper = np.sin(time) + 0.5
#     lower = np.sin(time) - 0.5
#     ax.fill_between(time, lower, upper, alpha=0.3, id="confidence_region")
#     ax.plot(time, np.sin(time), "b-", id="mean_line")
#     ax.set_xyt(x="Time [s]", y="Value", t="ax.fill_between()")
#     return fig, ax
# 
# 
# def plot_fill_betweenx(fig, ax, stx):
#     """ax.fill_betweenx() - Horizontal filled area."""
#     y = np.linspace(0, 10, 100)
#     x1 = np.sin(y)
#     x2 = np.sin(y) + 1
#     ax.fill_betweenx(y, x1, x2, alpha=0.3, id="horizontal_band")
#     ax.set_xyt(x="Value", y="Depth [m]", t="ax.fill_betweenx()")
#     return fig, ax
# 
# 
# def plot_stx_fill_between(fig, ax, stx):
#     """ax.stx_fill_between() - Enhanced fill between."""
#     np.random.seed(42)
#     time = np.linspace(0, 5, 50)
#     mean = np.sin(time * 2)
#     std = 0.3 + 0.1 * np.abs(np.sin(time))
#     ax.stx_fill_between(time, mean - std, mean + std, id="uncertainty_band")
#     ax.plot(time, mean, "k-", linewidth=1, id="central_estimate")
#     ax.set_xyt(x="Time [s]", y="Estimate", t="ax.stx_fill_between()")
#     return fig, ax
# 
# 
# def plot_stx_fillv(fig, ax, stx):
#     """ax.stx_fillv() - Vertical fill regions."""
#     time = np.linspace(0, 10, 100)
#     signal = np.sin(time * 2)
#     ax.plot(time, signal, "b-", id="signal")
#     ax.stx_fillv([2, 6], [4, 8], alpha=0.3, color="red", id="stimulus_periods")
#     ax.set_xyt(x="Time [s]", y="Amplitude", t="ax.stx_fillv()")
#     return fig, ax
# 
# 
# # =============================================================================
# # Grid & Matrix
# # =============================================================================
# def plot_imshow(fig, ax, stx):
#     """ax.imshow() - Image display."""
#     np.random.seed(42)
#     image_data = np.random.rand(20, 20)
#     im = ax.imshow(image_data, cmap="viridis", id="random_image")
#     fig.colorbar(im, ax=ax, shrink=0.8)
#     ax.set_xyt(x="X", y="Y", t="ax.imshow()")
#     return fig, ax
# 
# 
# def plot_matshow(fig, ax, stx):
#     """ax.matshow() - Matrix display."""
#     np.random.seed(42)
#     correlation_matrix = np.corrcoef(np.random.rand(5, 100))
#     im = ax.matshow(correlation_matrix, cmap="RdBu_r", id="correlation_matrix")
#     fig.colorbar(im, ax=ax, shrink=0.8)
#     ax.set_xyt(x="Variable", y="Variable", t="ax.matshow()")
#     return fig, ax
# 
# 
# def plot_stx_imshow(fig, ax, stx):
#     """ax.stx_imshow() - Enhanced image display."""
#     np.random.seed(42)
#     brain_slice = np.random.rand(64, 64) * 100
#     ax.stx_imshow(brain_slice, cmap="hot", id="brain_activity")
#     ax.set_xyt(x="X [voxels]", y="Y [voxels]", t="ax.stx_imshow()")
#     return fig, ax
# 
# 
# def plot_stx_image(fig, ax, stx):
#     """ax.stx_image() - Image from 2D array."""
#     np.random.seed(42)
#     grayscale_image = np.random.rand(32, 32)
#     ax.stx_image(grayscale_image, id="grayscale_image")
#     ax.set_xyt(x="X", y="Y", t="ax.stx_image()")
#     return fig, ax
# 
# 
# def plot_stx_heatmap(fig, ax, stx):
#     """ax.stx_heatmap() - Annotated heatmap."""
#     np.random.seed(42)
#     data = np.random.rand(5, 5)
#     ax.stx_heatmap(data, id="annotated_heatmap")
#     ax.set_xyt(x="Feature", y="Sample", t="ax.stx_heatmap()")
#     return fig, ax
# 
# 
# def plot_stx_conf_mat(fig, ax, stx):
#     """ax.stx_conf_mat() - Confusion matrix."""
#     np.random.seed(42)
#     conf_matrix = np.array([[45, 5, 2], [3, 42, 8], [1, 6, 48]])
#     ax.stx_conf_mat(conf_matrix, id="classification_results")
#     ax.set_xyt(x="Predicted", y="Actual", t="ax.stx_conf_mat()")
#     return fig, ax
# 
# 
# # =============================================================================
# # Contours
# # =============================================================================
# def plot_contour(fig, ax, stx):
#     """ax.contour() - Contour lines."""
#     x = np.linspace(-3, 3, 100)
#     y = np.linspace(-3, 3, 100)
#     X, Y = np.meshgrid(x, y)
#     Z = np.sin(X) * np.cos(Y)
#     cs = ax.contour(X, Y, Z, levels=10, id="contour_lines")
#     ax.clabel(cs, inline=True, fontsize=6)
#     ax.set_xyt(x="X", y="Y", t="ax.contour()")
#     return fig, ax
# 
# 
# def plot_contourf(fig, ax, stx):
#     """ax.contourf() - Filled contours."""
#     x = np.linspace(-3, 3, 100)
#     y = np.linspace(-3, 3, 100)
#     X, Y = np.meshgrid(x, y)
#     Z = np.exp(-(X**2 + Y**2) / 2)
#     cf = ax.contourf(X, Y, Z, levels=20, cmap="viridis", id="gaussian_2d")
#     fig.colorbar(cf, ax=ax, shrink=0.8)
#     ax.set_xyt(x="X", y="Y", t="ax.contourf()")
#     return fig, ax
# 
# 
# def plot_stx_contour(fig, ax, stx):
#     """ax.stx_contour() - Enhanced contour plot."""
#     np.random.seed(42)
#     x = np.linspace(-2, 2, 50)
#     y = np.linspace(-2, 2, 50)
#     X, Y = np.meshgrid(x, y)
#     Z = X * np.exp(-(X**2) - Y**2)
#     ax.stx_contour(X, Y, Z, id="potential_field")
#     ax.set_xyt(x="X", y="Y", t="ax.stx_contour()")
#     return fig, ax
# 
# 
# # =============================================================================
# # Vector Fields
# # =============================================================================
# def plot_quiver(fig, ax, stx):
#     """ax.quiver() - Vector field."""
#     x = np.linspace(-2, 2, 10)
#     y = np.linspace(-2, 2, 10)
#     X, Y = np.meshgrid(x, y)
#     U = -Y
#     V = X
#     ax.quiver(X, Y, U, V, id="rotation_field")
#     ax.set_xyt(x="X", y="Y", t="ax.quiver()")
#     return fig, ax
# 
# 
# def plot_streamplot(fig, ax, stx):
#     """ax.streamplot() - Streamlines."""
#     x = np.linspace(-2, 2, 30)
#     y = np.linspace(-2, 2, 30)
#     X, Y = np.meshgrid(x, y)
#     U = -1 - X**2 + Y
#     V = 1 + X - Y**2
#     ax.streamplot(X, Y, U, V, density=1, id="flow_field")
#     ax.set_xyt(x="X", y="Y", t="ax.streamplot()")
#     return fig, ax
# 
# 
# # =============================================================================
# # Special
# # =============================================================================
# def plot_pie(fig, ax, stx):
#     """ax.pie() - Pie chart."""
#     sizes = [30, 25, 20, 15, 10]
#     labels = ["Category A", "Category B", "Category C", "Category D", "Category E"]
#     ax.pie(sizes, labels=labels, autopct="%1.1f%%", id="category_breakdown")
#     ax.set_title("ax.pie()")
#     return fig, ax
# 
# 
# def plot_stx_raster(fig, ax, stx):
#     """ax.stx_raster() - Spike raster plot."""
#     np.random.seed(42)
#     n_neurons, n_timepoints = 20, 1000
#     spike_times = [
#         np.sort(
#             np.random.choice(n_timepoints, np.random.randint(10, 50), replace=False)
#         )
#         for _ in range(n_neurons)
#     ]
#     ax.stx_raster(spike_times, id="neural_spikes")
#     ax.set_xyt(x="Time [ms]", y="Neuron", t="ax.stx_raster()")
#     return fig, ax
# 
# 
# def plot_stx_rectangle(fig, ax, stx):
#     """ax.stx_rectangle() - Rectangle annotation."""
#     np.random.seed(42)
#     time = np.linspace(0, 10, 200)
#     signal = np.sin(time * 2) + np.random.normal(0, 0.1, 200)
#     ax.plot(time, signal, "b-", alpha=0.7, id="signal")
#     ax.stx_rectangle(2, -0.5, 2, 1.5, alpha=0.3, color="red", id="region_of_interest")
#     ax.set_xyt(x="Time [s]", y="Amplitude", t="ax.stx_rectangle()")
#     return fig, ax
# 
# 
# # =============================================================================
# # Plot Registry
# # =============================================================================
# PLOT_FUNCTIONS = {
#     # Line
#     "plot": plot_plot,
#     "step": plot_step,
#     "stx_line": plot_stx_line,
#     "stx_shaded_line": plot_stx_shaded_line,
#     # Statistical
#     "stx_mean_std": plot_stx_mean_std,
#     "stx_mean_ci": plot_stx_mean_ci,
#     "stx_median_iqr": plot_stx_median_iqr,
#     "errorbar": plot_errorbar,
#     "stx_errorbar": plot_stx_errorbar,
#     # Distribution
#     "hist": plot_hist,
#     "hist2d": plot_hist2d,
#     "stx_kde": plot_stx_kde,
#     "stx_ecdf": plot_stx_ecdf,
#     "stx_joyplot": plot_stx_joyplot,
#     # Categorical
#     "bar": plot_bar,
#     "barh": plot_barh,
#     "stx_bar": plot_stx_bar,
#     "stx_barh": plot_stx_barh,
#     "boxplot": plot_boxplot,
#     "violinplot": plot_violinplot,
#     "stx_box": plot_stx_box,
#     "stx_violin": plot_stx_violin,
#     "stx_boxplot": plot_stx_boxplot,
#     "stx_violinplot": plot_stx_violinplot,
#     # Scatter
#     "scatter": plot_scatter,
#     "stx_scatter": plot_stx_scatter,
#     "stem": plot_stem,
#     "hexbin": plot_hexbin,
#     # Area
#     "fill_between": plot_fill_between,
#     "fill_betweenx": plot_fill_betweenx,
#     "stx_fill_between": plot_stx_fill_between,
#     "stx_fillv": plot_stx_fillv,
#     # Grid
#     "imshow": plot_imshow,
#     "matshow": plot_matshow,
#     "stx_imshow": plot_stx_imshow,
#     "stx_image": plot_stx_image,
#     "stx_heatmap": plot_stx_heatmap,
#     "stx_conf_mat": plot_stx_conf_mat,
#     # Contour
#     "contour": plot_contour,
#     "contourf": plot_contourf,
#     "stx_contour": plot_stx_contour,
#     # Vector
#     "quiver": plot_quiver,
#     "streamplot": plot_streamplot,
#     # Special
#     "pie": plot_pie,
#     "stx_raster": plot_stx_raster,
#     "stx_rectangle": plot_stx_rectangle,
# }
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/gallery/_plots.py
# --------------------------------------------------------------------------------
