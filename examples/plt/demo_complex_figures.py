#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 03:20:49 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/plt/demo_complex_figures.py


"""Demo: Complex multi-panel figures with multiple plot types per axis."""

import numpy as np
import scitex as stx
from scitex.plt.styles.presets import SCITEX_STYLE


@stx.session
def main(
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Demo: Complex multi-panel figures with multiple plot types per axis."""

    logger.info("=" * 70)
    logger.info(
        "Demo: Complex Multi-Panel Figures (Multiple Plot Types per Axis)"
    )
    logger.info("=" * 70)

    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    x = np.linspace(0, 10, n_samples)
    y1 = np.sin(x) + np.random.normal(0, 0.1, n_samples)
    y2 = np.cos(x) + np.random.normal(0, 0.1, n_samples)
    groups = ["Control", "Treatment A", "Treatment B"]
    group_data = [np.random.normal(loc, 0.5, 50) for loc in [0, 1, 2]]

    # ========================================
    # Example 1: Multi-type per axis - Line + Scatter + Fill
    # ========================================
    logger.info("\n[01] Multi-type per axis: Line + Scatter + Fill")

    STYLE = SCITEX_STYLE.copy()
    # Note: nrows/ncols must be passed as positional args, not via kwargs
    # STYLE["figsize_mm"] = (160, 140)  # figsize_mm not directly supported

    fig, axes = stx.plt.subplots(2, 2, **STYLE)

    # Panel A: Line + Scatter + Vertical fill regions on same axis
    axes[0, 0].plot(x, y1, "-", linewidth=1, label="Trend", id="trend-line")
    axes[0, 0].plot(
        x,
        y1,
        "o",
        markersize=2,
        alpha=0.5,
        label="Data points",
        id="data-points",
    )
    axes[0, 0].stx_fillv([2, 6], [4, 8], alpha=0.2, color="gray", id="regions")
    axes[0, 0].set_title("A) Line + Scatter + Fill Regions")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Signal [a.u.]")
    axes[0, 0].legend()

    # Panel B: Multiple lines + error bars on same axis
    y_mean = np.sin(x)
    y_err = 0.2 + 0.1 * np.random.rand(n_samples)
    axes[0, 1].plot(x, y_mean, "-", color="blue", label="Mean", id="mean-line")
    axes[0, 1].fill_between(
        x, y_mean - y_err, y_mean + y_err, alpha=0.3, id="uncertainty"
    )
    axes[0, 1].errorbar(
        x[::10],
        y_mean[::10],
        yerr=y_err[::10],
        fmt="o",
        markersize=4,
        capsize=2,
        label="Error bars",
        id="errorbars",
    )
    axes[0, 1].set_title("B) Line + Fill + Error Bars")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Value [a.u.]")
    axes[0, 1].legend()

    # Panel C: Box + Strip + Scatter overlay
    for i, (data, label) in enumerate(zip(group_data, groups)):
        # Add jittered points behind boxes
        jitter = np.random.normal(0, 0.05, len(data))
        axes[1, 0].scatter(
            np.full_like(data, i) + jitter,
            data,
            alpha=0.3,
            s=10,
            id=f"scatter-{label}",
        )
    axes[1, 0].stx_box(group_data, labels=groups, id="boxplot")
    axes[1, 0].set_title("C) Box + Scatter Overlay")
    axes[1, 0].set_xlabel("Group")
    axes[1, 0].set_ylabel("Value [a.u.]")

    # Panel D: Histogram + KDE overlay
    hist_data = np.concatenate(group_data)
    axes[1, 1].hist(
        hist_data,
        bins=25,
        density=True,
        alpha=0.6,
        label="Histogram",
        id="hist",
    )
    # Add KDE line
    from scipy import stats

    kde_x = np.linspace(hist_data.min() - 1, hist_data.max() + 1, 200)
    kde = stats.gaussian_kde(hist_data)
    axes[1, 1].plot(
        kde_x, kde(kde_x), "-", linewidth=2, label="KDE", id="kde-overlay"
    )
    # Add vertical lines for means
    for i, (data, label) in enumerate(zip(group_data, groups)):
        axes[1, 1].axvline(
            np.mean(data),
            linestyle="--",
            alpha=0.7,
            label=f"{label} mean",
            id=f"mean-{label}",
        )
    axes[1, 1].set_title("D) Histogram + KDE + Mean Lines")
    axes[1, 1].set_xlabel("Value [a.u.]")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend(fontsize=5)

    stx.io.save(fig, "./png/01_multi_type_per_axis.png")
    fig.close()

    # ========================================
    # Example 2: Scientific Figure - Multiple Analyses
    # ========================================
    logger.info(
        "\n[02] Scientific figure: Time series + Statistics + Correlation"
    )

    # Note: nrows/ncols must be passed as positional args

    fig, axes = stx.plt.subplots(2, 3, **STYLE)

    # Panel A: Multi-channel time series with annotations
    time = np.linspace(0, 5, 500)
    ch1 = np.sin(2 * np.pi * time) + 0.3 * np.random.randn(500)
    ch2 = np.sin(2 * np.pi * time + np.pi / 4) + 0.3 * np.random.randn(500) + 3
    ch3 = np.sin(2 * np.pi * time + np.pi / 2) + 0.3 * np.random.randn(500) + 6

    axes[0, 0].plot(time, ch1, label="Ch1", id="ch1")
    axes[0, 0].plot(time, ch2, label="Ch2", id="ch2")
    axes[0, 0].plot(time, ch3, label="Ch3", id="ch3")
    # Add stimulus period
    axes[0, 0].stx_fillv(
        [1, 3], [2, 4], alpha=0.2, color="red", id="stim-period"
    )
    axes[0, 0].stx_rectangle(
        1,
        ch1.min() - 0.5,
        1,
        ch3.max() - ch1.min() + 1,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        id="stim-box",
    )
    axes[0, 0].set_title("A) Multi-channel Recording")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Amplitude [mV]")
    axes[0, 0].legend(loc="upper right")

    # Panel B: Shaded line with individual traces
    data_2d = np.random.randn(20, 100)
    mean_trace = data_2d.mean(axis=0)
    std_trace = data_2d.std(axis=0)
    trace_x = np.arange(100)

    # Plot individual traces faintly
    for i in range(min(5, len(data_2d))):
        axes[0, 1].plot(
            trace_x, data_2d[i], alpha=0.2, color="gray", id=f"trace-{i}"
        )
    # Plot mean with shading
    axes[0, 1].stx_shaded_line(
        trace_x,
        mean_trace - std_trace,
        mean_trace,
        mean_trace + std_trace,
        id="mean-std",
    )
    axes[0, 1].set_title("B) Individual Traces + Mean +/- Std")
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].set_ylabel("Value [a.u.]")

    # Panel C: Heatmap with contour overlay
    xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    zz = np.exp(-(xx**2 + yy**2) / 2) + 0.5 * np.exp(
        -((xx - 1) ** 2 + (yy - 1) ** 2) / 0.5
    )
    axes[0, 2].imshow(
        zz, extent=[-3, 3, -3, 3], origin="lower", cmap="viridis", id="heatmap"
    )
    axes[0, 2].contour(
        xx, yy, zz, levels=5, colors="white", linewidths=0.5, id="contours"
    )
    axes[0, 2].set_title("C) Heatmap + Contour Overlay")
    axes[0, 2].set_xlabel("X [a.u.]")
    axes[0, 2].set_ylabel("Y [a.u.]")

    # Panel D: Violin + Strip + Mean markers
    for i, (data, label) in enumerate(zip(group_data, groups)):
        jitter = np.random.normal(0, 0.08, len(data))
        axes[1, 0].scatter(
            np.full_like(data, i) + jitter,
            data,
            alpha=0.4,
            s=8,
            id=f"strip-{label}",
        )
    axes[1, 0].stx_violin(group_data, labels=groups, id="violin")
    # Add mean markers
    for i, data in enumerate(group_data):
        axes[1, 0].plot(
            i,
            np.mean(data),
            "D",
            color="red",
            markersize=6,
            id=f"mean-marker-{i}",
        )
    axes[1, 0].set_title("D) Violin + Strip + Mean")
    axes[1, 0].set_xlabel("Group")
    axes[1, 0].set_ylabel("Value [a.u.]")

    # Panel E: Scatter + Regression + CI
    scatter_x = np.random.randn(50)
    scatter_y = 0.8 * scatter_x + 0.5 * np.random.randn(50)
    axes[1, 1].scatter(scatter_x, scatter_y, alpha=0.6, s=20, id="scatter")
    # Add regression line
    slope, intercept = np.polyfit(scatter_x, scatter_y, 1)
    reg_x = np.linspace(scatter_x.min(), scatter_x.max(), 100)
    reg_y = slope * reg_x + intercept
    axes[1, 1].plot(
        reg_x,
        reg_y,
        "-",
        color="red",
        linewidth=2,
        label=f"y={slope:.2f}x+{intercept:.2f}",
        id="regression",
    )
    # Add confidence band (simplified)
    ci = 0.3
    axes[1, 1].fill_between(
        reg_x, reg_y - ci, reg_y + ci, alpha=0.2, color="red", id="ci-band"
    )
    axes[1, 1].set_title("E) Scatter + Regression + CI")
    axes[1, 1].set_xlabel("X [a.u.]")
    axes[1, 1].set_ylabel("Y [a.u.]")
    axes[1, 1].legend()

    # Panel F: Bar + Error bars + Significance markers
    means = [np.mean(d) for d in group_data]
    stds = [np.std(d) for d in group_data]
    bar_x = np.arange(len(groups))
    axes[1, 2].bar(bar_x, means, yerr=stds, capsize=3, alpha=0.7, id="bars")
    # Add significance bracket
    y_max = max(means) + max(stds) + 0.3
    axes[1, 2].plot(
        [0, 0, 2, 2],
        [y_max, y_max + 0.1, y_max + 0.1, y_max],
        "k-",
        linewidth=1,
        id="sig-bracket",
    )
    axes[1, 2].text(
        1, y_max + 0.15, "***", ha="center", va="bottom", fontsize=10
    )
    axes[1, 2].set_xticks(bar_x)
    axes[1, 2].set_xticklabels(groups)
    axes[1, 2].set_title("F) Bar + Error + Significance")
    axes[1, 2].set_xlabel("Group")
    axes[1, 2].set_ylabel("Mean Value [a.u.]")

    stx.io.save(fig, "./png/02_scientific_figure.png")
    fig.close()

    # ========================================
    # Example 3: Neuroscience-style Figure
    # ========================================
    logger.info("\n[03] Neuroscience figure: Raster + PSTH + Waveforms")

    # Note: nrows/ncols must be passed as positional args

    fig, axes = stx.plt.subplots(3, 2, **STYLE)

    # Generate spike data
    n_trials = 30
    spike_trains = []
    for trial in range(n_trials):
        # Higher firing rate during stimulus (0.5-1.5s)
        pre_stim = np.random.uniform(0, 0.5, np.random.poisson(5))
        stim = np.random.uniform(0.5, 1.5, np.random.poisson(15))
        post_stim = np.random.uniform(1.5, 2.0, np.random.poisson(5))
        spike_trains.append(
            np.sort(np.concatenate([pre_stim, stim, post_stim]))
        )

    # Panel A: Raster + stimulus period
    axes[0, 0].stx_raster(spike_trains, id="raster")
    axes[0, 0].stx_fillv([0.5], [1.5], alpha=0.2, color="yellow", id="stim")
    axes[0, 0].axvline(
        0.5, color="green", linestyle="--", linewidth=1, id="stim-onset"
    )
    axes[0, 0].axvline(
        1.5, color="red", linestyle="--", linewidth=1, id="stim-offset"
    )
    axes[0, 0].set_title("A) Raster Plot")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Trial")

    # Panel B: PSTH (histogram) + smoothed firing rate
    all_spikes = np.concatenate(spike_trains)
    hist_counts, bin_edges = np.histogram(all_spikes, bins=40, range=(0, 2))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    firing_rate = hist_counts / n_trials / (bin_edges[1] - bin_edges[0])  # Hz

    axes[0, 1].bar(
        bin_centers,
        firing_rate,
        width=bin_edges[1] - bin_edges[0],
        alpha=0.6,
        id="psth",
    )
    # Smoothed rate
    from scipy.ndimage import gaussian_filter1d

    smooth_rate = gaussian_filter1d(firing_rate.astype(float), sigma=2)
    axes[0, 1].plot(
        bin_centers,
        smooth_rate,
        "-",
        color="red",
        linewidth=2,
        label="Smoothed",
        id="smooth-rate",
    )
    axes[0, 1].stx_fillv(
        [0.5], [1.5], alpha=0.2, color="yellow", id="stim-period"
    )
    axes[0, 1].set_title("B) PSTH + Smoothed Rate")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Firing Rate [Hz]")
    axes[0, 1].legend()

    # Panel C: Spike waveforms overlay
    n_waveforms = 50
    waveform_time = np.linspace(0, 1.5, 30)  # 1.5ms waveform
    for i in range(n_waveforms):
        # Generate realistic-looking waveform with noise
        waveform = -np.exp(
            -((waveform_time - 0.3) ** 2) / 0.02
        ) + 0.3 * np.exp(-((waveform_time - 0.8) ** 2) / 0.05)
        waveform += 0.05 * np.random.randn(len(waveform_time))
        axes[1, 0].plot(
            waveform_time,
            waveform,
            alpha=0.2,
            color="blue",
            id=f"waveform-{i}",
        )
    # Mean waveform
    mean_waveform = -np.exp(
        -((waveform_time - 0.3) ** 2) / 0.02
    ) + 0.3 * np.exp(-((waveform_time - 0.8) ** 2) / 0.05)
    axes[1, 0].plot(
        waveform_time,
        mean_waveform,
        "-",
        color="red",
        linewidth=2,
        label="Mean",
        id="mean-waveform",
    )
    axes[1, 0].set_title("C) Spike Waveforms")
    axes[1, 0].set_xlabel("Time [ms]")
    axes[1, 0].set_ylabel("Amplitude [a.u.]")
    axes[1, 0].legend()

    # Panel D: ISI histogram + Exponential fit
    isis = []
    for train in spike_trains:
        if len(train) > 1:
            isis.extend(np.diff(train) * 1000)  # Convert to ms
    isis = np.array(isis)

    axes[1, 1].hist(isis, bins=30, density=True, alpha=0.6, id="isi-hist")
    # Fit exponential
    isi_x = np.linspace(0, isis.max(), 100)
    mean_isi = np.mean(isis)
    exp_fit = (1 / mean_isi) * np.exp(-isi_x / mean_isi)
    axes[1, 1].plot(
        isi_x,
        exp_fit,
        "-",
        color="red",
        linewidth=2,
        label=f"Exp fit (tau={mean_isi:.1f}ms)",
        id="exp-fit",
    )
    axes[1, 1].set_title("D) ISI Distribution")
    axes[1, 1].set_xlabel("ISI [ms]")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()

    # Panel E: Tuning curve with error bars and fit
    orientations = np.arange(0, 180, 20)
    responses = (
        10
        + 8 * np.exp(-((orientations - 90) ** 2) / (2 * 30**2))
        + np.random.randn(len(orientations))
    )
    response_sem = 1 + 0.5 * np.random.rand(len(orientations))

    axes[2, 0].errorbar(
        orientations,
        responses,
        yerr=response_sem,
        fmt="o",
        capsize=3,
        id="tuning-data",
    )
    # Gaussian fit
    fit_x = np.linspace(0, 180, 100)
    fit_y = 10 + 8 * np.exp(-((fit_x - 90) ** 2) / (2 * 30**2))
    axes[2, 0].plot(
        fit_x,
        fit_y,
        "-",
        color="red",
        linewidth=2,
        label="Gaussian fit",
        id="tuning-fit",
    )
    axes[2, 0].axvline(
        90, color="gray", linestyle="--", alpha=0.5, id="preferred"
    )
    axes[2, 0].set_title("E) Orientation Tuning Curve")
    axes[2, 0].set_xlabel("Orientation [deg]")
    axes[2, 0].set_ylabel("Response [spikes/s]")
    axes[2, 0].legend()

    # Panel F: Population activity heatmap + mean trace
    n_neurons = 20
    n_time = 100
    pop_activity = np.zeros((n_neurons, n_time))
    for i in range(n_neurons):
        peak_time = 30 + np.random.randint(-10, 10)
        pop_activity[i] = np.exp(
            -((np.arange(n_time) - peak_time) ** 2) / (2 * 10**2)
        )
        pop_activity[i] += 0.1 * np.random.randn(n_time)

    im = axes[2, 1].imshow(
        pop_activity, aspect="auto", cmap="hot", id="pop-heatmap"
    )
    # Add mean population trace on secondary y-axis (simplified - just overlay)
    mean_pop = pop_activity.mean(axis=0) * n_neurons  # Scale for visibility
    ax2 = axes[2, 1].twinx()
    ax2.plot(
        mean_pop, "-", color="cyan", linewidth=2, label="Mean", id="mean-pop"
    )
    ax2.set_ylabel("Mean Activity")
    ax2._axis_mpl.tick_params(
        axis="y", labelcolor="cyan"
    )  # Color the y-tick labels
    ax2._axis_mpl.yaxis.label.set_color("cyan")  # Color the y-label
    axes[2, 1].set_title("F) Population Activity")
    axes[2, 1].set_xlabel("Time [bins]")
    axes[2, 1].set_ylabel("Neuron #")

    stx.io.save(fig, "./png/03_neuroscience_figure.png")
    fig.close()

    logger.info("\n" + "=" * 70)
    logger.info("All complex figure demos completed")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    main()

# EOF
