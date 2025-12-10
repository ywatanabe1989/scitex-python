#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: examples/vis/gui_editors/demo_03_neuroscience.py

"""
Demo 03: Neuroscience figure (3x2) - Raster + PSTH + Waveforms

Port: 5053

Usage:
    ./demo_03_neuroscience.py              # Flask backend (default)
    ./demo_03_neuroscience.py --backend qt # Qt backend
"""

import numpy as np
from pathlib import Path
from typing import Literal
import scitex as stx
from scitex.plt.styles.presets import SCITEX_STYLE

Backend = Literal["auto", "flask", "dearpygui", "qt", "tkinter", "mpl"]
PORT = 5053


def create_figure(output_dir: Path) -> Path:
    """Create neuroscience figure: Raster + PSTH + Waveforms."""
    from scipy.ndimage import gaussian_filter1d

    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    STYLE = SCITEX_STYLE.copy()
    fig, axes = stx.plt.subplots(3, 2, **STYLE)

    # Generate spike data
    n_trials = 30
    spike_trains = []
    for trial in range(n_trials):
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

    # Panel B: PSTH + smoothed firing rate
    all_spikes = np.concatenate(spike_trains)
    hist_counts, bin_edges = np.histogram(all_spikes, bins=40, range=(0, 2))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    firing_rate = hist_counts / n_trials / (bin_edges[1] - bin_edges[0])

    axes[0, 1].bar(
        bin_centers,
        firing_rate,
        width=bin_edges[1] - bin_edges[0],
        alpha=0.6,
        id="psth",
    )
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
    waveform_time = np.linspace(0, 1.5, 30)
    for i in range(n_waveforms):
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
            isis.extend(np.diff(train) * 1000)
    isis = np.array(isis)

    axes[1, 1].hist(isis, bins=30, density=True, alpha=0.6, id="isi-hist")
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

    # Panel F: Population activity heatmap
    n_neurons = 20
    n_time = 100
    pop_activity = np.zeros((n_neurons, n_time))
    for i in range(n_neurons):
        peak_time = 30 + np.random.randint(-10, 10)
        pop_activity[i] = np.exp(
            -((np.arange(n_time) - peak_time) ** 2) / (2 * 10**2)
        )
        pop_activity[i] += 0.1 * np.random.randn(n_time)

    axes[2, 1].imshow(
        pop_activity, aspect="auto", cmap="hot", id="pop-heatmap"
    )
    axes[2, 1].set_title("F) Population Activity")
    axes[2, 1].set_xlabel("Time [bins]")
    axes[2, 1].set_ylabel("Neuron #")

    png_path = output_dir / "03_neuroscience_figure.png"
    stx.io.save(fig, png_path)
    fig.close()

    return png_path.with_suffix(".json")


@stx.session
def main(
    backend: Backend = "flask",
    CONFIG=stx.INJECTED,
    logger=stx.INJECTED,
):
    """
    Demo 03: Neuroscience figure (3x2)

    Features:
        - Raster Plot with stimulus period
        - PSTH + Smoothed Firing Rate
        - Spike Waveforms overlay
        - ISI Distribution + Exponential fit
        - Orientation Tuning Curve
        - Population Activity Heatmap

    Parameters
    ----------
    backend : str
        GUI backend: flask, dearpygui, qt, tkinter, mpl

    Port: 5053
    """
    out = Path(CONFIG.SDIR_OUT)

    logger.info("=" * 60)
    logger.info("Demo 03: Neuroscience figure (3x2)")
    logger.info(f"Port: {PORT}")
    logger.info("=" * 60)

    json_path = create_figure(out)
    logger.info(f"Created: {json_path}")

    logger.info(f"\nLaunching editor (backend={backend}, port={PORT})...")
    stx.vis.edit(str(json_path), backend=backend, port=PORT)

    return 0


if __name__ == "__main__":
    main()

# EOF
