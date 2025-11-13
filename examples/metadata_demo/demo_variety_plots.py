#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate variety of plots with QR-encoded metadata

Creates multiple types of scientific figures with embedded metadata and QR codes
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from PIL import Image

# Add scitex to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import scitex as stx

# Try to import QR code library
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    print("⚠ qrcode not available. Install with: pip install qrcode[pil]")
    sys.exit(1)

# SciTeX logo path
LOGO_PATH = Path(__file__).parent.parent.parent / "docs" / "scitex_logos" / "vectorstock" / "vectorstock_38853699-navy-inverted-192x192.png"


def add_metadata_footer(fig, metadata, title=""):
    """Add metadata footer with QR code and logo to a figure."""

    # Adjust figure to make room for metadata
    fig.subplots_adjust(bottom=0.25)

    # Create footer area
    ax_footer = fig.add_axes([0.1, 0.02, 0.8, 0.18])
    ax_footer.axis('off')

    # Generate QR code
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=3,
        border=1,
    )
    qr.add_data(metadata_json)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")

    # QR code (left)
    ax_qr = fig.add_axes([0.02, 0.02, 0.15, 0.18])
    ax_qr.imshow(qr_img, cmap='gray')
    ax_qr.axis('off')

    # Metadata text (center)
    ax_meta = fig.add_axes([0.19, 0.02, 0.64, 0.18])
    ax_meta.axis('off')

    # Separator
    ax_meta.text(0.5, 0.95, '✂ ' + '─' * 25 + ' METADATA ' + '─' * 25 + ' ✂',
                ha='center', va='top', fontsize=8,
                family='monospace', color='#666',
                transform=ax_meta.transAxes)

    # Format metadata
    meta_lines = []
    items = list(metadata.items())
    for i in range(0, len(items), 2):
        line = f"{items[i][0]}: {items[i][1]}"
        if i + 1 < len(items):
            line += f"  |  {items[i+1][0]}: {items[i+1][1]}"
        meta_lines.append(line)

    meta_text = '\n'.join(meta_lines)
    ax_meta.text(0.05, 0.75, meta_text,
                ha='left', va='top', fontsize=7,
                family='monospace', color='#333',
                transform=ax_meta.transAxes)

    # Logo (right)
    ax_logo = fig.add_axes([0.85, 0.02, 0.13, 0.18])
    if LOGO_PATH.exists():
        logo_img = Image.open(LOGO_PATH)
        ax_logo.imshow(logo_img, extent=[0.35, 0.65, 0.35, 0.65])
    else:
        ax_logo.text(0.5, 0.5, 'S', ha='center', va='center',
                    fontsize=6, color='#1e3a5f', weight='bold')
    ax_logo.set_xlim(0, 1)
    ax_logo.set_ylim(0, 1)
    ax_logo.axis('off')


def plot_line_timeseries():
    """Generate line plot with time series data."""

    print("\n" + "=" * 70)
    print("1. Line Plot: Neural Time Series")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate data
    t = np.linspace(0, 3, 1500)
    signal = np.sin(2 * np.pi * 3 * t) * np.exp(-t / 2)
    noise = np.random.normal(0, 0.08, len(t))

    ax.plot(t, signal + noise, 'b-', linewidth=0.8, alpha=0.6, label='Recorded')
    ax.plot(t, signal, 'r--', linewidth=2, label='Ground Truth')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Amplitude (μV)', fontsize=11)
    ax.set_title('LFP Recording - Hippocampus CA1', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    metadata = {
        'experiment': 'lfp_recording_CA1',
        'subject': 'Rat_A12',
        'region': 'Hippocampus_CA1',
        'electrode': 'tetrode_03',
        'sampling_rate': 500,
        'session': '2024-11-14',
        'analysis': 'raw_LFP',
        'version': stx.__version__
    }

    add_metadata_footer(fig, metadata)

    output_path = Path(__file__).parent / "output" / "variety_01_line_timeseries.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stx.io.save(fig, str(output_path), metadata=metadata)
    print(f"✓ Saved to: {output_path}")
    plt.close()

    return output_path


def plot_scatter():
    """Generate scatter plot."""

    print("\n" + "=" * 70)
    print("2. Scatter Plot: Neural Clusters")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate clustered data
    np.random.seed(42)
    n_points = 300

    # Cluster 1
    x1 = np.random.normal(2, 0.5, n_points // 3)
    y1 = np.random.normal(2, 0.5, n_points // 3)

    # Cluster 2
    x2 = np.random.normal(4, 0.4, n_points // 3)
    y2 = np.random.normal(5, 0.4, n_points // 3)

    # Cluster 3
    x3 = np.random.normal(6, 0.6, n_points // 3)
    y3 = np.random.normal(2.5, 0.6, n_points // 3)

    ax.scatter(x1, y1, c='#1f77b4', s=30, alpha=0.6, label='Pyramidal')
    ax.scatter(x2, y2, c='#ff7f0e', s=30, alpha=0.6, label='Interneuron')
    ax.scatter(x3, y3, c='#2ca02c', s=30, alpha=0.6, label='Unclassified')

    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title('Spike Sorting - Principal Component Analysis', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    metadata = {
        'experiment': 'spike_sorting_mEC',
        'subject': 'Mouse_B07',
        'region': 'medial_entorhinal_cortex',
        'n_clusters': 3,
        'n_spikes': n_points,
        'method': 'PCA_clustering',
        'date': '2024-11-14',
        'version': stx.__version__
    }

    add_metadata_footer(fig, metadata)

    output_path = Path(__file__).parent / "output" / "variety_02_scatter_clusters.png"
    stx.io.save(fig, str(output_path), metadata=metadata)
    print(f"✓ Saved to: {output_path}")
    plt.close()

    return output_path


def plot_bar_chart():
    """Generate bar chart."""

    print("\n" + "=" * 70)
    print("3. Bar Chart: Firing Rates by Region")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))

    regions = ['CA1', 'CA3', 'DG', 'mEC', 'lEC']
    firing_rates = [8.5, 12.3, 5.7, 15.2, 9.8]
    errors = [1.2, 1.8, 0.9, 2.1, 1.4]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(regions, firing_rates, yerr=errors, color=colors, alpha=0.7,
                   capsize=5, edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Brain Region', fontsize=11)
    ax.set_ylabel('Mean Firing Rate (Hz)', fontsize=11)
    ax.set_title('Regional Firing Rates - Spatial Navigation Task', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    metadata = {
        'experiment': 'spatial_navigation',
        'task': 'T_maze',
        'n_sessions': 12,
        'n_subjects': 8,
        'regions': ','.join(regions),
        'metric': 'mean_firing_rate_Hz',
        'date': '2024-11-14',
        'version': stx.__version__
    }

    add_metadata_footer(fig, metadata)

    output_path = Path(__file__).parent / "output" / "variety_03_bar_firing_rates.png"
    stx.io.save(fig, str(output_path), metadata=metadata)
    print(f"✓ Saved to: {output_path}")
    plt.close()

    return output_path


def plot_heatmap():
    """Generate heatmap."""

    print("\n" + "=" * 70)
    print("4. Heatmap: Place Field Activity")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate place field data
    np.random.seed(42)
    x_bins, y_bins = 20, 20
    place_field = np.zeros((y_bins, x_bins))

    # Add hot spots
    for _ in range(3):
        cx, cy = np.random.randint(2, 18, 2)
        for i in range(y_bins):
            for j in range(x_bins):
                dist = np.sqrt((i - cy)**2 + (j - cx)**2)
                place_field[i, j] += 10 * np.exp(-dist / 3)

    place_field += np.random.normal(0, 1, place_field.shape)
    place_field = np.maximum(place_field, 0)

    im = ax.imshow(place_field, cmap='hot', aspect='auto', interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Firing Rate (Hz)')

    ax.set_xlabel('X Position (cm)', fontsize=11)
    ax.set_ylabel('Y Position (cm)', fontsize=11)
    ax.set_title('Place Field - Grid Cell Activity', fontsize=13, fontweight='bold')
    ax.set_xticks([0, 5, 10, 15, 19])
    ax.set_xticklabels(['0', '25', '50', '75', '100'])
    ax.set_yticks([0, 5, 10, 15, 19])
    ax.set_yticklabels(['0', '25', '50', '75', '100'])

    metadata = {
        'experiment': 'place_field_mapping',
        'subject': 'Rat_C03',
        'cell_type': 'grid_cell',
        'region': 'medial_entorhinal_cortex',
        'arena_size': '100x100_cm',
        'bins': f'{x_bins}x{y_bins}',
        'date': '2024-11-14',
        'version': stx.__version__
    }

    add_metadata_footer(fig, metadata)

    output_path = Path(__file__).parent / "output" / "variety_04_heatmap_place_field.png"
    stx.io.save(fig, str(output_path), metadata=metadata)
    print(f"✓ Saved to: {output_path}")
    plt.close()

    return output_path


def plot_multi_panel():
    """Generate multi-panel figure."""

    print("\n" + "=" * 70)
    print("5. Multi-panel: Comprehensive Analysis")
    print("=" * 70)

    fig = plt.figure(figsize=(12, 8))

    # Panel A: Time series
    ax1 = plt.subplot(2, 2, 1)
    t = np.linspace(0, 2, 1000)
    signal = np.sin(2 * np.pi * 4 * t) * np.exp(-t)
    ax1.plot(t, signal, 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('A. Neural Oscillation', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Panel B: Spectrum
    ax2 = plt.subplot(2, 2, 2)
    freqs = np.linspace(0, 50, 100)
    power = 100 * np.exp(-(freqs - 4)**2 / 10) + 20 * np.exp(-(freqs - 8)**2 / 5)
    ax2.plot(freqs, power, 'r-', linewidth=2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power (dB)')
    ax2.set_title('B. Power Spectrum', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Panel C: Histogram
    ax3 = plt.subplot(2, 2, 3)
    data = np.random.gamma(2, 2, 1000)
    ax3.hist(data, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Inter-spike Interval (ms)')
    ax3.set_ylabel('Count')
    ax3.set_title('C. ISI Distribution', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel D: Phase plot
    ax4 = plt.subplot(2, 2, 4)
    theta = np.linspace(0, 4 * np.pi, 200)
    r = 1 + 0.5 * np.sin(8 * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax4.plot(x, y, 'purple', linewidth=2)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('D. Phase Portrait', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')

    plt.suptitle('Comprehensive Neural Analysis', fontsize=14, fontweight='bold', y=0.98)

    metadata = {
        'experiment': 'comprehensive_analysis',
        'subject': 'Mouse_D15',
        'panels': 'A:oscillation,B:spectrum,C:ISI,D:phase',
        'region': 'prefrontal_cortex',
        'n_trials': 50,
        'duration': '120_min',
        'date': '2024-11-14',
        'version': stx.__version__
    }

    add_metadata_footer(fig, metadata)

    output_path = Path(__file__).parent / "output" / "variety_05_multi_panel.png"
    stx.io.save(fig, str(output_path), metadata=metadata)
    print(f"✓ Saved to: {output_path}")
    plt.close()

    return output_path


def main():
    """Generate all variety plots."""

    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "SciTeX Variety Plots with QR Metadata" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")

    if not QR_AVAILABLE:
        print("\n✗ qrcode library not available. Please install: pip install qrcode[pil]")
        return 1

    try:
        paths = []
        paths.append(plot_line_timeseries())
        paths.append(plot_scatter())
        paths.append(plot_bar_chart())
        paths.append(plot_heatmap())
        paths.append(plot_multi_panel())

        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print("\n✓ Generated 5 variety plots with QR-encoded metadata!")
        print("\nOutput files:")
        for i, path in enumerate(paths, 1):
            print(f"  {i}. {path.name}")

        print("\nAll figures include:")
        print("  • Embedded metadata in PNG file")
        print("  • Visual QR code for scanning")
        print("  • Minimal 16×16px SciTeX logo")
        print("  • Human-readable metadata text")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
