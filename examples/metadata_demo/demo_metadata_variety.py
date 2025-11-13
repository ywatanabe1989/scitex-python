#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test variety of metadata sizes with optimization

Creates the same line plot with different amounts of metadata to test:
- QR code size adaptation
- Layout optimization
- Placement strategies
- Minimal footprint
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import scitex as stx

try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    print("⚠ qrcode not available")
    sys.exit(1)

LOGO_PATH = Path(__file__).parent.parent.parent / "docs" / "scitex_logos" / "vectorstock" / "vectorstock_38853699-navy-inverted-192x192.png"


def create_base_plot():
    """Create the base line plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    t = np.linspace(0, 2, 1000)
    signal = np.sin(2 * np.pi * 5 * t) * np.exp(-t / 2)
    noise = np.random.normal(0, 0.08, len(t))

    ax.plot(t, signal + noise, 'b-', linewidth=0.8, alpha=0.6, label='Recorded')
    ax.plot(t, signal, 'r--', linewidth=2, label='True')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Amplitude (μV)', fontsize=11)
    ax.set_title('Neural Signal Analysis', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def add_compact_footer(fig, metadata, url="https://scitex.ai"):
    """Add ultra-compact metadata footer."""

    fig.subplots_adjust(bottom=0.20)

    # Generate QR code
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=2,  # Ultra-small
        border=1,
    )
    qr.add_data(metadata_json)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")

    qr_version = qr.version
    qr_size_px = qr.modules_count * 2

    # QR code (left corner)
    qr_width = 0.08
    ax_qr = fig.add_axes([0.01, 0.01, qr_width, 0.15])
    ax_qr.imshow(qr_img, cmap='gray')
    ax_qr.axis('off')

    # Metadata text (center)
    meta_left = 0.01 + qr_width + 0.01
    meta_width = 0.88 - qr_width
    ax_meta = fig.add_axes([meta_left, 0.01, meta_width, 0.15])
    ax_meta.axis('off')

    # Compact metadata display
    meta_lines = []
    items = list(metadata.items())

    # Pack into 2-3 columns depending on size
    if len(items) <= 4:
        cols = 2
    elif len(items) <= 8:
        cols = 3
    else:
        cols = 4

    for i in range(0, len(items), cols):
        parts = []
        for j in range(cols):
            if i + j < len(items):
                k, v = items[i + j]
                parts.append(f"{k}:{v}")
        meta_lines.append(" | ".join(parts))

    meta_text = '\n'.join(meta_lines)

    ax_meta.text(0.01, 0.90, meta_text,
                ha='left', va='top', fontsize=6,
                family='monospace', color='#333',
                transform=ax_meta.transAxes)

    # Signature (bottom right) - ultra compact
    ax_meta.text(0.99, 0.02, url,
                ha='right', va='bottom', fontsize=6,
                color='#1e3a5f',
                transform=ax_meta.transAxes)

    # Logo (right corner) - tiny
    logo_width = 0.04
    ax_logo = fig.add_axes([0.95, 0.01, logo_width, 0.15])
    if LOGO_PATH.exists():
        logo_img = Image.open(LOGO_PATH)
        ax_logo.imshow(logo_img, extent=[0.2, 0.8, 0.2, 0.8])
    else:
        ax_logo.text(0.5, 0.5, 'S', ha='center', va='center',
                    fontsize=5, color='#1e3a5f', weight='bold')
    ax_logo.set_xlim(0, 1)
    ax_logo.set_ylim(0, 1)
    ax_logo.axis('off')

    return qr_version, qr_size_px, len(metadata_json)


def test_minimal_metadata():
    """Test with minimal metadata (2-3 fields)."""

    print("\n" + "=" * 70)
    print("1. MINIMAL Metadata (2-3 fields)")
    print("=" * 70)

    fig, ax = create_base_plot()

    metadata = {
        'exp': 's01',
        'date': '2024-11-14',
    }

    qr_ver, qr_px, json_len = add_compact_footer(fig, metadata)

    output = Path(__file__).parent / "output" / "meta_variety_1_minimal.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    stx.io.save(fig, str(output), metadata=metadata)

    print(f"✓ Metadata: {len(metadata)} fields, {json_len} chars")
    print(f"✓ QR: Version {qr_ver}, {qr_px}×{qr_px} px")
    print(f"✓ Saved: {output.name}")
    plt.close()

    return output


def test_small_metadata():
    """Test with small metadata (4-5 fields)."""

    print("\n" + "=" * 70)
    print("2. SMALL Metadata (4-5 fields)")
    print("=" * 70)

    fig, ax = create_base_plot()

    metadata = {
        'exp': 'seizure_pred_001',
        'subj': 'S001',
        'elec': 'Fp1',
        'fs': 1000,
        'date': '2024-11-14',
    }

    qr_ver, qr_px, json_len = add_compact_footer(fig, metadata)

    output = Path(__file__).parent / "output" / "meta_variety_2_small.png"
    stx.io.save(fig, str(output), metadata=metadata)

    print(f"✓ Metadata: {len(metadata)} fields, {json_len} chars")
    print(f"✓ QR: Version {qr_ver}, {qr_px}×{qr_px} px")
    print(f"✓ Saved: {output.name}")
    plt.close()

    return output


def test_medium_metadata():
    """Test with medium metadata (6-8 fields)."""

    print("\n" + "=" * 70)
    print("3. MEDIUM Metadata (6-8 fields)")
    print("=" * 70)

    fig, ax = create_base_plot()

    metadata = {
        'exp': 'seizure_prediction_001',
        'subj': 'S001',
        'session': '2024-11-14_01',
        'elec': 'Fp1',
        'region': 'frontal',
        'fs': 1000,
        'analysis': 'PAC',
        'ver': stx.__version__,
    }

    qr_ver, qr_px, json_len = add_compact_footer(fig, metadata)

    output = Path(__file__).parent / "output" / "meta_variety_3_medium.png"
    stx.io.save(fig, str(output), metadata=metadata)

    print(f"✓ Metadata: {len(metadata)} fields, {json_len} chars")
    print(f"✓ QR: Version {qr_ver}, {qr_px}×{qr_px} px")
    print(f"✓ Saved: {output.name}")
    plt.close()

    return output


def test_large_metadata():
    """Test with large metadata (12+ fields)."""

    print("\n" + "=" * 70)
    print("4. LARGE Metadata (12+ fields)")
    print("=" * 70)

    fig, ax = create_base_plot()

    metadata = {
        'experiment': 'seizure_prediction_001',
        'subject': 'S001',
        'session': '2024-11-14_session_01',
        'electrode': 'Fp1',
        'region': 'frontal_cortex',
        'sampling_rate': 1000,
        'duration': 120,
        'analysis': 'PAC',
        'filter_low': 1,
        'filter_high': 100,
        'artifact_removal': True,
        'baseline': 'pre_ictal',
        'version': stx.__version__,
    }

    qr_ver, qr_px, json_len = add_compact_footer(fig, metadata)

    output = Path(__file__).parent / "output" / "meta_variety_4_large.png"
    stx.io.save(fig, str(output), metadata=metadata)

    print(f"✓ Metadata: {len(metadata)} fields, {json_len} chars")
    print(f"✓ QR: Version {qr_ver}, {qr_px}×{qr_px} px")
    print(f"✓ Saved: {output.name}")
    plt.close()

    return output


def test_extreme_metadata():
    """Test with extreme metadata (20+ fields)."""

    print("\n" + "=" * 70)
    print("5. EXTREME Metadata (20+ fields)")
    print("=" * 70)

    fig, ax = create_base_plot()

    metadata = {
        'experiment': 'seizure_prediction_multimodal_001',
        'subject': 'S001',
        'session': '2024-11-14_session_01',
        'electrode': 'Fp1',
        'reference': 'Cz',
        'region': 'frontal_cortex',
        'hemisphere': 'left',
        'sampling_rate': 1000,
        'duration_sec': 120,
        'analysis': 'PAC_PLV_coherence',
        'filter_low_hz': 1,
        'filter_high_hz': 100,
        'notch_filter': '50Hz',
        'artifact_removal': 'ICA',
        'baseline': 'pre_ictal_30min',
        'condition': 'awake_resting',
        'medication': 'none',
        'age': 28,
        'gender': 'M',
        'diagnosis': 'temporal_lobe_epilepsy',
        'version': stx.__version__,
        'pipeline': 'v2.1.3_standard',
    }

    qr_ver, qr_px, json_len = add_compact_footer(fig, metadata)

    output = Path(__file__).parent / "output" / "meta_variety_5_extreme.png"
    stx.io.save(fig, str(output), metadata=metadata)

    print(f"✓ Metadata: {len(metadata)} fields, {json_len} chars")
    print(f"✓ QR: Version {qr_ver}, {qr_px}×{qr_px} px")
    print(f"✓ Saved: {output.name}")
    plt.close()

    return output


def main():
    """Run all metadata variety tests."""

    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 12 + "SciTeX Metadata Variety Optimization Test" + " " * 14 + "║")
    print("╚" + "═" * 68 + "╝")

    if not QR_AVAILABLE:
        print("\n✗ qrcode library required")
        return 1

    try:
        paths = []
        paths.append(test_minimal_metadata())
        paths.append(test_small_metadata())
        paths.append(test_medium_metadata())
        paths.append(test_large_metadata())
        paths.append(test_extreme_metadata())

        print("\n" + "=" * 70)
        print("Summary - Metadata Optimization Test")
        print("=" * 70)

        print("\n✓ Generated 5 variations with different metadata sizes")
        print("\nOptimizations applied:")
        print("  • QR box_size: 2px (ultra-small)")
        print("  • QR error correction: L (minimal)")
        print("  • Dynamic column layout (2-4 cols)")
        print("  • Compact field names")
        print("  • URL: https://scitex.ai")
        print("  • Logo: tiny corner display")

        print("\nGenerated files:")
        for i, p in enumerate(paths, 1):
            print(f"  {i}. {p.name}")

        print("\n💡 Compare outputs to find optimal metadata amount!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
