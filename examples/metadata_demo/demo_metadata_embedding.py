#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Image Metadata Embedding with SciTeX

This script demonstrates how to embed and retrieve research metadata
in image files using scitex.io.

Features demonstrated:
1. Saving images with metadata (PNG and JPEG)
2. Reading metadata from images
3. Creating a visual demo with metadata displayed at the bottom
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
    print("   QR code demos will be skipped.\n")

# SciTeX logo path
LOGO_PATH = Path(__file__).parent.parent.parent / "docs" / "scitex_logos" / "vectorstock" / "vectorstock_38853699-navy-inverted-192x192.png"


def demo_basic_metadata():
    """Demonstrate basic metadata embedding and reading."""

    print("=" * 70)
    print("Demo 1: Basic Metadata Embedding")
    print("=" * 70)

    # Create a simple plot
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.exp(-x / 10)
    ax.plot(x, y, 'b-', linewidth=2, label='Neural Response')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude (μV)', fontsize=12)
    ax.set_title('Example Neural Signal', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Define metadata
    metadata = {
        'experiment': 'seizure_prediction_001',
        'session': '2024-11-14_session_01',
        'analysis': 'PAC',
        'subject_id': 'S001',
        'electrode': 'Fp1',
        'sampling_rate': 1000,
        'created': datetime.now().isoformat(),
        'notes': 'Pre-ictal period recording'
    }

    # Save with metadata (PNG)
    output_png = Path(__file__).parent / "output" / "demo_basic.png"
    output_png.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n📝 Saving image with metadata to: {output_png}")
    stx.io.save(fig, str(output_png), metadata=metadata)
    print("✓ Image saved successfully")

    # Read metadata back
    print(f"\n📖 Reading metadata from: {output_png}")
    retrieved_metadata = stx.io.read_metadata(str(output_png))

    if retrieved_metadata:
        print("✓ Metadata retrieved successfully:")
        for key, value in retrieved_metadata.items():
            print(f"  {key}: {value}")
    else:
        print("✗ No metadata found")

    # Save with metadata (JPEG)
    output_jpg = Path(__file__).parent / "output" / "demo_basic.jpg"
    print(f"\n📝 Saving image as JPEG with metadata to: {output_jpg}")
    stx.io.save(fig, str(output_jpg), metadata=metadata)
    print("✓ JPEG saved successfully")

    # Read JPEG metadata
    print(f"\n📖 Reading metadata from JPEG: {output_jpg}")
    retrieved_metadata_jpg = stx.io.read_metadata(str(output_jpg))

    if retrieved_metadata_jpg:
        print("✓ JPEG metadata retrieved successfully")
    else:
        print("✗ No metadata found in JPEG")

    plt.close()

    return output_png, output_jpg


def demo_visual_metadata_with_qr():
    """Create a demo image with metadata in QR code."""

    if not QR_AVAILABLE:
        print("\n⚠ Skipping QR code demo (qrcode library not available)")
        return None

    print("\n" + "=" * 70)
    print("Demo 2: Visual Metadata with QR Code")
    print("=" * 70)

    # Create the main plot
    fig = plt.figure(figsize=(10, 10))

    # Main plot area (top portion)
    ax_main = plt.subplot2grid((10, 10), (0, 0), rowspan=7, colspan=10)

    # Generate sample neural data
    t = np.linspace(0, 2, 1000)
    signal = np.sin(2 * np.pi * 5 * t) * np.exp(-t / 2)
    noise = np.random.normal(0, 0.1, len(t))

    ax_main.plot(t, signal + noise, 'b-', linewidth=1, alpha=0.7, label='Recorded Signal')
    ax_main.plot(t, signal, 'r--', linewidth=2, label='True Signal')
    ax_main.set_xlabel('Time (s)', fontsize=11)
    ax_main.set_ylabel('Amplitude (μV)', fontsize=11)
    ax_main.set_title('Neural Signal Analysis with QR-Encoded Metadata',
                      fontsize=13, fontweight='bold', pad=15)
    ax_main.legend(loc='upper right')
    ax_main.grid(True, alpha=0.3)

    # Metadata
    metadata = {
        'experiment': 'seizure_prediction_001',
        'session': '2024-11-14_session_01',
        'subject': 'S001',
        'electrode': 'Fp1',
        'sampling_rate': 1000,
        'analysis': 'PAC',
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'scitex_version': stx.__version__
    }

    # Separator line
    ax_sep = plt.subplot2grid((10, 10), (7, 0), rowspan=1, colspan=10)
    ax_sep.text(0.5, 0.5, '✂ ' + '─' * 30 + ' METADATA ' + '─' * 30 + ' ✂',
                ha='center', va='center', fontsize=10,
                family='monospace', color='#666')
    ax_sep.set_xlim(0, 1)
    ax_sep.set_ylim(0, 1)
    ax_sep.axis('off')

    # Generate QR code (minimal size, responsive to metadata)
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    qr = qrcode.QRCode(
        version=1,  # Start with smallest version
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=3,  # Very small box size
        border=1,
    )
    qr.add_data(metadata_json)
    qr.make(fit=True)  # Auto-adjust to minimal version needed
    qr_img = qr.make_image(fill_color="black", back_color="white")

    # Determine QR code size based on version
    qr_version = qr.version
    # Dynamically allocate columns based on QR size
    # Version 1-5: 1 column, 6-10: 1.5 columns, 11+: 2 columns
    if qr_version <= 5:
        qr_cols = 1
    elif qr_version <= 10:
        qr_cols = 1
    else:
        qr_cols = 2

    # QR code area (bottom left) - truly minimal
    ax_qr = plt.subplot2grid((10, 10), (8, 0), rowspan=2, colspan=qr_cols)
    ax_qr.imshow(qr_img, cmap='gray')
    ax_qr.axis('off')

    # Metadata display area (bottom middle) - adaptive
    meta_start_col = qr_cols
    meta_cols = 8 - qr_cols
    ax_meta = plt.subplot2grid((10, 10), (8, meta_start_col), rowspan=2, colspan=meta_cols)

    # Format metadata text (more compact, 2 columns)
    meta_lines = []
    items = list(metadata.items())
    for i in range(0, len(items), 2):
        line = f"{items[i][0]}: {items[i][1]}"
        if i + 1 < len(items):
            line += f"  |  {items[i+1][0]}: {items[i+1][1]}"
        meta_lines.append(line)

    meta_text = '\n'.join(meta_lines)

    ax_meta.text(0.05, 0.95, meta_text,
                 ha='left', va='top', fontsize=7,
                 family='monospace', color='#333',
                 transform=ax_meta.transAxes)

    ax_meta.set_xlim(0, 1)
    ax_meta.set_ylim(0, 1)
    ax_meta.axis('off')

    # Logo area (bottom right) - minimal display
    ax_logo = plt.subplot2grid((10, 10), (8, 8), rowspan=2, colspan=2)

    # Load and display logo at 16×16 pixel dimensions
    if LOGO_PATH.exists():
        logo_img = Image.open(LOGO_PATH)
        # Display at exact 16×16 pixel dimensions
        ax_logo.imshow(logo_img, extent=[0.35, 0.65, 0.35, 0.65])
        ax_logo.set_xlim(0, 1)
        ax_logo.set_ylim(0, 1)
        ax_logo.axis('off')
    else:
        # Fallback: minimal text
        ax_logo.text(0.5, 0.5, 'S',
                     ha='center', va='center', fontsize=6,
                     color='#1e3a5f', weight='bold',
                     transform=ax_logo.transAxes)
        ax_logo.set_xlim(0, 1)
        ax_logo.set_ylim(0, 1)
        ax_logo.axis('off')

    plt.tight_layout()

    # Save with metadata embedded
    output_path = Path(__file__).parent / "output" / "demo_with_qr.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n📝 Saving visual demo with QR code to: {output_path}")
    stx.io.save(fig, str(output_path), metadata=metadata)
    print("✓ Visual demo with QR code saved successfully")
    print(f"  Metadata size: {len(metadata_json)} characters")
    print(f"  QR version: {qr.version} (minimal, auto-sized)")
    print(f"  QR physical size: ~{qr.modules_count * 3}×{qr.modules_count * 3} pixels")
    print(f"  Layout columns: QR={qr_cols}, Metadata={meta_cols}, Logo=2")

    plt.close()

    return output_path


def demo_visual_metadata():
    """Create a demo image with metadata visually displayed at the bottom."""

    print("\n" + "=" * 70)
    print("Demo 3: Visual Metadata Display (No QR)")
    print("=" * 70)

    # Create the main plot
    fig = plt.figure(figsize=(10, 10))

    # Main plot area (top 70%)
    ax_main = plt.subplot2grid((10, 1), (0, 0), rowspan=7)

    # Generate sample neural data
    t = np.linspace(0, 2, 1000)
    signal = np.sin(2 * np.pi * 5 * t) * np.exp(-t / 2)
    noise = np.random.normal(0, 0.1, len(t))

    ax_main.plot(t, signal + noise, 'b-', linewidth=1, alpha=0.7, label='Recorded Signal')
    ax_main.plot(t, signal, 'r--', linewidth=2, label='True Signal')
    ax_main.set_xlabel('Time (s)', fontsize=11)
    ax_main.set_ylabel('Amplitude (μV)', fontsize=11)
    ax_main.set_title('Neural Signal Analysis with Embedded Metadata',
                      fontsize=13, fontweight='bold', pad=15)
    ax_main.legend(loc='upper right')
    ax_main.grid(True, alpha=0.3)

    # Separator line
    ax_sep = plt.subplot2grid((10, 1), (7, 0), rowspan=1)
    ax_sep.text(0.5, 0.5, '✂ ' + '─' * 30 + ' METADATA ' + '─' * 30 + ' ✂',
                ha='center', va='center', fontsize=10,
                family='monospace', color='#666')
    ax_sep.set_xlim(0, 1)
    ax_sep.set_ylim(0, 1)
    ax_sep.axis('off')

    # Metadata display area (bottom 20%)
    ax_meta = plt.subplot2grid((10, 10), (8, 0), rowspan=2, colspan=8)

    # Metadata
    metadata = {
        'experiment': 'seizure_prediction_001',
        'session': '2024-11-14_session_01',
        'subject': 'S001',
        'electrode': 'Fp1',
        'sampling_rate': 1000,
        'analysis': 'PAC',
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'scitex_version': stx.__version__
    }

    # Format metadata text
    meta_lines = []
    for i, (key, value) in enumerate(metadata.items()):
        if i % 2 == 0:
            meta_lines.append(f"{key}: {value}")
        else:
            if len(meta_lines) > 0:
                meta_lines[-1] += f"  |  {key}: {value}"

    meta_text = '\n'.join(meta_lines)

    ax_meta.text(0.05, 0.95, meta_text,
                 ha='left', va='top', fontsize=8,
                 family='monospace', color='#333',
                 transform=ax_meta.transAxes)

    ax_meta.set_xlim(0, 1)
    ax_meta.set_ylim(0, 1)
    ax_meta.axis('off')

    # Logo area (bottom right)
    ax_logo = plt.subplot2grid((10, 10), (8, 8), rowspan=2, colspan=2)

    # Load and display logo (minimal 16x16px)
    if LOGO_PATH.exists():
        logo_img = Image.open(LOGO_PATH)
        # Resize to minimal size: 16×16 pixels
        logo_img = logo_img.resize((16, 16), Image.Resampling.LANCZOS)
        ax_logo.imshow(logo_img)
        ax_logo.axis('off')
    else:
        # Fallback: minimal text
        ax_logo.text(0.5, 0.5, 'S',
                     ha='center', va='center', fontsize=6,
                     color='#1e3a5f', weight='bold',
                     transform=ax_logo.transAxes)
        ax_logo.set_xlim(0, 1)
        ax_logo.set_ylim(0, 1)
        ax_logo.axis('off')

    plt.tight_layout()

    # Save with metadata embedded
    output_path = Path(__file__).parent / "output" / "demo_visual.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n📝 Saving visual demo with embedded metadata to: {output_path}")
    stx.io.save(fig, str(output_path), metadata=metadata)
    print("✓ Visual demo saved successfully")

    plt.close()

    return output_path


def demo_without_metadata():
    """Show that images can still be saved without metadata (backward compatible)."""

    print("\n" + "=" * 70)
    print("Demo 4: Backward Compatibility (No Metadata)")
    print("=" * 70)

    # Create a simple plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], 'go-')
    ax.set_title('Plot without metadata')

    # Save without metadata (existing behavior)
    output_path = Path(__file__).parent / "output" / "demo_no_metadata.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n📝 Saving image WITHOUT metadata to: {output_path}")
    stx.io.save(fig, str(output_path))
    print("✓ Image saved successfully (no metadata)")

    # Try to read metadata (should return None)
    print(f"\n📖 Reading metadata from: {output_path}")
    retrieved_metadata = stx.io.read_metadata(str(output_path))

    if retrieved_metadata is None:
        print("✓ Correctly returned None (no metadata embedded)")
    else:
        print(f"⚠ Unexpected metadata found: {retrieved_metadata}")

    plt.close()

    return output_path


def main():
    """Run all demos."""

    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "SciTeX Metadata Embedding Demo" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Run demos
    try:
        png_path, jpg_path = demo_basic_metadata()
        qr_path = demo_visual_metadata_with_qr()
        visual_path = demo_visual_metadata()
        no_meta_path = demo_without_metadata()

        # Summary
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print("\n✓ All demos completed successfully!")
        print(f"\nGenerated files:")
        print(f"  1. Basic PNG with metadata: {png_path}")
        print(f"  2. Basic JPEG with metadata: {jpg_path}")
        if qr_path:
            print(f"  3. Visual demo with QR code: {qr_path}")
        print(f"  4. Visual demo: {visual_path}")
        print(f"  5. No metadata (backward compatible): {no_meta_path}")

        print("\n" + "=" * 70)
        print("Usage in your code:")
        print("=" * 70)
        print("""
# Save image with metadata
metadata = {'experiment': 'test_001', 'date': '2024-11-14'}
stx.io.save(fig, 'result.png', metadata=metadata)

# Read metadata
meta = stx.io.read_metadata('result.png')
print(meta['experiment'])  # 'test_001'

# Check if metadata exists
if stx.io.has_metadata('result.png'):
    print("Image has metadata!")
        """)

    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
