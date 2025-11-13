#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Minimal QR-only metadata embedding

Shows how to use the simple QR-only approach:
- No text overlay
- No logo
- Just a small QR code
- Metadata automatically includes https://scitex.ai
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import scitex as stx


def demo_qr_only_minimal():
    """Minimal example - just QR code."""

    print("\n" + "=" * 70)
    print("Demo 1: Minimal QR Code (bottom-right)")
    print("=" * 70)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    t = np.linspace(0, 2, 1000)
    signal = np.sin(2 * np.pi * 5 * t) * np.exp(-t / 2)
    noise = np.random.normal(0, 0.08, len(t))

    ax.plot(t, signal + noise, 'b-', linewidth=0.8, alpha=0.6)
    ax.plot(t, signal, 'r--', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (μV)')
    ax.set_title('Neural Signal')
    ax.grid(True, alpha=0.3)

    # Metadata (URL will be added automatically)
    metadata = {
        'exp': 's01',
        'subj': 'S001',
        'date': '2024-11-14',
    }

    output = Path(__file__).parent / "output" / "qr_only_01_minimal.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save with QR code (no text, no logo)
    stx.io.save(fig, str(output), metadata=metadata, add_qr=True)

    print(f"✓ Saved to: {output.name}")
    print(f"✓ Metadata fields: {len(metadata)} + url")
    print(f"✓ QR position: bottom-right")
    plt.close()

    return output


def demo_qr_positions():
    """Show different QR positions."""

    print("\n" + "=" * 70)
    print("Demo 2: QR Code Positions")
    print("=" * 70)

    positions = ['bottom-right', 'bottom-left', 'top-right', 'top-left']
    outputs = []

    for pos in positions:
        fig, ax = plt.subplots(figsize=(10, 6))
        t = np.linspace(0, 2, 1000)
        signal = np.sin(2 * np.pi * 5 * t) * np.exp(-t / 2)

        ax.plot(t, signal, 'b-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'QR Position: {pos}')
        ax.grid(True, alpha=0.3)

        metadata = {'exp': 's01', 'position': pos}

        output = Path(__file__).parent / "output" / f"qr_only_02_{pos.replace('-', '_')}.png"
        stx.io.save(fig, str(output), metadata=metadata, add_qr=True, qr_position=pos)

        print(f"✓ {pos}: {output.name}")
        outputs.append(output)
        plt.close()

    return outputs


def demo_without_qr():
    """Show metadata without QR code (just embedded)."""

    print("\n" + "=" * 70)
    print("Demo 3: Metadata Only (no visual QR)")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))
    t = np.linspace(0, 2, 1000)
    signal = np.sin(2 * np.pi * 5 * t) * np.exp(-t / 2)

    ax.plot(t, signal, 'b-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Clean Figure (metadata embedded, no QR overlay)')
    ax.grid(True, alpha=0.3)

    metadata = {'exp': 's01', 'subj': 'S001'}

    output = Path(__file__).parent / "output" / "qr_only_03_no_visual_qr.png"
    stx.io.save(fig, str(output), metadata=metadata, add_qr=False)

    print(f"✓ Saved to: {output.name}")
    print(f"✓ Metadata embedded, no visual QR code")

    # Read back metadata
    meta = stx.io.read_metadata(str(output))
    print(f"✓ Metadata readable: {meta}")
    plt.close()

    return output


def main():
    """Run all demos."""

    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "SciTeX QR-Only Demo" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")

    try:
        demo_qr_only_minimal()
        demo_qr_positions()
        demo_without_qr()

        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print("\n✓ All demos completed!")
        print("\nUsage:")
        print("""
# Save with minimal QR code
stx.io.save(fig, 'output.png',
            metadata={'exp': 's01'},
            add_qr=True)

# Metadata without visual QR
stx.io.save(fig, 'output.png',
            metadata={'exp': 's01'},
            add_qr=False)

# Different QR positions
stx.io.save(fig, 'output.png',
            metadata={'exp': 's01'},
            add_qr=True,
            qr_position='top-left')
        """)

        print("\nFeatures:")
        print("  • https://scitex.ai automatically added to metadata")
        print("  • Minimal QR code (smallest possible)")
        print("  • No text overlay, no logo")
        print("  • 4 position options")
        print("  • Metadata always embedded in file")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
