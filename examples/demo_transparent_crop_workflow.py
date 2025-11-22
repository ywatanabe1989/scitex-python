#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-11-19 14:30:00 (ywatanabe)"
# File: ./examples/demo_transparent_crop_workflow.py

"""
Demonstration of the publication-ready figure workflow with transparent backgrounds.

This demo showcases the professional workflow for creating publication figures:
1. Create figure with large margins and transparent background
2. Save with high DPI
3. Auto-crop to remove excess whitespace
4. Result: Perfect publication-ready figure

This workflow is used by professional scientific illustrators for
Nature, Cell, and other high-impact journals.
"""

import numpy as np
import scitex as stx


def demo_transparent_workflow():
    """
    Demonstrate the transparent background + crop workflow.

    Workflow:
    1. Create figure with generous margins (prevents content cutoff)
    2. Use transparent background (works in any layout)
    3. Save at publication DPI (300)
    4. Auto-crop to remove excess whitespace
    """
    print("\n" + "="*70)
    print("Demo: Transparent Background + Auto-Crop Workflow")
    print("="*70)

    # Step 1: Create figure with large margins and transparent background
    print("\n[1] Creating figure with large margins and transparent background...")
    fig, ax = stx.plt.subplots(
        ax_width_mm=40,
        ax_height_mm=28,
        margin_left_mm=20,      # Generous margins
        margin_right_mm=20,
        margin_top_mm=20,
        margin_bottom_mm=20,
        ax_thickness_mm=0.2,
        tick_length_mm=0.8,
        mode='publication',
        dpi=300,
        transparent=True,       # ← Transparent background!
    )

    # Plot some data
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    ax.plot(x, y, 'b-', linewidth=1.5, label='sin(x)', id="sine")
    ax.set_xlabel('x (radians)')
    ax.set_ylabel('y')
    ax.set_title('Sine Wave with Transparent Background')
    ax.legend(frameon=False)

    # Ensure spines are visible on transparent background
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)

    # Step 2: Save with high DPI
    print("[2] Saving with 300 DPI and transparent=True...")
    original_path = "transparent_demo_original.png"
    stx.io.save(fig, original_path, dpi=300, transparent=True)

    # Get the actual file path from session output directory
    import os
    script_name = os.path.basename(__file__).replace('.py', '_out')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    session_out = os.path.join(script_dir, script_name)
    actual_original_path = os.path.join(session_out, original_path)

    fig.close()

    print(f"    ✓ Saved: {actual_original_path}")
    print(f"    - Large margins (20mm on all sides)")
    print(f"    - Transparent background")
    print(f"    - 300 DPI for publication quality")

    # Step 3: Auto-crop to remove whitespace
    print("\n[3] Auto-cropping to remove excess whitespace...")
    cropped_path = "transparent_demo_cropped.png"
    actual_cropped_path = os.path.join(session_out, cropped_path)

    import subprocess
    result = subprocess.run(
        [
            "python", "scripts/python/crop_tif.py",
            "-i", actual_original_path,
            "-o", actual_cropped_path,
            "-m", "2",  # 2px margin
            "-v"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"    ✓ Cropped: {cropped_path}")
        print(f"    - Whitespace removed")
        print(f"    - Content perfectly framed")
        print(f"    - Transparency preserved")
    else:
        print(f"    ✗ Cropping failed: {result.stderr}")

    print("\n" + "="*70)
    print("Workflow Complete!")
    print("="*70)
    print("\nBenefits of this approach:")
    print("  ✓ Never worry about content being cut off")
    print("  ✓ Works in any document/presentation layout")
    print("  ✓ Professional publication quality")
    print("  ✓ Easy to reproduce and automate")
    print("\nOutput files:")
    print(f"  - Original: {original_path} (with large margins)")
    print(f"  - Cropped:  {cropped_path} (publication-ready)")


@stx.session
def main(verbose=True):
    """Run transparent background + crop workflow demo."""

    demo_transparent_workflow()

    return 0


if __name__ == "__main__":
    main()

# EOF
