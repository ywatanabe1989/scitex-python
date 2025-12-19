#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/examples/fig/14_zip_portability.py

"""
Example 14: Single-File Portability (ZIP load/save)

Demonstrates:
- .zip.d directory bundle ↔ .zip archive
- Portability and robustness of .zip as archive
- Round-trip: dir → zip → delete dir → load zip → export
"""

import hashlib
import shutil

import numpy as np

import scitex as stx
from scitex import INJECTED
from scitex.fig import Figz


def file_hash(path):
    """Compute MD5 hash of a file."""
    if not path.exists():
        return None
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


@stx.session(verbose=True, agg=True)
def main(plt=INJECTED, CONFIG=INJECTED, logger=INJECTED):
    """Demonstrate ZIP portability."""
    logger.info("Example 14: ZIP Portability Demo")

    out_dir = CONFIG["SDIR_OUT"]
    dir_bundle = out_dir / "portable_figure.zip.d"
    zip_bundle = out_dir / "portable_figure.zip"

    # Clean up any existing files
    if dir_bundle.exists():
        shutil.rmtree(dir_bundle)
    if zip_bundle.exists():
        zip_bundle.unlink()

    # === Step 1: Create directory bundle ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Create .zip.d directory bundle")
    logger.info("=" * 60)

    fig = Figz(dir_bundle, name="Portable Figure", size_mm={"width": 100, "height": 70})

    x = np.linspace(0, 10, 50)
    fig_a, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(x, np.sin(x), "b-", linewidth=1.5, label="sin(x)")
    ax.plot(x, np.cos(x), "r--", linewidth=1.5, label="cos(x)")
    ax.legend()
    ax.set_title("Portable Plot")

    fig.add_element(
        "plot_A",
        "plot",
        fig_a,
        position={"x_mm": 5, "y_mm": 5},
        size={"width_mm": 90, "height_mm": 60},
    )
    plt.close(fig_a)
    fig.set_panel_info(
        "plot_A", panel_letter="A", description="Trigonometric functions"
    )
    fig.save()

    dir_svg_hash = file_hash(dir_bundle / "exports" / "figure.svg")
    dir_png_hash = file_hash(dir_bundle / "exports" / "figure.png")

    logger.info(f"  Created: {dir_bundle}")
    logger.info(f"  SVG hash: {dir_svg_hash}")
    logger.info(f"  PNG hash: {dir_png_hash}")

    # === Step 2: Pack to ZIP ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Pack to .zip archive")
    logger.info("=" * 60)

    fig_packed = fig.pack(zip_bundle)
    zip_size = zip_bundle.stat().st_size

    logger.info(f"  Packed to: {zip_bundle}")
    logger.info(f"  ZIP size: {zip_size / 1024:.1f} KB")

    # === Step 3: Delete directory bundle ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Delete directory bundle")
    logger.info("=" * 60)

    shutil.rmtree(dir_bundle)
    logger.info(f"  Deleted: {dir_bundle}")
    logger.info(f"  Only ZIP remains: {zip_bundle.exists()}")

    # === Step 4: Load from ZIP ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Load from .zip")
    logger.info("=" * 60)

    fig_from_zip = Figz(zip_bundle)
    logger.info(f"  Loaded: {fig_from_zip}")
    logger.info(f"  Elements: {fig_from_zip.list_element_ids()}")

    # === Step 5: Unpack to directory ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Unpack back to directory")
    logger.info("=" * 60)

    fig_unpacked = fig_from_zip.unpack(dir_bundle)
    fig_unpacked.save()  # Regenerate exports

    unpacked_svg_hash = file_hash(dir_bundle / "exports" / "figure.svg")
    unpacked_png_hash = file_hash(dir_bundle / "exports" / "figure.png")

    logger.info(f"  Unpacked to: {dir_bundle}")
    logger.info(f"  SVG hash: {unpacked_svg_hash}")
    logger.info(f"  PNG hash: {unpacked_png_hash}")

    # === Verification ===
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION")
    logger.info("=" * 60)

    logger.info(f"\n{'Stage':<20} {'SVG Hash':<14} {'PNG Hash':<14}")
    logger.info("-" * 50)
    logger.info(f"{'Original':<20} {dir_svg_hash:<14} {dir_png_hash:<14}")
    logger.info(
        f"{'After round-trip':<20} {unpacked_svg_hash:<14} {unpacked_png_hash:<14}"
    )

    # Note: Hashes may differ due to metadata/timestamps in exports
    # The important thing is that the figure renders correctly

    logger.info("\n" + "-" * 40)
    logger.info("Round-trip workflow:")
    logger.info("  1. Create .zip.d (directory bundle)")
    logger.info("  2. Pack to .zip (ZIP archive)")
    logger.info("  3. Delete directory, keep only ZIP")
    logger.info("  4. Load from ZIP")
    logger.info("  5. Unpack to directory if needed")

    logger.info("\n" + "=" * 60)
    logger.info("Key takeaway:")
    logger.info("  .zip is portable - share a single file")
    logger.info("  .zip.d (directory) is editable - inspect/modify files")
    logger.info("  Both are fully interchangeable")
    logger.info("=" * 60)

    logger.success("Example 14 completed!")


if __name__ == "__main__":
    main()

# EOF
