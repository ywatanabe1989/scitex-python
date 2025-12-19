#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/examples/fig/07_roundtrip_edit.py

"""
Example 07: Round-Trip Edit

Demonstrates:
- Canonical vs cache separation
- Edit theme.json without touching data/spec
- Delete cache/exports, re-render
- Deterministic regeneration
"""

import hashlib
import json
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
    """Demonstrate round-trip editing with canonical/cache separation."""
    logger.info("Example 07: Round-Trip Edit Demo")

    out_dir = CONFIG["SDIR_OUT"]
    bundle_path = out_dir / "roundtrip_demo.zip.d"

    # === Step 1: Create initial bundle ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Create initial bundle")
    logger.info("=" * 60)

    fig = Figz(
        bundle_path, name="Round-Trip Demo", size_mm={"width": 120, "height": 80}
    )

    x = np.linspace(0, 10, 50)
    fig_a, ax = plt.subplots(figsize=(4, 3))
    ax.plot(x, np.sin(x), linewidth=1.5, label="sin(x)")
    ax.plot(x, np.cos(x), linewidth=1.5, label="cos(x)")
    ax.legend()
    ax.set_title("Original")

    fig.add_element(
        "plot_A",
        "plot",
        fig_a,
        position={"x_mm": 10, "y_mm": 5},
        size={"width_mm": 100, "height_mm": 70},
    )
    plt.close(fig_a)
    fig.set_panel_info(
        "plot_A", panel_letter="A", description="Trigonometric functions"
    )
    fig.save()

    # Record initial hashes
    initial_hashes = {
        "spec.json": file_hash(bundle_path / "spec.json"),
        "encoding.json": file_hash(bundle_path / "encoding.json"),
        "theme.json": file_hash(bundle_path / "theme.json"),
        "exports/figure.png": file_hash(bundle_path / "exports" / "figure.png"),
    }

    logger.info("\nInitial file hashes:")
    for name, h in initial_hashes.items():
        logger.info(f"  {name}: {h}")

    # === Step 2: Edit theme.json only ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Edit theme.json (change font size)")
    logger.info("=" * 60)

    theme_path = bundle_path / "theme.json"
    with open(theme_path) as f:
        theme = json.load(f)

    # Modify theme
    original_font_size = theme.get("typography", {}).get("size_pt", 7.0)
    theme["typography"]["size_pt"] = 10.0  # Change font size
    theme["colors"]["mode"] = "dark"  # Change color mode

    with open(theme_path, "w") as f:
        json.dump(theme, f, indent=2)

    logger.info(f"  Changed typography.size_pt: {original_font_size} -> 10.0")
    logger.info("  Changed colors.mode: light -> dark")

    # === Step 3: Delete cache and exports ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Delete cache/* and exports/*")
    logger.info("=" * 60)

    cache_dir = bundle_path / "cache"
    exports_dir = bundle_path / "exports"

    deleted_cache = list(cache_dir.glob("*")) if cache_dir.exists() else []
    deleted_exports = list(exports_dir.glob("*")) if exports_dir.exists() else []

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    if exports_dir.exists():
        shutil.rmtree(exports_dir)

    logger.info(f"  Deleted {len(deleted_cache)} cache files")
    logger.info(f"  Deleted {len(deleted_exports)} export files")

    # === Step 4: Re-render ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Re-render (reload and save)")
    logger.info("=" * 60)

    fig2 = Figz(bundle_path)
    fig2.save()

    # Record final hashes
    final_hashes = {
        "spec.json": file_hash(bundle_path / "spec.json"),
        "encoding.json": file_hash(bundle_path / "encoding.json"),
        "theme.json": file_hash(bundle_path / "theme.json"),
        "exports/figure.png": file_hash(bundle_path / "exports" / "figure.png"),
    }

    # === Step 5: Report changes ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Change Report")
    logger.info("=" * 60)

    logger.info(f"\n{'File':<25} {'Initial':<14} {'Final':<14} {'Status':<10}")
    logger.info("-" * 65)

    for name in initial_hashes:
        initial = initial_hashes[name]
        final = final_hashes[name]
        if initial == final:
            status = "UNCHANGED"
        elif initial is None:
            status = "NEW"
        elif final is None:
            status = "DELETED"
        else:
            status = "CHANGED"
        logger.info(
            f"{name:<25} {initial or 'N/A':<14} {final or 'N/A':<14} {status:<10}"
        )

    # Verify cache regenerated
    logger.info("\n" + "-" * 40)
    logger.info("Cache regeneration check:")
    cache_files = list(cache_dir.glob("*")) if cache_dir.exists() else []
    export_files = list(exports_dir.glob("*")) if exports_dir.exists() else []
    logger.info(f"  cache/* files: {len(cache_files)}")
    logger.info(f"  exports/* files: {len(export_files)}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    logger.info("  - spec.json: UNCHANGED (canonical structure)")
    logger.info("  - encoding.json: UNCHANGED (data bindings)")
    logger.info("  - theme.json: CHANGED (edited aesthetics)")
    logger.info("  - exports/*: REGENERATED (derived from theme)")
    logger.info("  - cache/*: REGENERATED (derived)")
    logger.info("=" * 60)

    logger.success("Example 07 completed!")


if __name__ == "__main__":
    main()

# EOF
