#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/examples/fig/15_style_bridge.py

"""
Example 15: Backward Compatibility (style.json bridge)

Demonstrates:
- Legacy style.json is still supported
- Loader splits into encoding.json + theme.json internally
- Seamless migration path for existing bundles
"""

import json
import shutil

import numpy as np

import scitex as stx
from scitex import INJECTED
from scitex.fig import Figz


@stx.session(verbose=True, agg=True)
def main(plt=INJECTED, CONFIG=INJECTED, logger=INJECTED):
    """Demonstrate backward compatibility with style.json."""
    logger.info("Example 15: Style.json Bridge Demo")

    out_dir = CONFIG["SDIR_OUT"]
    legacy_bundle = out_dir / "legacy_bundle.stx.d"

    # Clean up
    if legacy_bundle.exists():
        shutil.rmtree(legacy_bundle)

    # === Create a "legacy" bundle with only style.json ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Create legacy bundle (style.json only)")
    logger.info("=" * 60)

    # Create directory structure manually (simulating old format)
    legacy_bundle.mkdir(parents=True)

    # Write legacy spec.json
    spec = {
        "schema": {"name": "scitex.bundle", "version": "1.0.0"},  # Old version
        "type": "figure",
        "bundle_id": "legacy-demo-12345",
        "title": "Legacy Figure",
        "size_mm": {"width": 100, "height": 70},
        "elements": [],
    }
    with open(legacy_bundle / "spec.json", "w") as f:
        json.dump(spec, f, indent=2)

    # Write legacy style.json (combined encoding + theme)
    style = {
        "theme": {"mode": "light", "font_family": "Arial"},
        "traces": [
            {
                "trace_id": "line0",
                "color": "#1f77b4",
                "linewidth": 1.5,
                "x_col": "time",
                "y_col": "value",
            }
        ],
        "grid": True,
        "font": {"family": "Arial", "size": 10},
    }
    with open(legacy_bundle / "style.json", "w") as f:
        json.dump(style, f, indent=2)

    logger.info(f"  Created: {legacy_bundle}")
    logger.info("  Files: spec.json, style.json (legacy format)")
    logger.info(f"  style.json content: {json.dumps(style, indent=2)[:100]}...")

    # Check no encoding.json or theme.json yet
    has_encoding = (legacy_bundle / "encoding.json").exists()
    has_theme = (legacy_bundle / "theme.json").exists()
    logger.info(f"\n  encoding.json exists: {has_encoding}")
    logger.info(f"  theme.json exists: {has_theme}")

    # === Load legacy bundle (triggers migration) ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Load legacy bundle (triggers internal split)")
    logger.info("=" * 60)

    fig = Figz(legacy_bundle)
    logger.info(f"  Loaded: {fig}")

    # Add a plot to make it functional
    x = np.linspace(0, 10, 50)
    fig_a, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(x, np.sin(x), linewidth=1.5)
    ax.set_title("Migrated Plot")

    fig.add_element(
        "plot_A",
        "plot",
        fig_a,
        position={"x_mm": 5, "y_mm": 5},
        size={"width_mm": 90, "height_mm": 60},
    )
    plt.close(fig_a)

    # === Save (creates new format files) ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Save (generates new format)")
    logger.info("=" * 60)

    fig.save()

    # Check what files exist now
    files_after = list(legacy_bundle.glob("*.json"))
    logger.info(f"  JSON files after save: {[f.name for f in files_after]}")

    has_encoding = (legacy_bundle / "encoding.json").exists()
    has_theme = (legacy_bundle / "theme.json").exists()
    has_style = (legacy_bundle / "style.json").exists()

    logger.info(f"\n  encoding.json exists: {has_encoding}")
    logger.info(f"  theme.json exists: {has_theme}")
    logger.info(f"  style.json exists: {has_style}")

    # Show encoding.json content
    if has_encoding:
        with open(legacy_bundle / "encoding.json") as f:
            encoding = json.load(f)
        logger.info(f"\n  encoding.json:\n{json.dumps(encoding, indent=2)}")

    # Show theme.json content (summarized)
    if has_theme:
        with open(legacy_bundle / "theme.json") as f:
            theme = json.load(f)
        logger.info(
            f"\n  theme.json colors.mode: {theme.get('colors', {}).get('mode')}"
        )
        logger.info(f"  theme.json typography: {theme.get('typography')}")

    # === Verify exports work ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Verify exports")
    logger.info("=" * 60)

    exports_dir = legacy_bundle / "exports"
    if exports_dir.exists():
        exports = list(exports_dir.glob("*"))
        logger.info(f"  Exports: {[e.name for e in exports]}")
    else:
        logger.info("  No exports directory")

    # === Summary ===
    logger.info("\n" + "=" * 60)
    logger.info("MIGRATION SUMMARY")
    logger.info("=" * 60)

    logger.info("\nBefore (legacy):")
    logger.info("  - spec.json (v1.0.0)")
    logger.info("  - style.json (combined encoding + theme)")

    logger.info("\nAfter (new format):")
    logger.info("  - spec.json (v2.0.0)")
    logger.info("  - encoding.json (data bindings)")
    logger.info("  - theme.json (aesthetics)")
    logger.info("  - style.json (kept for compatibility)")

    logger.info("\n" + "-" * 40)
    logger.info("Key takeaway:")
    logger.info("  Old bundles with style.json still work")
    logger.info("  Saving adds encoding.json + theme.json")
    logger.info("  Gradual migration, no breaking changes")
    logger.info("=" * 60)

    logger.success("Example 15 completed!")


if __name__ == "__main__":
    main()

# EOF
