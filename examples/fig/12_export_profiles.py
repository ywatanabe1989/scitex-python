#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/examples/fig/12_export_profiles.py

"""
Example 12: Export Profiles (Journal Presets)

Demonstrates:
- Same canonical plot, multiple export styles
- Journal-specific presets (Nature, Cell, IEEE)
- Only theme/render differs, data unchanged
"""

import hashlib
import json

import numpy as np

import scitex as stx
from scitex import INJECTED
from scitex.fig import Figz

# Journal export profiles
PROFILES = {
    "nature": {
        "name": "Nature",
        "typography": {"family": "Arial", "size_pt": 7.0},
        "linewidth": 0.5,
        "dpi": 300,
        "colors": {"mode": "light"},
    },
    "cell": {
        "name": "Cell",
        "typography": {"family": "Helvetica", "size_pt": 8.0},
        "linewidth": 1.0,
        "dpi": 300,
        "colors": {"mode": "light"},
    },
    "ieee": {
        "name": "IEEE",
        "typography": {"family": "Times New Roman", "size_pt": 8.0},
        "linewidth": 0.75,
        "dpi": 600,
        "colors": {"mode": "light"},
    },
}


def file_hash(path):
    """Compute MD5 hash of a file."""
    if not path.exists():
        return None
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


@stx.session(verbose=True, agg=True)
def main(plt=INJECTED, CONFIG=INJECTED, logger=INJECTED):
    """Demonstrate export profiles for different journals."""
    logger.info("Example 12: Export Profiles Demo")

    out_dir = CONFIG["SDIR_OUT"]
    bundle_path = out_dir / "export_profiles.stx.d"

    # === Create base figure ===
    logger.info("\n" + "=" * 60)
    logger.info("Creating base figure...")
    logger.info("=" * 60)

    fig = Figz(
        bundle_path, name="Export Profiles Demo", size_mm={"width": 85, "height": 60}
    )

    x = np.linspace(0, 10, 100)
    np.random.seed(42)

    fig_a, ax = plt.subplots(figsize=(3.35, 2.36))  # ~85mm x 60mm at 300dpi
    ax.plot(x, np.sin(x), linewidth=1.0, label="Signal")
    ax.fill_between(x, np.sin(x) - 0.2, np.sin(x) + 0.2, alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title("Neural Response")
    ax.legend(loc="upper right")

    fig.add_element(
        "plot_A",
        "plot",
        fig_a,
        position={"x_mm": 2, "y_mm": 2},
        size={"width_mm": 81, "height_mm": 56},
    )
    plt.close(fig_a)
    fig.set_panel_info("plot_A", panel_letter="A", description="Neural response signal")
    fig.save()

    # Record base data hash
    base_data_hash = file_hash(bundle_path / "data" / "data_info.json")
    base_encoding_hash = file_hash(bundle_path / "encoding.json")

    logger.info(f"Base data hash: {base_data_hash}")
    logger.info(f"Base encoding hash: {base_encoding_hash}")

    # === Export with different profiles ===
    logger.info("\n" + "=" * 60)
    logger.info("Exporting with different journal profiles...")
    logger.info("=" * 60)

    results = {}

    for profile_id, profile in PROFILES.items():
        logger.info(f"\n--- Profile: {profile['name']} ---")

        # Create profile-specific export directory
        profile_exports = bundle_path / "exports" / f"profile_{profile_id}"
        profile_exports.mkdir(parents=True, exist_ok=True)

        # Modify theme.json temporarily
        theme_path = bundle_path / "theme.json"
        with open(theme_path) as f:
            theme = json.load(f)

        # Apply profile settings
        theme["typography"] = profile["typography"]
        theme["colors"] = profile["colors"]

        with open(theme_path, "w") as f:
            json.dump(theme, f, indent=2)

        # Re-render with profile
        fig2 = Figz(bundle_path)

        # Generate exports at profile DPI
        for fmt in ["png", "svg", "pdf"]:
            export_bytes = fig2.render_preview_format(fmt, dpi=profile["dpi"])
            export_path = profile_exports / f"figure.{fmt}"
            with open(export_path, "wb") as f:
                f.write(export_bytes)

        # Record results
        results[profile_id] = {
            "name": profile["name"],
            "dpi": profile["dpi"],
            "font": profile["typography"]["family"],
            "font_size": profile["typography"]["size_pt"],
            "png_hash": file_hash(profile_exports / "figure.png"),
            "data_hash": file_hash(bundle_path / "data" / "data_info.json"),
        }

        logger.info(
            f"  Font: {profile['typography']['family']} @ {profile['typography']['size_pt']}pt"
        )
        logger.info(f"  DPI: {profile['dpi']}")
        logger.info(f"  Exports: {profile_exports}")

    # === Report ===
    logger.info("\n" + "=" * 60)
    logger.info("EXPORT PROFILES COMPARISON")
    logger.info("=" * 60)

    logger.info(
        f"\n{'Profile':<10} {'Font':<20} {'Size':<6} {'DPI':<6} {'PNG Hash':<14}"
    )
    logger.info("-" * 60)

    for profile_id, result in results.items():
        logger.info(
            f"{result['name']:<10} {result['font']:<20} {result['font_size']:<6} "
            f"{result['dpi']:<6} {result['png_hash']:<14}"
        )

    # Verify data unchanged
    logger.info("\n" + "-" * 40)
    logger.info("Data integrity check:")
    all_same = all(r["data_hash"] == base_data_hash for r in results.values())
    logger.info(f"  Base data hash: {base_data_hash}")
    logger.info(f"  All profiles have same data: {all_same}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Key takeaway:")
    logger.info("  - Same data, same encoding across all profiles")
    logger.info("  - Only theme.json and render settings differ")
    logger.info("  - Each journal gets optimized exports")
    logger.info("=" * 60)

    logger.success("Example 12 completed!")


if __name__ == "__main__":
    main()

# EOF
