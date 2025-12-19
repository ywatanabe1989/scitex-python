#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/examples/fig/09_data_swap_same_encoding.py

"""
Example 09: Data Swap with Same Encoding

Demonstrates:
- Encoding is reusable across datasets
- Data is portable (same column structure)
- Schema supports reproducible re-plotting
"""

import hashlib
import json

import numpy as np
import pandas as pd

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
    """Demonstrate data swapping with constant encoding."""
    logger.info("Example 09: Data Swap Same Encoding Demo")

    out_dir = CONFIG["SDIR_OUT"]
    bundle_path = out_dir / "data_swap_demo.zip.d"

    # === Dataset 1: Low amplitude signal ===
    logger.info("\n" + "=" * 60)
    logger.info("DATASET 1: Low amplitude signal")
    logger.info("=" * 60)

    np.random.seed(42)
    x1 = np.linspace(0, 10, 50)
    y1 = np.sin(x1) + np.random.normal(0, 0.1, 50)

    df1 = pd.DataFrame({"x": x1, "y": y1, "label": "dataset_1"})
    logger.info(f"  y range: [{y1.min():.2f}, {y1.max():.2f}]")

    # Create figure with dataset 1
    fig = Figz(bundle_path, name="Data Swap Demo", size_mm={"width": 120, "height": 80})

    fig_a, ax = plt.subplots(figsize=(4, 3))
    ax.plot(x1, y1, "o-", markersize=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Dataset 1")

    fig.add_element(
        "plot_A",
        "plot",
        fig_a,
        position={"x_mm": 10, "y_mm": 5},
        size={"width_mm": 100, "height_mm": 70},
    )
    plt.close(fig_a)
    fig.save()

    # Save data as CSV
    data_dir = bundle_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df1.to_csv(data_dir / "data.csv", index=False)

    # Create encoding.json (defines x→x_axis, y→y_axis mapping)
    encoding = {
        "schema": {"name": "scitex.plt.encoding", "version": "1.0.0"},
        "traces": [
            {
                "trace_id": "plot_A",
                "bindings": [
                    {"channel": "x", "column": "x", "scale": "linear"},
                    {"channel": "y", "column": "y", "scale": "linear"},
                ],
            }
        ],
    }
    with open(bundle_path / "encoding.json", "w") as f:
        json.dump(encoding, f, indent=2)

    encoding_hash_1 = file_hash(bundle_path / "encoding.json")
    data_hash_1 = file_hash(data_dir / "data.csv")
    export_hash_1 = file_hash(bundle_path / "exports" / "figure.png")

    logger.info(f"  encoding.json hash: {encoding_hash_1}")
    logger.info(f"  data.csv hash: {data_hash_1}")
    logger.info(f"  figure.png hash: {export_hash_1}")

    # === Dataset 2: High amplitude signal (same columns!) ===
    logger.info("\n" + "=" * 60)
    logger.info("DATASET 2: High amplitude signal (SWAP)")
    logger.info("=" * 60)

    x2 = np.linspace(0, 10, 50)
    y2 = 3 * np.sin(x2) + np.random.normal(0, 0.3, 50)  # 3x amplitude!

    df2 = pd.DataFrame({"x": x2, "y": y2, "label": "dataset_2"})
    logger.info(f"  y range: [{y2.min():.2f}, {y2.max():.2f}]")

    # Swap data (same columns, different values)
    df2.to_csv(data_dir / "data.csv", index=False)

    # Re-create plot with new data
    fig2 = Figz(bundle_path)  # Reload

    fig_b, ax2 = plt.subplots(figsize=(4, 3))
    ax2.plot(x2, y2, "o-", markersize=3, color="red")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Dataset 2 (swapped)")

    # Remove old element and add updated one
    fig2.remove_element("plot_A")
    fig2.add_element(
        "plot_A",
        "plot",
        fig_b,
        position={"x_mm": 10, "y_mm": 5},
        size={"width_mm": 100, "height_mm": 70},
    )
    plt.close(fig_b)
    fig2.save()

    encoding_hash_2 = file_hash(bundle_path / "encoding.json")
    data_hash_2 = file_hash(data_dir / "data.csv")
    export_hash_2 = file_hash(bundle_path / "exports" / "figure.png")

    logger.info(f"  encoding.json hash: {encoding_hash_2}")
    logger.info(f"  data.csv hash: {data_hash_2}")
    logger.info(f"  figure.png hash: {export_hash_2}")

    # === Report ===
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON REPORT")
    logger.info("=" * 60)

    logger.info(f"\n{'File':<20} {'Dataset 1':<14} {'Dataset 2':<14} {'Status':<12}")
    logger.info("-" * 60)

    # encoding.json should be UNCHANGED
    enc_status = "UNCHANGED" if encoding_hash_1 == encoding_hash_2 else "CHANGED"
    logger.info(
        f"{'encoding.json':<20} {encoding_hash_1:<14} {encoding_hash_2:<14} {enc_status:<12}"
    )

    # data.csv should be CHANGED
    data_status = "UNCHANGED" if data_hash_1 == data_hash_2 else "CHANGED"
    logger.info(
        f"{'data/data.csv':<20} {data_hash_1:<14} {data_hash_2:<14} {data_status:<12}"
    )

    # figure.png should be CHANGED
    export_status = "UNCHANGED" if export_hash_1 == export_hash_2 else "CHANGED"
    logger.info(
        f"{'exports/figure.png':<20} {export_hash_1:<14} {export_hash_2:<14} {export_status:<12}"
    )

    # Y-range sanity check
    logger.info("\n" + "-" * 40)
    logger.info("Y-range sanity check:")
    logger.info(f"  Dataset 1: [{y1.min():.2f}, {y1.max():.2f}]")
    logger.info(f"  Dataset 2: [{y2.min():.2f}, {y2.max():.2f}]")
    logger.info(f"  Range increased: {(y2.max() - y2.min()) > (y1.max() - y1.min())}")

    logger.info("\n" + "=" * 60)
    logger.info("Key takeaway:")
    logger.info("  encoding.json defines HOW to plot (x→x, y→y)")
    logger.info("  data.csv contains WHAT to plot (values)")
    logger.info("  Same encoding + different data = updated plot")
    logger.info("=" * 60)

    logger.success("Example 09 completed!")


if __name__ == "__main__":
    main()

# EOF
