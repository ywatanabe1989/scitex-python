#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/examples/fig/01_basic_figure.py

"""
Example 01: Basic Figure Bundle Creation

Demonstrates:
- Creating a new .stx figure bundle
- Setting figure title and size
- Saving as ZIP (.stx) or directory (.stx.d)
"""

import scitex as stx
from scitex import INJECTED
from scitex.fig import Figz


@stx.session(verbose=True, agg=True)
def main(CONFIG=INJECTED, logger=INJECTED):
    """Create a basic figure bundle."""
    logger.info("Example 01: Basic Figure Bundle Creation")

    out_dir = CONFIG["SDIR_OUT"]

    # Create a new figure bundle
    fig = Figz(
        out_dir / "my_figure.stx",
        name="My First Figure",
        size_mm={"width": 170, "height": 120},
    )

    logger.info(f"Created: {fig}")
    logger.info(f"Bundle ID: {fig.bundle_id}")
    logger.info(f"Size: {fig.size_mm}")
    logger.info(f"Type: {fig.bundle_type}")

    # Save as ZIP archive
    fig.save()
    logger.info(f"Saved as ZIP: {fig.path}")

    # Save as directory bundle
    dir_path = out_dir / "my_figure.stx.d"
    fig.save(dir_path)
    logger.info(f"Saved as directory: {dir_path}")

    # Reload and verify
    reloaded = Figz(out_dir / "my_figure.stx")
    logger.info(f"Reloaded: {reloaded}")
    logger.info(f"Elements: {reloaded.elements}")

    logger.success("Example 01 completed!")


if __name__ == "__main__":
    main()

# EOF
