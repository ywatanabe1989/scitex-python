#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/examples/fig/03_text_and_shapes.py

"""
Example 03: Text and Shape Elements

Demonstrates:
- Adding text elements (titles, labels)
- Adding shape elements (arrows, brackets, lines)
- Adding symbols and equations
"""

import scitex as stx
from scitex import INJECTED
from scitex.fig import Figz


@stx.session(verbose=True, agg=True)
def main(CONFIG=INJECTED, logger=INJECTED):
    """Add text, shapes, symbols, and equations."""
    logger.info("Example 03: Text and Shape Elements")

    out_dir = CONFIG["SDIR_OUT"]

    fig = Figz(
        out_dir / "annotations.stx.d",
        name="Annotated Figure",
        size_mm={"width": 170, "height": 100},
    )

    # === Text Elements ===
    logger.info("Adding text elements...")
    fig.add_element(
        "title",
        "text",
        "Main Title",
        position={"x_mm": 85, "y_mm": 5},
        fontsize=14,
        ha="center",
        va="top",
    )
    fig.add_element(
        "subtitle",
        "text",
        "Subtitle with context",
        position={"x_mm": 85, "y_mm": 12},
        fontsize=10,
        ha="center",
    )

    # === Shape Elements ===
    logger.info("Adding shape elements...")
    fig.add_element(
        "arrow1",
        "shape",
        content={
            "shape_type": "arrow",
            "start": {"x_mm": 50, "y_mm": 45},
            "end": {"x_mm": 35, "y_mm": 48},
        },
        position={"x_mm": 0, "y_mm": 0},
    )
    fig.add_element(
        "bracket1",
        "shape",
        content={
            "shape_type": "bracket",
            "start": {"x_mm": 100, "y_mm": 60},
            "end": {"x_mm": 140, "y_mm": 60},
        },
        position={"x_mm": 0, "y_mm": 0},
    )

    # === Symbol Elements ===
    logger.info("Adding symbol elements...")
    fig.add_element(
        "star1",
        "symbol",
        position={"x_mm": 120, "y_mm": 55},
        symbol_type="star",
        fontsize=14,
        color="gold",
    )
    fig.add_element(
        "sig_mark",
        "symbol",
        position={"x_mm": 120, "y_mm": 65},
        symbol_type="dagger",
        fontsize=12,
    )

    # === Equation Elements ===
    logger.info("Adding equation elements...")
    fig.add_element(
        "eq1",
        "equation",
        position={"x_mm": 85, "y_mm": 85},
        latex=r"$E = mc^2$",
        fontsize=12,
    )
    fig.add_element(
        "eq2",
        "equation",
        position={"x_mm": 85, "y_mm": 92},
        latex=r"$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$",
        fontsize=10,
    )

    # Save
    fig.save()
    logger.info(f"Saved: {fig.path}")

    for elem_type in ["text", "shape", "symbol", "equation"]:
        ids = fig.list_element_ids(elem_type)
        logger.info(f"  {elem_type}: {ids}")

    logger.success("Example 03 completed!")


if __name__ == "__main__":
    main()

# EOF
