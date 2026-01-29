#!/usr/bin/env python3
# Timestamp: 2026-01-29
# File: src/scitex/canvas/_legacy.py
"""Legacy functions for backward compatibility.

All functions here are deprecated. Use figrecipe instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def add_panel(
    parent_dir,
    canvas_name,
    panel_name,
    source,
    xy_mm=(0, 0),
    size_mm=(50, 50),
    label="",
    bundle=False,
    **kwargs,
):
    """Add a panel to canvas (DEPRECATED).

    Use figrecipe.compose() instead.
    """
    from .io import add_panel_from_image, add_panel_from_scitex

    source = Path(source)
    panel_properties = {
        "position": {"x_mm": xy_mm[0], "y_mm": xy_mm[1]},
        "size": {"width_mm": size_mm[0], "height_mm": size_mm[1]},
        **kwargs,
    }
    if label:
        panel_properties["label"] = {"text": label, "position": "top-left"}

    json_sibling = source.parent / f"{source.stem}.json"
    if json_sibling.exists():
        return add_panel_from_scitex(
            project_dir=parent_dir,
            canvas_name=canvas_name,
            panel_name=panel_name,
            source_png=source,
            panel_properties=panel_properties,
            bundle=bundle,
        )
    else:
        return add_panel_from_image(
            project_dir=parent_dir,
            canvas_name=canvas_name,
            panel_name=panel_name,
            source_image=source,
            panel_properties=panel_properties,
            bundle=bundle,
        )


def save_figure(
    panels,
    path,
    spec=None,
    as_zip=None,
):
    """Save panels as a .figure bundle (DEPRECATED).

    Use figrecipe.compose() and figrecipe.save() instead.
    """
    from scitex.io.bundle import BundleType, save

    p = Path(path)
    spath = str(path)

    if as_zip is None:
        as_zip = spath.endswith(".zip")

    if spec is None:
        spec = _generate_figure_spec(panels)

    bundle_data = {
        "spec": spec,
        "plots": {},
    }

    for panel_id, plot_source in panels.items():
        plot_path = Path(plot_source)
        if plot_path.exists():
            bundle_data["plots"][panel_id] = str(plot_path)

    return save(bundle_data, p, bundle_type=BundleType.FIGURE, as_zip=as_zip)


def load_figure(path):
    """Load a .figure bundle (DEPRECATED).

    Use figrecipe.load() instead.
    """
    from scitex.io.bundle import load

    bundle = load(path)

    if bundle["type"] != "figure":
        raise ValueError(f"Not a .figure bundle: {path}")

    result = {
        "spec": bundle.get("spec", {}),
        "panels": {},
    }

    for panel_id, plot_bundle in bundle.get("plots", {}).items():
        result["panels"][panel_id] = {
            "spec": plot_bundle.get("spec", {}),
            "data": plot_bundle.get("data"),
        }

    return result


def _generate_figure_spec(panels) -> dict[str, Any]:
    """Generate figure.json spec from panels."""
    spec: dict[str, Any] = {
        "schema": {"name": "scitex.canvas.figure", "version": "1.0.0"},
        "figure": {
            "id": "figure",
            "title": "",
            "caption": "",
            "styles": {
                "size": {"width_mm": 180, "height_mm": 120},
                "background": "#ffffff",
            },
        },
        "panels": [],
    }

    panel_ids = sorted(panels.keys())
    n_panels = len(panel_ids)

    if n_panels == 0:
        return spec

    cols = min(n_panels, 2)

    panel_w = 80
    panel_h = 50
    margin = 5

    for i, panel_id in enumerate(panel_ids):
        row = i // cols
        col = i % cols

        x = margin + col * (panel_w + margin)
        y = margin + row * (panel_h + margin)

        spec["panels"].append(
            {
                "id": panel_id,
                "label": panel_id,
                "caption": "",
                "plot": f"{panel_id}.plot",
                "position": {"x_mm": x, "y_mm": y},
                "size": {"width_mm": panel_w, "height_mm": panel_h},
            }
        )

    return spec


# EOF
