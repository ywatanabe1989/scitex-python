#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/flask_editor/_core/_export_helpers.py

"""Export and compose helpers for figure bundles."""

import io
import json as json_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from .._core import WebEditor

__all__ = ["export_composed_figure", "compose_panels_to_figure"]


def export_composed_figure(
    editor: "WebEditor",
    formats: List[str] = None,
    dpi: int = 150,
) -> Dict[str, Any]:
    """Compose and export figure to bundle.

    Parameters
    ----------
    editor : WebEditor
        The editor instance with panel_info.
    formats : list of str
        Output formats (default: ["png", "svg"]).
    dpi : int
        Resolution for raster output.

    Returns
    -------
    dict
        Result with 'success' and 'exported' keys.
    """
    if formats is None:
        formats = ["png", "svg"]

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from scitex.io import ZipBundle

    if not editor.panel_info:
        return {"success": False, "error": "No panel info"}

    bundle_path = editor.panel_info.get("bundle_path")
    figure_dir = editor.panel_info.get("figure_dir")

    if not bundle_path and not figure_dir:
        return {"success": False, "error": "No bundle path"}

    figure_name = (
        Path(bundle_path).stem
        if bundle_path
        else (Path(figure_dir).stem.replace(".figure", "") if figure_dir else "figure")
    )

    # Read spec.json and layout.json
    spec, layout_overrides = _read_spec_and_layout(
        bundle_path, figure_dir, editor.panel_info
    )

    # Get figure dimensions
    fig_width_mm, fig_height_mm = _get_figure_dimensions(spec)
    fig_width_in = fig_width_mm / 25.4
    fig_height_in = fig_height_mm / 25.4

    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi, facecolor="white")

    # Compose panels
    _compose_panels(
        fig,
        spec,
        editor.panel_info,
        layout_overrides,
        fig_width_mm,
        fig_height_mm,
    )

    exported = {}

    # Save to bundle
    if bundle_path:
        with ZipBundle(bundle_path, mode="a") as bundle:
            for fmt in formats:
                buf = io.BytesIO()
                fig.savefig(
                    buf,
                    format=fmt,
                    dpi=dpi,
                    bbox_inches="tight",
                    facecolor="white",
                    pad_inches=0.02,
                )
                buf.seek(0)
                export_path = f"exports/{figure_name}.{fmt}"
                bundle.write_bytes(export_path, buf.read())
                exported[fmt] = export_path

    plt.close(fig)
    return {"success": True, "exported": exported}


def compose_panels_to_figure(
    editor: "WebEditor",
    fmt: str = "png",
    dpi: int = 150,
) -> io.BytesIO:
    """Compose panels into a figure and return as BytesIO.

    Parameters
    ----------
    editor : WebEditor
        The editor instance.
    fmt : str
        Output format.
    dpi : int
        Resolution.

    Returns
    -------
    io.BytesIO
        The composed figure as bytes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bundle_path = editor.panel_info.get("bundle_path")
    figure_dir = editor.panel_info.get("figure_dir")

    spec, layout_overrides = _read_spec_and_layout(
        bundle_path, figure_dir, editor.panel_info
    )

    fig_width_mm, fig_height_mm = _get_figure_dimensions(spec)
    fig_width_in = fig_width_mm / 25.4
    fig_height_in = fig_height_mm / 25.4

    fig = plt.figure(
        figsize=(fig_width_in, fig_height_in),
        dpi=dpi,
        facecolor="white",
    )

    _compose_panels(
        fig,
        spec,
        editor.panel_info,
        layout_overrides,
        fig_width_mm,
        fig_height_mm,
    )

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format=fmt if fmt != "jpg" else "jpeg",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.02,
    )
    plt.close(fig)
    buf.seek(0)
    return buf


def _read_spec_and_layout(bundle_path, figure_dir, panel_info):
    """Read spec.json and layout.json from bundle or directory."""
    from scitex.io import ZipBundle

    spec = {}
    layout_overrides = {}

    if bundle_path:
        try:
            with ZipBundle(bundle_path, mode="r") as bundle:
                spec = bundle.read_json("spec.json")
                try:
                    layout_overrides = bundle.read_json("layout.json")
                except:
                    pass
        except:
            pass
    elif figure_dir:
        spec_path = Path(figure_dir) / "spec.json"
        if spec_path.exists():
            with open(spec_path) as f:
                spec = json_module.load(f)
        layout_path = Path(figure_dir) / "layout.json"
        if layout_path.exists():
            with open(layout_path) as f:
                layout_overrides = json_module.load(f)

    # In-memory layout overrides take precedence
    if panel_info and panel_info.get("layout"):
        layout_overrides = panel_info.get("layout", {})

    return spec, layout_overrides


def _get_figure_dimensions(spec):
    """Extract figure dimensions from spec."""
    fig_width_mm = 180
    fig_height_mm = 120

    if "figure" in spec:
        fig_info = spec.get("figure", {})
        styles = fig_info.get("styles", {})
        size = styles.get("size", {})
        fig_width_mm = size.get("width_mm", 180)
        fig_height_mm = size.get("height_mm", 120)

    return fig_width_mm, fig_height_mm


def _compose_panels(
    fig, spec, panel_info, layout_overrides, fig_width_mm, fig_height_mm
):
    """Compose panels onto the figure."""
    import zipfile

    import numpy as np
    from PIL import Image

    from scitex.io import ZipBundle

    panels_spec = spec.get("panels", [])
    panel_paths = panel_info.get("panel_paths", [])
    panel_is_zip = panel_info.get("panel_is_zip", [])

    exclude_patterns = ["hitmap", "overview", "thumb", "preview"]

    for panel_spec in panels_spec:
        panel_id = panel_spec.get("id", "")
        pos = panel_spec.get("position", {})
        size = panel_spec.get("size", {})

        # Skip auxiliary panels
        panel_id_lower = panel_id.lower()
        if any(
            skip in panel_id_lower for skip in ["overview", "thumb", "preview", "aux"]
        ):
            continue

        # Find panel path
        panel_path, panel_name, is_zip = _find_panel_path(
            panel_id, panel_paths, panel_is_zip
        )
        if not panel_path:
            continue

        # Get layout override
        override = layout_overrides.get(panel_name, {})
        override_pos = override.get("position", {})
        override_size = override.get("size", {})

        x_mm = override_pos.get("x_mm", pos.get("x_mm", 0))
        y_mm = override_pos.get("y_mm", pos.get("y_mm", 0))
        w_mm = override_size.get("width_mm", size.get("width_mm", 60))
        h_mm = override_size.get("height_mm", size.get("height_mm", 40))

        x_frac = x_mm / fig_width_mm
        y_frac = 1 - (y_mm + h_mm) / fig_height_mm
        w_frac = w_mm / fig_width_mm
        h_frac = h_mm / fig_height_mm

        # Load and place panel image
        try:
            if is_zip:
                with ZipBundle(panel_path, mode="r") as plot_bundle:
                    with zipfile.ZipFile(panel_path, "r") as zf:
                        png_files = [
                            n
                            for n in zf.namelist()
                            if n.endswith(".png")
                            and "exports/" in n
                            and not any(p in n.lower() for p in exclude_patterns)
                        ]
                        if png_files:
                            preview_path = png_files[0]
                            if ".plot/" in preview_path:
                                preview_path = preview_path.split(".plot/")[-1]
                            img_data = plot_bundle.read_bytes(preview_path)
                            img = Image.open(io.BytesIO(img_data))
                            ax = fig.add_axes([x_frac, y_frac, w_frac, h_frac])
                            ax.imshow(np.array(img))
                            ax.axis("off")
            else:
                plot_dir = Path(panel_path)
                exports_dir = plot_dir / "exports"
                if exports_dir.exists():
                    for png_file in exports_dir.glob("*.png"):
                        if not any(
                            p in png_file.name.lower() for p in exclude_patterns
                        ):
                            img = Image.open(png_file)
                            ax = fig.add_axes([x_frac, y_frac, w_frac, h_frac])
                            ax.imshow(np.array(img))
                            ax.axis("off")
                            break
        except Exception as e:
            print(f"Could not load panel {panel_id}: {e}")

        # Draw panel letter
        if panel_id and len(panel_id) <= 2:
            letter_x = x_frac + 0.01
            letter_y = y_frac + h_frac - 0.02
            fig.text(
                letter_x,
                letter_y,
                panel_id,
                fontsize=14,
                fontweight="bold",
                color="black",
                ha="left",
                va="top",
                transform=fig.transFigure,
                bbox=dict(
                    boxstyle="square,pad=0.1",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.8,
                ),
            )


def _find_panel_path(panel_id, panel_paths, panel_is_zip):
    """Find panel path matching the panel ID."""
    for idx, pp in enumerate(panel_paths):
        pp_name = Path(pp).stem.replace(".plot", "")
        if (
            pp_name == panel_id
            or pp_name.startswith(f"panel_{panel_id}_")
            or pp_name == f"panel_{panel_id}"
            or f"_{panel_id}_" in pp_name
        ):
            panel_name = Path(pp).name
            is_zip = panel_is_zip[idx] if idx < len(panel_is_zip) else False
            return pp, panel_name, is_zip
    return None, None, False


# EOF
