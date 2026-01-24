#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/flask_editor/_core/_routes_export.py

"""Export and download Flask routes for the editor."""

import io
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._core import WebEditor

__all__ = [
    "create_export_route",
    "create_download_route",
    "create_download_figz_route",
]


def create_export_route(app, editor: "WebEditor"):
    """Create the export route."""
    from flask import jsonify, request

    @app.route("/export", methods=["POST"])
    def export_figure():
        try:
            data = request.get_json()
            formats = data.get("formats", ["png", "svg"])

            if not editor.panel_info:
                return jsonify({"success": False, "error": "No panel info available"})

            bundle_path = editor.panel_info.get("bundle_path")
            if not bundle_path:
                return jsonify({"success": False, "error": "Bundle path not available"})

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            from scitex.io import ZipBundle

            figure_name = Path(bundle_path).stem
            dpi = data.get("dpi", 150)

            with ZipBundle(bundle_path, mode="a") as bundle:
                try:
                    spec = bundle.read_json("spec.json")
                except:
                    spec = {}

                fig_width_mm, fig_height_mm = _get_figure_dimensions(spec)
                fig_width_in = fig_width_mm / 25.4
                fig_height_in = fig_height_mm / 25.4

                fig = plt.figure(
                    figsize=(fig_width_in, fig_height_in),
                    dpi=dpi,
                    facecolor="white",
                )

                _compose_panels_from_spec(
                    fig, spec, bundle, fig_width_mm, fig_height_mm
                )

                exported = {}
                for fmt in formats:
                    buf = io.BytesIO()
                    if fmt in ["png", "jpeg", "jpg"]:
                        fig.savefig(
                            buf,
                            format="png" if fmt == "png" else "jpeg",
                            dpi=dpi,
                            bbox_inches="tight",
                            facecolor="white",
                            pad_inches=0.02,
                        )
                    elif fmt == "svg":
                        fig.savefig(
                            buf, format="svg", bbox_inches="tight", pad_inches=0.02
                        )
                    elif fmt == "pdf":
                        fig.savefig(
                            buf, format="pdf", bbox_inches="tight", pad_inches=0.02
                        )
                    else:
                        continue

                    buf.seek(0)
                    export_path = f"exports/{figure_name}.{fmt}"
                    bundle.write_bytes(export_path, buf.read())
                    exported[fmt] = export_path

                plt.close(fig)

            return jsonify(
                {
                    "success": True,
                    "exported": exported,
                    "bundle_path": str(bundle_path),
                }
            )

        except Exception as e:
            import traceback

            return jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

    return export_figure


def create_download_route(app, editor: "WebEditor"):
    """Create the download route."""
    from flask import send_file

    @app.route("/download/<fmt>")
    def download_figure(fmt):
        try:
            mime_types = {
                "png": "image/png",
                "jpeg": "image/jpeg",
                "jpg": "image/jpeg",
                "svg": "image/svg+xml",
                "pdf": "application/pdf",
            }

            if fmt not in mime_types:
                return f"Unsupported format: {fmt}", 400

            # For figure bundles, download the composed figure
            if editor.panel_info:
                from ._export_helpers import compose_panels_to_figure

                bundle_path = editor.panel_info.get("bundle_path")
                figure_dir = editor.panel_info.get("figure_dir")
                figure_name = (
                    Path(bundle_path).stem
                    if bundle_path
                    else (
                        Path(figure_dir).stem.replace(".figure", "")
                        if figure_dir
                        else "figure"
                    )
                )

                if bundle_path or figure_dir:
                    dpi = 150 if fmt in ["jpeg", "jpg"] else 300
                    buf = compose_panels_to_figure(editor, fmt=fmt, dpi=dpi)
                    return send_file(
                        buf,
                        mimetype=mime_types[fmt],
                        as_attachment=True,
                        download_name=f"{figure_name}.{fmt}",
                    )

            # For single pltz files, render from csv_data
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            from .._renderer import render_preview_with_bboxes

            figure_name = "figure"
            if editor.json_path:
                figure_name = Path(editor.json_path).stem

            img_data, _, _ = render_preview_with_bboxes(
                editor.csv_data,
                editor.current_overrides,
                metadata=editor.metadata,
                dark_mode=False,
            )

            if fmt == "png":
                import base64

                content = base64.b64decode(img_data)
                buf = io.BytesIO(content)
                return send_file(
                    buf,
                    mimetype=mime_types[fmt],
                    as_attachment=True,
                    download_name=f"{figure_name}.{fmt}",
                )

            # For other formats, re-render
            from .._plotter import plot_from_csv

            fig, ax = plt.subplots(figsize=(8, 6))
            plot_from_csv(ax, editor.csv_data, editor.current_overrides)

            buf = io.BytesIO()
            dpi = 150 if fmt in ["jpeg", "jpg"] else 300
            fig.savefig(
                buf,
                format=fmt if fmt != "jpg" else "jpeg",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white" if fmt in ["jpeg", "jpg"] else None,
            )
            plt.close(fig)
            buf.seek(0)

            return send_file(
                buf,
                mimetype=mime_types[fmt],
                as_attachment=True,
                download_name=f"{figure_name}.{fmt}",
            )

        except Exception as e:
            import traceback

            return f"Error: {str(e)}\n{traceback.format_exc()}", 500

    return download_figure


def create_download_figz_route(app, editor: "WebEditor"):
    """Create the download_figz route."""
    from flask import send_file

    @app.route("/download_figz")
    def download_figz():
        try:
            if not editor.panel_info:
                return "No panel info available", 404

            bundle_path = editor.panel_info.get("bundle_path")
            if not bundle_path:
                return "Bundle path not available", 404

            return send_file(
                bundle_path,
                mimetype="application/zip",
                as_attachment=True,
                download_name=Path(bundle_path).name,
            )

        except Exception as e:
            return str(e), 500

    return download_figz


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


def _compose_panels_from_spec(fig, spec, bundle, fig_width_mm, fig_height_mm):
    """Compose panels onto figure from spec (used in export route)."""
    import tempfile

    import numpy as np
    from PIL import Image

    from scitex.io import ZipBundle

    panels_spec = spec.get("panels", [])

    for panel_spec in panels_spec:
        panel_id = panel_spec.get("id", "")
        plot_name = panel_spec.get("plot", "")
        pos = panel_spec.get("position", {})
        size = panel_spec.get("size", {})

        x_mm = pos.get("x_mm", 0)
        y_mm = pos.get("y_mm", 0)
        w_mm = size.get("width_mm", 60)
        h_mm = size.get("height_mm", 40)

        x_frac = x_mm / fig_width_mm
        y_frac = 1 - (y_mm + h_mm) / fig_height_mm
        w_frac = w_mm / fig_width_mm
        h_frac = h_mm / fig_height_mm

        img_loaded = False
        for plot_path in [f"{panel_id}.plot", plot_name.replace(".d", "")]:
            if img_loaded:
                break
            try:
                plot_bytes = bundle.read_bytes(plot_path)
                with tempfile.NamedTemporaryFile(suffix=".plot", delete=False) as tmp:
                    tmp.write(plot_bytes)
                    tmp_path = tmp.name
                try:
                    with ZipBundle(tmp_path, mode="r") as plot_bundle:
                        for preview_path in [
                            "exports/preview.png",
                            "preview.png",
                            f"exports/{panel_id}.png",
                        ]:
                            try:
                                img_data = plot_bundle.read_bytes(preview_path)
                                img = Image.open(io.BytesIO(img_data))
                                ax = fig.add_axes([x_frac, y_frac, w_frac, h_frac])
                                ax.imshow(np.array(img))
                                ax.axis("off")
                                img_loaded = True
                                break
                            except:
                                continue
                finally:
                    import os

                    os.unlink(tmp_path)
            except Exception as e:
                print(f"Could not load plot {plot_path}: {e}")
                continue


# EOF
