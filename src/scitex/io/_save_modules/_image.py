#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 12:23:32 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_image.py

import os

__FILE__ = __file__

import io as _io
import logging

import plotly
from PIL import Image

logger = logging.getLogger(__name__)


def save_image(
    obj,
    spath,
    metadata=None,
    add_qr=False,
    qr_position="bottom-right",
    verbose=False,
    **kwargs,
):
    # Add URL to metadata if not present
    if metadata is not None:
        if verbose:
            logger.info(f"📝 Saving figure with metadata to: {spath}")

        if "url" not in metadata:
            metadata = dict(metadata)
            metadata["url"] = "https://scitex.ai"
            if verbose:
                logger.info("  • Auto-added URL: https://scitex.ai")

        # Add QR code to figure if requested
        if add_qr:
            if verbose:
                logger.info(f"  • Adding QR code at position: {qr_position}")
            try:
                from .._qr_utils import add_qr_to_figure

                # Only add QR for matplotlib figures
                if hasattr(obj, "savefig") or (
                    hasattr(obj, "figure") and hasattr(obj.figure, "savefig")
                ):
                    fig = obj if hasattr(obj, "savefig") else obj.figure
                    obj = add_qr_to_figure(fig, metadata, position=qr_position)
            except Exception as e:
                import warnings

                warnings.warn(f"Failed to add QR code: {e}")

    # png
    if spath.endswith(".png"):
        # plotly
        if isinstance(obj, plotly.graph_objs.Figure):
            obj.write_image(file=spath, format="png")
        # PIL image
        elif isinstance(obj, Image.Image):
            obj.save(spath)
        # matplotlib
        else:
            try:
                obj.savefig(spath, **kwargs)
            except:
                obj.figure.savefig(spath, **kwargs)
        del obj

    # tiff
    elif spath.endswith(".tiff") or spath.endswith(".tif"):
        # PIL image
        if isinstance(obj, Image.Image):
            obj.save(spath)
        # matplotlib
        else:
            # Use kwargs dpi if provided, otherwise default to 300
            save_kwargs = {"format": "tiff", "dpi": kwargs.get("dpi", 300)}
            save_kwargs.update(kwargs)
            try:
                obj.savefig(spath, **save_kwargs)
            except:
                obj.figure.savefig(spath, **save_kwargs)

        del obj

    # jpeg
    elif spath.endswith(".jpeg") or spath.endswith(".jpg"):
        buf = _io.BytesIO()

        # plotly
        if isinstance(obj, plotly.graph_objs.Figure):
            obj.write_image(buf, format="png")
            buf.seek(0)
            img = Image.open(buf)
            img.convert("RGB").save(
                spath, "JPEG", quality=100, subsampling=0, optimize=False
            )
            buf.close()

        # PIL image
        elif isinstance(obj, Image.Image):
            # Save with maximum quality for JPEG (quality=100 for daily use)
            obj.save(spath, quality=100, subsampling=0, optimize=False)

        # matplotlib
        else:
            save_kwargs = {"format": "png"}
            save_kwargs.update(kwargs)
            try:
                obj.savefig(buf, **save_kwargs)
            except:
                obj.figure.savefig(buf, **save_kwargs)

            buf.seek(0)
            img = Image.open(buf)
            # Save JPEG with very high quality settings for daily use (quality=98 is near-lossless)
            img.convert("RGB").save(
                spath, "JPEG", quality=100, subsampling=0, optimize=False
            )
            buf.close()
        del obj

    # GIF
    elif spath.endswith(".gif"):
        # PIL image
        if isinstance(obj, Image.Image):
            obj.save(spath, save_all=True)
        # plotly - convert via PNG first
        elif isinstance(obj, plotly.graph_objs.Figure):
            buf = _io.BytesIO()
            obj.write_image(buf, format="png")
            buf.seek(0)
            img = Image.open(buf)
            img.save(spath, "GIF")
            buf.close()
        # matplotlib
        else:
            buf = _io.BytesIO()
            save_kwargs = {"format": "png"}
            save_kwargs.update(kwargs)
            try:
                obj.savefig(buf, **save_kwargs)
            except:
                obj.figure.savefig(buf, **save_kwargs)
            buf.seek(0)
            img = Image.open(buf)
            img.save(spath, "GIF")
            buf.close()
        del obj

    # SVG
    elif spath.endswith(".svg"):
        # Plotly
        if isinstance(obj, plotly.graph_objs.Figure):
            obj.write_image(file=spath, format="svg")
        # Matplotlib
        else:
            save_kwargs = {"format": "svg"}
            save_kwargs.update(kwargs)
            try:
                obj.savefig(spath, **save_kwargs)
            except AttributeError:
                obj.figure.savefig(spath, **save_kwargs)
        del obj

    # PDF
    elif spath.endswith(".pdf"):
        # Plotly
        if isinstance(obj, plotly.graph_objs.Figure):
            obj.write_image(file=spath, format="pdf")
        # PIL Image - convert to PDF
        elif isinstance(obj, Image.Image):
            # Convert RGBA to RGB if needed
            if obj.mode == "RGBA":
                rgb_img = Image.new("RGB", obj.size, (255, 255, 255))
                rgb_img.paste(obj, mask=obj.split()[3])
                rgb_img.save(spath, "PDF")
            else:
                obj.save(spath, "PDF")
        # Matplotlib
        else:
            save_kwargs = {"format": "pdf"}
            save_kwargs.update(kwargs)
            try:
                obj.savefig(spath, **save_kwargs)
            except AttributeError:
                obj.figure.savefig(spath, **save_kwargs)
        del obj

    # Embed metadata if provided
    if metadata is not None:
        from .._metadata import embed_metadata

        try:
            embed_metadata(spath, metadata)
            if verbose:
                logger.debug(f"  • Embedded metadata: {metadata}")
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to embed metadata: {e}")

# EOF
