#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 14:52:34 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/io/_save_modules/_save_image.py
# ----------------------------------------
import os
__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import io as _io

import plotly
from PIL import Image


def save_image(obj, spath, metadata=None, add_qr=False, qr_position='bottom-right', **kwargs):
    # Add URL to metadata if not present
    if metadata is not None:
        if 'url' not in metadata:
            metadata = dict(metadata)
            metadata['url'] = 'https://scitex.ai'

        # Add QR code to figure if requested
        if add_qr:
            try:
                from .._qr_utils import add_qr_to_figure
                # Only add QR for matplotlib figures
                if hasattr(obj, 'savefig') or (hasattr(obj, 'figure') and hasattr(obj.figure, 'savefig')):
                    fig = obj if hasattr(obj, 'savefig') else obj.figure
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
                obj.savefig(spath)
            except:
                obj.figure.savefig(spath)
        del obj

    # tiff
    elif spath.endswith(".tiff") or spath.endswith(".tif"):
        # PIL image
        if isinstance(obj, Image.Image):
            obj.save(spath)
        # matplotlib
        else:
            try:
                obj.savefig(spath, dpi=300, format="tiff")
            except:
                obj.figure.savefig(spath, dpi=300, format="tiff")

        del obj

    # jpeg
    elif spath.endswith(".jpeg") or spath.endswith(".jpg"):
        buf = _io.BytesIO()

        # plotly
        if isinstance(obj, plotly.graph_objs.Figure):
            obj.write_image(buf, format="png")
            buf.seek(0)
            img = Image.open(buf)
            img.convert("RGB").save(spath, "JPEG")
            buf.close()

        # PIL image
        elif isinstance(obj, Image.Image):
            obj.save(spath)

        # matplotlib
        else:
            try:
                obj.savefig(buf, format="png")
            except:
                obj.figure.savefig(buf, format="png")

            buf.seek(0)
            img = Image.open(buf)
            img.convert("RGB").save(spath, "JPEG")
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
            try:
                obj.savefig(buf, format="png")
            except:
                obj.figure.savefig(buf, format="png")
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
            try:
                obj.savefig(spath, format="svg")
            except AttributeError:
                obj.figure.savefig(spath, format="svg")
        del obj

    # PDF
    elif spath.endswith(".pdf"):
        # Plotly
        if isinstance(obj, plotly.graph_objs.Figure):
            obj.write_image(file=spath, format="pdf")
        # PIL Image - convert to PDF
        elif isinstance(obj, Image.Image):
            # Convert RGBA to RGB if needed
            if obj.mode == 'RGBA':
                rgb_img = Image.new('RGB', obj.size, (255, 255, 255))
                rgb_img.paste(obj, mask=obj.split()[3])
                rgb_img.save(spath, "PDF")
            else:
                obj.save(spath, "PDF")
        # Matplotlib
        else:
            try:
                obj.savefig(spath, format="pdf")
            except AttributeError:
                obj.figure.savefig(spath, format="pdf")
        del obj

    # Embed metadata if provided
    if metadata is not None:
        from .._metadata import embed_metadata
        try:
            embed_metadata(spath, metadata)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to embed metadata: {e}")

# EOF
