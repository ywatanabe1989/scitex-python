#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 12:23:32 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_image.py

import os

__FILE__ = __file__

import io as _io

import plotly
from PIL import Image

from scitex import logging

logger = logging.getLogger(__name__)


def save_image(
    obj,
    spath,
    metadata=None,
    add_qr=False,
    qr_position="bottom-right",
    verbose=False,
    save_stats=True,
    **kwargs,
):
    # Determine if spath is a file-like object (e.g., BytesIO)
    is_file_like = not isinstance(spath, str)

    # Get format from file extension or kwargs
    if is_file_like:
        fmt = kwargs.get('format', '').lower()
    else:
        # Get extension without the leading dot
        fmt = spath.lower().rsplit('.', 1)[-1] if '.' in spath else ''

    # Auto-save stats BEFORE saving (obj may be deleted during save)
    # Only for file paths, not file-like objects
    if save_stats and not is_file_like:
        _save_stats_from_figure(obj, spath, verbose=verbose)

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
                logger.warning(f"Failed to add QR code: {e}")

    # png
    if fmt == 'png':
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
    elif fmt in ('tiff', 'tif'):
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
    elif fmt in ('jpeg', 'jpg'):
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
    elif fmt == 'gif':
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
    elif fmt == 'svg':
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
    elif fmt == 'pdf':
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

    # Embed metadata if provided (only for file paths, not file-like objects)
    if metadata is not None and not is_file_like:
        from .._metadata import embed_metadata

        try:
            embed_metadata(spath, metadata)
            if verbose:
                logger.debug(f"  • Embedded metadata: {metadata}")
        except Exception as e:
            logger.warning(f"Failed to embed metadata: {e}")

def _save_stats_from_figure(obj, spath, verbose=False):
    """
    Extract and save statistical annotations from a figure.

    Saves to {basename}_stats.csv if stats are found.
    """
    try:
        from scitex.bridge import extract_stats_from_axes
    except ImportError:
        return  # Bridge not available

    # Get the matplotlib figure
    fig = None
    if hasattr(obj, "savefig"):
        fig = obj
    elif hasattr(obj, "figure") and hasattr(obj.figure, "savefig"):
        fig = obj.figure
    elif hasattr(obj, "_fig_mpl"):
        fig = obj._fig_mpl

    if fig is None:
        return

    # Extract stats from all axes
    all_stats = []
    try:
        for ax in fig.axes:
            stats = extract_stats_from_axes(ax)
            all_stats.extend(stats)
    except Exception:
        return  # Silently fail if extraction fails

    if not all_stats:
        return  # No stats to save

    # Build stats dataframe
    try:
        import pandas as pd

        stats_data = []
        for stat in all_stats:
            row = {
                "test_type": stat.test_type,
                "statistic_name": stat.statistic.get("name", ""),
                "statistic_value": stat.statistic.get("value", ""),
                "p_value": stat.p_value,
                "stars": stat.stars,
            }
            # Add effect size if available
            if stat.effect_size:
                row["effect_size_name"] = stat.effect_size.get("name", "")
                row["effect_size_value"] = stat.effect_size.get("value", "")
            # Add CI if available
            if stat.ci_95:
                row["ci_95_lower"] = stat.ci_95[0]
                row["ci_95_upper"] = stat.ci_95[1]
            # Add sample/group info if available (for consistent naming with plot CSV)
            if stat.samples:
                for group_name, group_info in stat.samples.items():
                    if isinstance(group_info, dict):
                        row[f"n_{group_name}"] = group_info.get("n")
                        row[f"mean_{group_name}"] = group_info.get("mean")
                        row[f"std_{group_name}"] = group_info.get("std")
            stats_data.append(row)

        stats_df = pd.DataFrame(stats_data)

        # Save to {basename}_stats.csv
        import os
        base, ext = os.path.splitext(spath)
        stats_path = f"{base}_stats.csv"
        stats_df.to_csv(stats_path, index=False)

        if verbose:
            logger.info(f"  • Auto-saved stats to: {stats_path}")

    except Exception as e:
        logger.warning(f"Failed to auto-save stats: {e}")


# EOF
