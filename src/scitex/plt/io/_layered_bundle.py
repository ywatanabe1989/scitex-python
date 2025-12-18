#!/usr/bin/env python3
# Timestamp: "2025-12-14 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/io/_layered_bundle.py

"""
Layered .pltz Bundle I/O - New schema with spec/style/geometry separation.

Bundle structure:
    plot.pltz.d/
        spec.json           # Semantic: WHAT to plot (canonical)
        style.json          # Appearance: HOW it looks (canonical)
        data.csv            # Raw data (immutable)
        exports/
            overview.png    # Preview image
            overview.svg    # Vector preview
        cache/
            geometry_px.json    # Derived: WHERE in pixels (regenerable)
            render_manifest.json # Render metadata (dpi, hashes)
            hitmap.png          # Hit testing image (regenerable)
            hitmap.svg          # Vector hit testing (regenerable)

Design Principles:
- spec.json + style.json = source of truth (edit these)
- cache/* = derived, can be deleted and regenerated
- Canonical units: ratio (0-1) for axes bbox, mm for panel size
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from scitex import logging
from scitex.plt.styles import get_default_dpi
from scitex.schema import (
    PLOT_GEOMETRY_VERSION,
    # Version constants
    PLOT_SPEC_VERSION,
    PLOT_STYLE_VERSION,
    BboxPx,
    PltzGeometry,
    PltzRenderedArtist,
    PltzRenderedAxes,
    PltzRenderManifest,
)

from ._pltz_builders import build_pltz_spec, build_pltz_style
from ._pltz_coords import adjust_coords_for_offset, adjust_path_data_for_offset
from ._pltz_load import load_layered_pltz_bundle, merge_layered_bundle
from ._pltz_overview import generate_pltz_overview, generate_pltz_readme

logger = logging.getLogger()

__all__ = [
    "save_layered_pltz_bundle",
    "load_layered_pltz_bundle",
    "merge_layered_bundle",
    "is_layered_bundle",
]


def is_layered_bundle(bundle_dir: Path) -> bool:
    """Check if a bundle uses the new layered format."""
    return (bundle_dir / "spec.json").exists()


def save_layered_pltz_bundle(
    fig,
    bundle_dir: Path,
    basename: str = "plot",
    dpi: Optional[int] = None,
    csv_df=None,
) -> None:
    """
    Save matplotlib figure as layered .pltz bundle.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or FigureWrapper
        The figure to save. Accepts both wrapped and unwrapped figures.
    bundle_dir : Path
        Output directory (e.g., plot.pltz.d).
    basename : str
        Base filename for exports.
    dpi : int, optional
        DPI for raster exports. If None, uses get_default_dpi() from config.
    csv_df : DataFrame, optional
        Data to embed as CSV.
    """
    # Unwrap figure if it's a FigureWrapper from scitex.plt.subplots()
    if hasattr(fig, "fig"):
        fig = fig.fig

    # Resolve DPI from config if not specified
    if dpi is None:
        dpi = get_default_dpi()
    import tempfile
    import warnings

    from PIL import Image as PILImage

    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    exports_dir = bundle_dir / "exports"
    cache_dir = bundle_dir / "cache"
    exports_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    # Extract figure dimensions
    fig_width_inch, fig_height_inch = fig.get_size_inches()

    # Build spec using extracted builder
    spec, csv_df, csv_hash, axes_list = build_pltz_spec(fig, basename, csv_df)

    # Build style using extracted builder
    style = build_pltz_style(fig, axes_list, fig_width_inch, fig_height_inch)

    # === Save exports and track coordinate transformations ===
    #
    # Cropping Pipeline:
    #
    #   ┌──────────────────────────────┐
    #   │  original_figure_size_px     │  ← matplotlib figure canvas
    #   │  ┌────────────────────────┐  │     (where bbox_px are measured)
    #   │  │ tight_bbox             │  │
    #   │  │  ┌──────────────────┐  │  │
    #   │  │  │ final_image_px   │  │  │  ← exported PNG (what user sees)
    #   │  │  └──────────────────┘  │  │
    #   │  └────────────────────────┘  │
    #   └──────────────────────────────┘
    #
    # To convert: final_coord = original_coord - total_crop_offset
    #
    # IMPORTANT: Use fig.dpi for coordinate calculations, NOT export dpi
    # extract_selectable_regions uses fig.dpi for all bbox calculations
    fig_dpi = fig.dpi
    display_fig_size_px = [
        int(fig.get_figwidth() * fig_dpi),
        int(fig.get_figheight() * fig_dpi),
    ]

    # Get matplotlib's tight bounding box (what bbox_inches='tight' crops to)
    # Note: get_tightbbox returns values in INCHES, not pixels
    fig.canvas.draw()  # Ensure renderer is ready
    renderer = fig.canvas.get_renderer()
    tight_bbox_inches = fig.get_tightbbox(renderer)

    # Convert from inches to display pixels (using fig.dpi, NOT export dpi)
    tight_bbox_display_px = {
        "x0": tight_bbox_inches.x0 * fig_dpi,
        "y0": tight_bbox_inches.y0 * fig_dpi,
        "x1": tight_bbox_inches.x1 * fig_dpi,
        "y1": tight_bbox_inches.y1 * fig_dpi,
    }

    # Convert from matplotlib display coords (y=0 at bottom)
    # to image coords (y=0 at top)
    tight_bbox_in_image_coords = {
        "left": tight_bbox_display_px["x0"],
        "upper": display_fig_size_px[1] - tight_bbox_display_px["y1"],  # Flip y
        "right": tight_bbox_display_px["x1"],
        "lower": display_fig_size_px[1] - tight_bbox_display_px["y0"],
    }

    # Scale factor: export_dpi / fig_dpi (to scale from display coords to export PNG coords)
    dpi_scale = dpi / fig_dpi

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*tight_layout.*")

            # Save PNG at full figure size (crop function will handle tight cropping)
            # This ensures coordinate transformations are accurate
            png_path = exports_dir / f"{basename}.png"
            fig.savefig(png_path, dpi=dpi, format="png", transparent=True)

            # Save SVG with tight bbox (separate concern, no coord issues)
            svg_path = exports_dir / f"{basename}.svg"
            fig.savefig(svg_path, bbox_inches="tight", format="svg")

            # Generate hitmap
            from scitex.plt.utils._hitmap import (
                HITMAP_AXES_COLOR,
                HITMAP_BACKGROUND_COLOR,
                apply_hitmap_colors,
                extract_path_data,
                extract_selectable_regions,
                restore_original_colors,
            )

            original_props, color_map, groups = apply_hitmap_colors(fig)

            # Store and set hitmap colors for hitmap generation
            saved_fig_facecolor = fig.patch.get_facecolor()
            saved_ax_facecolors = []
            for ax in axes_list:
                saved_ax_facecolors.append(ax.get_facecolor())
                ax.set_facecolor(HITMAP_BACKGROUND_COLOR)
                for spine in ax.spines.values():
                    spine.set_color(HITMAP_AXES_COLOR)

            fig.patch.set_facecolor(HITMAP_BACKGROUND_COLOR)

            # Save hitmap at full figure size (will crop with same box as main PNG)
            # Hitmap goes to cache/ (derived/regenerable data)
            hitmap_path = cache_dir / "hitmap.png"
            fig.savefig(
                hitmap_path, dpi=dpi, format="png", facecolor=HITMAP_BACKGROUND_COLOR
            )

            # Save hitmap as SVG for vector hit testing
            hitmap_svg_path = cache_dir / "hitmap.svg"
            fig.savefig(
                hitmap_svg_path, format="svg", facecolor=HITMAP_BACKGROUND_COLOR
            )

            # Restore colors
            restore_original_colors(original_props)
            fig.patch.set_facecolor(saved_fig_facecolor)
            for i, ax in enumerate(axes_list):
                ax.set_facecolor(saved_ax_facecolors[i])

            # Apply additional margin cropping (removes transparent edges)
            margin_crop_box = None
            try:
                from scitex.plt.utils._crop import crop

                _, margin_crop_box = crop(
                    str(png_path),
                    output_path=str(png_path),
                    overwrite=True,
                    margin=12,
                    verbose=False,
                    return_offset=True,
                )

                crop(
                    str(hitmap_path),
                    output_path=str(hitmap_path),
                    overwrite=True,
                    crop_box=(
                        margin_crop_box["left"],
                        margin_crop_box["upper"],
                        margin_crop_box["right"],
                        margin_crop_box["lower"],
                    ),
                    verbose=False,
                )
            except Exception as e:
                logger.debug(f"Crop failed: {e}")

    # === Coordinate transformation pipeline ===
    #
    # Strategy: Since we save at full figure size (no bbox_inches='tight'),
    # the crop function's crop_box IS the total offset from original figure to final PNG.
    #
    # We extract coordinates at export DPI so they match the saved PNG directly.

    # Temporarily set fig.dpi to export DPI for coordinate extraction
    saved_fig_dpi = fig.dpi
    fig.set_dpi(dpi)
    fig.canvas.draw()  # Redraw at new DPI

    # crop_box is the total offset from full figure to final PNG
    # (crop_box['left'], crop_box['upper']) = top-left corner of crop region
    total_offset_left = margin_crop_box["left"] if margin_crop_box else 0
    total_offset_upper = margin_crop_box["upper"] if margin_crop_box else 0

    # === Build PltzGeometry (cache) ===
    # Extract at export DPI so coords are in full figure space (matches saved PNG before crop)
    path_data = extract_path_data(fig)
    selectable_regions = extract_selectable_regions(fig)

    # Restore original DPI
    fig.set_dpi(saved_fig_dpi)

    # Get final image size (what user sees)
    with PILImage.open(png_path) as img:
        final_image_size_px = list(img.size)

    # Adjust coordinates: subtract total offset (both tight_bbox and margin_crop)
    # No DPI scaling needed since we extracted at export DPI
    selectable_regions = adjust_coords_for_offset(
        selectable_regions, total_offset_left, total_offset_upper
    )
    path_data = adjust_path_data_for_offset(
        path_data, total_offset_left, total_offset_upper
    )

    rendered_axes = []
    for ax_idx, ax_data in enumerate(path_data.get("axes", [])):
        bbox_data = ax_data.get("bbox_px", {})
        rendered_axes.append(
            PltzRenderedAxes(
                id=f"ax{ax_idx}",
                xlim=ax_data.get("xlim", [0, 1]),
                ylim=ax_data.get("ylim", [0, 1]),
                bbox_px=BboxPx(
                    x0=bbox_data.get("x0", 0),
                    y0=bbox_data.get("y0", 0),
                    width=bbox_data.get(
                        "width", bbox_data.get("x1", 0) - bbox_data.get("x0", 0)
                    ),
                    height=bbox_data.get(
                        "height", bbox_data.get("y1", 0) - bbox_data.get("y0", 0)
                    ),
                ),
            )
        )

    rendered_artists = []
    for artist in path_data.get("artists", []):
        bbox_data = artist.get("bbox_px", {})
        rendered_artists.append(
            PltzRenderedArtist(
                id=str(artist.get("id", "")),
                type=artist.get("type", "unknown"),
                axes_index=artist.get("axes_index", 0),
                bbox_px=BboxPx(
                    x0=bbox_data.get("x0", 0),
                    y0=bbox_data.get("y0", 0),
                    width=bbox_data.get(
                        "width", bbox_data.get("x1", 0) - bbox_data.get("x0", 0)
                    ),
                    height=bbox_data.get(
                        "height", bbox_data.get("y1", 0) - bbox_data.get("y0", 0)
                    ),
                )
                if bbox_data
                else None,
                path_px=artist.get("path_px"),
            )
        )

    geometry = PltzGeometry(
        source_hash=csv_hash or "",
        figure_px=final_image_size_px,  # Final cropped image size
        dpi=dpi,  # Export DPI (stored for consumers)
        axes=rendered_axes,
        artists=rendered_artists,
        hit_regions={
            "strategy": "hybrid",
            "hit_map": "hitmap.png",
            "color_map": {str(k): v for k, v in color_map.items()},
            "groups": groups,
            # Store DPI info for consumers that need to retrieve from data
            "fig_dpi": fig_dpi,  # Original matplotlib fig.dpi
            "export_dpi": dpi,  # Export DPI used for PNG
            "dpi_scale": dpi_scale,  # export_dpi / fig_dpi
        },
        selectable_regions=selectable_regions,
        # Note: crop_box is now None because all coordinates are already adjusted
        # to final_image space (no further transformation needed by consumers)
        crop_box=None,
    )

    # === Save all JSON files ===
    # spec.json
    spec_path = bundle_dir / "spec.json"
    with open(spec_path, "w") as f:
        json.dump(
            {
                "schema": {"name": "scitex.plt.spec", "version": PLOT_SPEC_VERSION},
                **asdict(spec),
            },
            f,
            indent=2,
            default=str,
        )

    # style.json
    style_path = bundle_dir / "style.json"
    with open(style_path, "w") as f:
        json.dump(
            {
                "schema": {"name": "scitex.plt.style", "version": PLOT_STYLE_VERSION},
                **asdict(style),
            },
            f,
            indent=2,
            default=str,
        )

    # cache/geometry_px.json
    geometry_path = cache_dir / "geometry_px.json"
    with open(geometry_path, "w") as f:
        json.dump(
            {
                "schema": {
                    "name": "scitex.plt.geometry",
                    "version": PLOT_GEOMETRY_VERSION,
                },
                "_comment": "CACHE - can be deleted and regenerated from spec + style",
                **asdict(geometry),
            },
            f,
            indent=2,
            default=str,
        )

    # cache/render_manifest.json
    spec_hash = hashlib.sha256(open(spec_path, "rb").read()).hexdigest()[:16]
    style_hash = hashlib.sha256(open(style_path, "rb").read()).hexdigest()[:16]
    manifest = PltzRenderManifest(
        source_hash=f"{spec_hash}:{style_hash}",
        panel_size_mm=[
            round(fig_width_inch * 25.4, 1),
            round(fig_height_inch * 25.4, 1),
        ],
        dpi=dpi,
        render_px=final_image_size_px,
        overview_png=f"exports/{basename}.png",
        overview_svg=f"exports/{basename}.svg",
        hitmap_png="cache/hitmap.png",
    )
    manifest_path = cache_dir / "render_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "schema": {
                    "name": "scitex.plt.render_manifest",
                    "version": PLOT_GEOMETRY_VERSION,
                },
                **asdict(manifest),
            },
            f,
            indent=2,
            default=str,
        )

    # Save CSV
    if csv_df is not None:
        csv_path = bundle_dir / f"{basename}.csv"
        csv_df.to_csv(csv_path, index=False)

    # Generate overview showing main image and hitmap side by side
    generate_pltz_overview(exports_dir, basename, cache_dir)

    # Generate dynamic README.md
    generate_pltz_readme(bundle_dir, basename, spec, style, geometry, manifest)

    logger.debug(f"Saved layered pltz bundle: {bundle_dir}")


# EOF
