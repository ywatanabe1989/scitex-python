#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_stx_bundle.py

"""Save functions for unified .stx bundle format."""

from pathlib import Path


def save_stx_bundle(obj, spath, as_zip=True, bundle_type=None, basename=None, **kwargs):
    """Save an object as a unified .stx bundle.

    The .stx format is the unified bundle format that supports:
    - figure: Publication figures with multiple panels (replaces .figz)
    - plot: Single matplotlib plots (replaces .pltz)
    - stats: Statistical results (replaces .statsz)

    The content type is auto-detected from the object:
    - Figz instance -> delegates to figz.save()
    - matplotlib.figure.Figure -> plot
    - dict with 'panels' or 'elements' -> figure
    - dict with 'comparisons' -> stats

    Bundle structure:
        output.stx.d/
            spec.json           # Type, schema, elements/data
            style.json          # Visual appearance
            data.csv            # Raw data (if applicable)
            exports/            # PNG, SVG, PDF exports

    Parameters
    ----------
    obj : Any
        Object to save (Figz, Figure, dict, etc.)
    spath : str or Path
        Output path (e.g., "output.stx" or "output.stx.d")
    as_zip : bool
        If True (default), save as ZIP archive. Use False for .stx.d directory.
    bundle_type : str, optional
        Force bundle type: 'figure', 'plot', or 'stats'. Auto-detected if None.
    **kwargs
        Additional arguments passed to format-specific savers.
    """
    from scitex.fig import Figz

    if isinstance(obj, Figz):
        # Delegate to Figz.save() - verbose=False since outer _save handles logging
        obj.save(spath, verbose=False)
        return

    p = Path(spath)

    # Extract basename from path if not provided
    if basename is None:
        basename = p.stem
        if basename.endswith(".stx"):
            basename = basename[:-4]
        elif basename.endswith(".d"):
            basename = Path(basename).stem
            if basename.endswith(".stx"):
                basename = basename[:-4]

    # Auto-detect content type from object
    content_type = bundle_type
    if content_type is None:
        import matplotlib.figure

        if isinstance(obj, matplotlib.figure.Figure):
            content_type = "plot"
        elif hasattr(obj, "figure"):
            content_type = "plot"
            obj = obj.figure
        elif isinstance(obj, dict):
            if "panels" in obj or "elements" in obj:
                content_type = "figure"
            elif "comparisons" in obj:
                content_type = "stats"
            else:
                content_type = "figure"  # Default for dicts
        else:
            raise ValueError(
                f"Cannot auto-detect bundle type for {type(obj).__name__}. "
                "Please specify bundle_type='figure', 'plot', or 'stats'."
            )

    # Route to appropriate handler based on content type
    if content_type == "plot":
        from ._pltz_stx import save_pltz_as_stx

        save_pltz_as_stx(obj, spath, as_zip=as_zip, basename=basename, **kwargs)
    elif content_type == "figure":
        import scitex.fig as sfig

        sfig.save_figz(obj, spath, as_zip=as_zip, **kwargs)
    elif content_type == "stats":
        import scitex.stats as sstats

        sstats.save_statsz(obj, spath, as_zip=as_zip, **kwargs)
    else:
        raise ValueError(f"Unknown bundle type: {content_type}")


# EOF
