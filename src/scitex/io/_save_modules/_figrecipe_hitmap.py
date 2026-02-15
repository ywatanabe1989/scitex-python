#!/usr/bin/env python3
"""Delegate hitmap generation to figrecipe for diagram figures."""

from pathlib import Path


def try_figrecipe_hitmap(obj, spath, kwargs):
    """Generate diagram hitmap via figrecipe if a diagram figure is detected.

    This is a thin delegation layer — all hitmap logic lives in figrecipe.
    """
    try:
        import matplotlib.figure

        mpl_fig = obj
        if not isinstance(obj, matplotlib.figure.Figure):
            mpl_fig = getattr(obj, "fig", getattr(obj, "figure", obj))
        diagram = getattr(mpl_fig, "_figrecipe_diagram", None)
        if diagram is None:
            return
        from figrecipe._diagram._diagram._hitmap import save_diagram_hitmap

        hitmap_path = Path(spath).with_stem(Path(spath).stem + "_hitmap")
        dpi = min(kwargs.get("dpi", 150), 150)
        save_diagram_hitmap(diagram, hitmap_path, dpi=dpi)
    except Exception:
        pass  # figrecipe is optional
