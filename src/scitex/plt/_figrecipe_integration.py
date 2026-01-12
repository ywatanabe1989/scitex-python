#!/usr/bin/env python3
# Timestamp: "2026-01-12 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_figrecipe_integration.py
"""
Figrecipe integration for scitex.plt.

This module provides graph visualization, editor, and other figrecipe features
when figrecipe >= 0.13.0 is installed.
"""

from typing import Any, Callable, Dict, List, Optional, Union

# Check if figrecipe is available
try:
    import figrecipe as fr
    from figrecipe._graph import (
        draw_graph as _fr_draw_graph,
    )
    from figrecipe._graph import (
        graph_to_record,
        record_to_graph,
    )
    from figrecipe._graph_presets import (
        get_preset as _fr_get_preset,
    )
    from figrecipe._graph_presets import (
        list_presets as _fr_list_presets,
    )
    from figrecipe._graph_presets import (
        register_preset as _fr_register_preset,
    )

    FIGRECIPE_AVAILABLE = True
except ImportError:
    FIGRECIPE_AVAILABLE = False


def draw_graph(
    ax,
    G,
    *,
    layout: str = "spring",
    pos: Optional[Dict] = None,
    seed: int = 42,
    preset: Optional[str] = None,
    # Node styling
    node_size: Union[str, Callable, float] = 100,
    node_color: Union[str, Callable, Any] = "#3498db",
    node_alpha: float = 0.8,
    node_shape: str = "o",
    node_edgecolors: str = "white",
    node_linewidths: float = 0.5,
    # Edge styling
    edge_width: Union[str, Callable, float] = 1.0,
    edge_color: Union[str, Callable, Any] = "gray",
    edge_alpha: float = 0.4,
    edge_style: str = "solid",
    arrows: Optional[bool] = None,
    arrowsize: float = 10,
    # Labels
    labels: Union[bool, Dict, str] = False,
    font_size: float = 6,
    font_color: str = "black",
    # Colormap
    colormap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **layout_kwargs,
) -> Dict[str, Any]:
    """Draw a NetworkX graph on scitex axes.

    This function wraps figrecipe's graph visualization with automatic
    axis unwrapping for scitex AxisWrapper objects.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or AxisWrapper
        The axes to draw on.
    G : networkx.Graph
        The graph to draw.
    layout : str
        Layout algorithm: 'spring', 'circular', 'kamada_kawai', 'shell',
        'spectral', 'random', 'planar', 'spiral', 'hierarchical'.
    pos : dict, optional
        Pre-computed node positions {node: (x, y)}.
    seed : int
        Random seed for layout reproducibility.
    preset : str, optional
        Name of a graph preset to apply (overrides other styling options).
    node_size : str, callable, or float
        Node sizes. Can be attribute name, callable (node, data) -> size, or scalar.
    node_color : str, callable, or any
        Node colors. Can be attribute name, callable, color name, or array.
    node_alpha : float
        Node transparency.
    node_shape : str
        Node marker shape.
    edge_width : str, callable, or float
        Edge widths. Can be attribute name, callable (u, v, data) -> width, or scalar.
    edge_color : str, callable, or any
        Edge colors.
    edge_alpha : float
        Edge transparency.
    arrows : bool, optional
        Draw arrows for directed graphs. Auto-detected if None.
    arrowsize : float
        Arrow head size for directed edges.
    labels : bool, dict, or str
        Node labels. True for node IDs, dict for custom labels, str for attribute name.
    font_size : float
        Label font size (default 6pt for publication).
    colormap : str
        Matplotlib colormap for numeric node colors.
    **layout_kwargs
        Additional kwargs passed to layout algorithm.

    Returns
    -------
    dict
        Dictionary with 'pos', 'node_collection', 'edge_collection'.

    Raises
    ------
    ImportError
        If figrecipe is not installed.

    Examples
    --------
    >>> import scitex.plt as splt
    >>> import networkx as nx
    >>>
    >>> G = nx.karate_club_graph()
    >>> fig, ax = splt.subplots()
    >>> splt.draw_graph(ax, G, layout="spring", labels=True)
    >>> fig.savefig("graph.png")

    >>> # With preset
    >>> splt.draw_graph(ax, G, preset="scitex")
    """
    if not FIGRECIPE_AVAILABLE:
        raise ImportError(
            "figrecipe >= 0.13.0 is required for graph visualization. "
            "Install with: pip install figrecipe"
        )

    # Unwrap scitex AxisWrapper if needed
    if hasattr(ax, "_axis_mpl"):
        ax_mpl = ax._axis_mpl
    elif hasattr(ax, "_ax"):
        ax_mpl = ax._ax
    else:
        ax_mpl = ax

    # Apply preset if specified
    if preset is not None:
        preset_kwargs = _fr_get_preset(preset)
        # Preset values serve as defaults, explicit args override
        for key, value in preset_kwargs.items():
            if key not in layout_kwargs:
                if key == "node_size" and node_size == 100:
                    node_size = value
                elif key == "node_color" and node_color == "#3498db":
                    node_color = value
                elif key == "node_alpha" and node_alpha == 0.8:
                    node_alpha = value
                elif key == "edge_width" and edge_width == 1.0:
                    edge_width = value
                elif key == "edge_color" and edge_color == "gray":
                    edge_color = value
                elif key == "edge_alpha" and edge_alpha == 0.4:
                    edge_alpha = value
                elif key == "font_size" and font_size == 6:
                    font_size = value
                else:
                    layout_kwargs[key] = value

    return _fr_draw_graph(
        ax_mpl,
        G,
        layout=layout,
        pos=pos,
        seed=seed,
        node_size=node_size,
        node_color=node_color,
        node_alpha=node_alpha,
        node_shape=node_shape,
        node_edgecolors=node_edgecolors,
        node_linewidths=node_linewidths,
        edge_width=edge_width,
        edge_color=edge_color,
        edge_alpha=edge_alpha,
        edge_style=edge_style,
        arrows=arrows,
        arrowsize=arrowsize,
        labels=labels,
        font_size=font_size,
        font_color=font_color,
        colormap=colormap,
        vmin=vmin,
        vmax=vmax,
        **layout_kwargs,
    )


def get_graph_preset(name: str) -> Dict[str, Any]:
    """Get a graph visualization preset by name.

    Parameters
    ----------
    name : str
        Preset name (e.g., "scitex", "minimal", "colorful").

    Returns
    -------
    dict
        Preset configuration dictionary.

    Raises
    ------
    ImportError
        If figrecipe is not installed.
    """
    if not FIGRECIPE_AVAILABLE:
        raise ImportError(
            "figrecipe >= 0.13.0 is required. Install with: pip install figrecipe"
        )
    return _fr_get_preset(name)


def list_graph_presets() -> List[str]:
    """List available graph visualization presets.

    Returns
    -------
    list of str
        Available preset names.

    Raises
    ------
    ImportError
        If figrecipe is not installed.
    """
    if not FIGRECIPE_AVAILABLE:
        raise ImportError(
            "figrecipe >= 0.13.0 is required. Install with: pip install figrecipe"
        )
    return _fr_list_presets()


def register_graph_preset(name: str, **kwargs):
    """Register a custom graph visualization preset.

    Parameters
    ----------
    name : str
        Name for the preset.
    **kwargs
        Preset configuration (node_size, node_color, edge_width, etc.).

    Raises
    ------
    ImportError
        If figrecipe is not installed.
    """
    if not FIGRECIPE_AVAILABLE:
        raise ImportError(
            "figrecipe >= 0.13.0 is required. Install with: pip install figrecipe"
        )
    return _fr_register_preset(name, **kwargs)


def edit(
    source=None,
    style=None,
    port: int = 5050,
    host: str = "127.0.0.1",
    open_browser: bool = True,
    hot_reload: bool = False,
    working_dir=None,
    desktop: bool = False,
):
    """Launch interactive GUI editor for figure styling.

    This function wraps figrecipe's interactive editor for visual
    figure manipulation and style exploration.

    Parameters
    ----------
    source : RecordingFigure, str, Path, or None
        Either a live RecordingFigure object, path to a .yaml recipe file,
        or None to create a new blank figure.
    style : str or dict, optional
        Style preset name or style dict.
    port : int, optional
        Flask server port (default: 5050).
    host : str, optional
        Host to bind Flask server (default: "127.0.0.1", use "0.0.0.0" for Docker).
    open_browser : bool, optional
        Whether to open browser automatically (default: True).
    hot_reload : bool, optional
        Enable hot reload (default: False).
    working_dir : str or Path, optional
        Working directory for file browser (default: directory containing source).
    desktop : bool, optional
        Launch as native desktop window using pywebview (default: False).
        Requires: pip install figrecipe[desktop]

    Returns
    -------
    dict
        Final style overrides after editing session.

    Raises
    ------
    ImportError
        If figrecipe is not installed.

    Examples
    --------
    >>> import scitex.plt as splt
    >>>
    >>> # Launch empty editor
    >>> splt.edit()
    >>>
    >>> # Edit existing recipe
    >>> splt.edit("my_figure.yaml")
    >>>
    >>> # Edit live figure
    >>> fig, ax = splt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> overrides = splt.edit(fig)
    """
    if not FIGRECIPE_AVAILABLE:
        raise ImportError(
            "figrecipe >= 0.13.0 is required for the editor. "
            "Install with: pip install figrecipe"
        )

    return fr.edit(
        source,
        style=style,
        port=port,
        host=host,
        open_browser=open_browser,
        hot_reload=hot_reload,
        working_dir=working_dir,
        desktop=desktop,
    )


def check_available() -> bool:
    """Check if figrecipe integration is available.

    Returns
    -------
    bool
        True if figrecipe >= 0.13.0 is installed.
    """
    return FIGRECIPE_AVAILABLE


__all__ = [
    "FIGRECIPE_AVAILABLE",
    "draw_graph",
    "get_graph_preset",
    "list_graph_presets",
    "register_graph_preset",
    "edit",
    "check_available",
]

# EOF
