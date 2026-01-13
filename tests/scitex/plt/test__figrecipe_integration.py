#!/usr/bin/env python3
# Timestamp: "2026-01-12 (ywatanabe)"
# File: tests/scitex/plt/test__figrecipe_integration.py
"""Tests for scitex.plt figrecipe integration."""

import pytest

from scitex.plt._figrecipe_integration import _AVAILABLE


class TestDrawGraph:
    """Tests for draw_graph function."""

    @pytest.mark.skipif(not _AVAILABLE, reason="figrecipe not installed")
    def test_draw_graph_basic(self):
        """Test basic graph drawing."""
        import matplotlib.pyplot as plt
        import networkx as nx

        from scitex.plt import draw_graph

        G = nx.karate_club_graph()
        fig, ax = plt.subplots()

        result = draw_graph(ax, G, layout="spring", seed=42)

        assert "pos" in result
        assert len(result["pos"]) == G.number_of_nodes()
        assert result["node_collection"] is not None

        plt.close(fig)

    @pytest.mark.skipif(not _AVAILABLE, reason="figrecipe not installed")
    def test_draw_graph_with_labels(self):
        """Test graph drawing with labels."""
        import matplotlib.pyplot as plt
        import networkx as nx

        from scitex.plt import draw_graph

        G = nx.path_graph(5)
        fig, ax = plt.subplots()

        result = draw_graph(ax, G, labels=True, font_size=6)

        assert result["label_collection"] is not None

        plt.close(fig)

    @pytest.mark.skipif(not _AVAILABLE, reason="figrecipe not installed")
    def test_draw_graph_unwraps_axis(self):
        """Test that draw_graph properly unwraps scitex AxisWrapper."""
        import matplotlib.pyplot as plt
        import networkx as nx

        from scitex.plt import draw_graph

        G = nx.path_graph(3)
        fig, ax = plt.subplots()

        # Create mock wrapper
        class MockWrapper:
            def __init__(self, ax):
                self._axis_mpl = ax

        wrapped_ax = MockWrapper(ax)

        # Should work with wrapped axis
        result = draw_graph(wrapped_ax, G)
        assert len(result["pos"]) == 3

        plt.close(fig)

    @pytest.mark.skipif(not _AVAILABLE, reason="figrecipe not installed")
    def test_draw_graph_directed(self):
        """Test directed graph drawing."""
        import matplotlib.pyplot as plt
        import networkx as nx

        from scitex.plt import draw_graph

        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        fig, ax = plt.subplots()

        result = draw_graph(ax, G, arrows=True)

        assert result["edge_collection"] is not None

        plt.close(fig)

    @pytest.mark.skipif(not _AVAILABLE, reason="figrecipe not installed")
    def test_draw_graph_custom_styling(self):
        """Test graph drawing with custom styling."""
        import matplotlib.pyplot as plt
        import networkx as nx

        from scitex.plt import draw_graph

        G = nx.complete_graph(4)
        fig, ax = plt.subplots()

        result = draw_graph(
            ax,
            G,
            node_size=200,
            node_color="red",
            edge_width=2.0,
            edge_color="blue",
        )

        assert result["node_collection"] is not None

        plt.close(fig)


class TestEdit:
    """Tests for edit function."""

    @pytest.mark.skipif(not _AVAILABLE, reason="figrecipe not installed")
    def test_edit_import(self):
        """Test that edit function is importable."""
        from scitex.plt import edit

        assert callable(edit)

    def test_import_error_without_figrecipe(self):
        """Test that ImportError is raised when figrecipe unavailable."""
        if _AVAILABLE:
            pytest.skip("figrecipe is installed")

        import networkx as nx

        from scitex.plt import draw_graph

        G = nx.path_graph(3)

        with pytest.raises(ImportError, match="figrecipe"):
            draw_graph(None, G)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_figrecipe_integration.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2026-01-12 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_figrecipe_integration.py
# """Figrecipe integration for scitex.plt (graph visualization, editor)."""
# 
# from typing import Any, Callable, Dict, Optional, Union
# 
# try:
#     import figrecipe as fr
#     from figrecipe._graph import draw_graph as _fr_draw_graph
# 
#     _AVAILABLE = True
# except ImportError:
#     _AVAILABLE = False
# 
# 
# def draw_graph(
#     ax,
#     G,
#     *,
#     layout: str = "spring",
#     pos: Optional[Dict] = None,
#     seed: int = 42,
#     node_size: Union[str, Callable, float] = 100,
#     node_color: Union[str, Callable, Any] = "#3498db",
#     node_alpha: float = 0.8,
#     edge_width: Union[str, Callable, float] = 1.0,
#     edge_color: Union[str, Callable, Any] = "gray",
#     edge_alpha: float = 0.4,
#     arrows: Optional[bool] = None,
#     labels: Union[bool, Dict, str] = False,
#     font_size: float = 6,
#     colormap: str = "viridis",
#     **kwargs,
# ) -> Dict[str, Any]:
#     """Draw a NetworkX graph on axes.
# 
#     Parameters
#     ----------
#     ax : Axes or AxisWrapper
#         The axes to draw on.
#     G : networkx.Graph
#         The graph to draw.
#     layout : str
#         Layout: 'spring', 'circular', 'kamada_kawai', 'shell', 'hierarchical'.
#     pos : dict, optional
#         Pre-computed positions {node: (x, y)}.
#     seed : int
#         Random seed for layout.
#     node_size, node_color, node_alpha : styling
#         Node appearance settings.
#     edge_width, edge_color, edge_alpha : styling
#         Edge appearance settings.
#     arrows : bool, optional
#         Draw arrows (auto-detect if None).
#     labels : bool, dict, or str
#         Node labels.
#     font_size : float
#         Label font size (6pt default for publication).
#     colormap : str
#         Colormap for numeric node colors.
# 
#     Returns
#     -------
#     dict
#         {'pos': positions, 'node_collection': ..., 'edge_collection': ...}
#     """
#     if not _AVAILABLE:
#         raise ImportError("figrecipe >= 0.13.0 required: pip install figrecipe")
# 
#     # Unwrap scitex AxisWrapper
#     ax_mpl = getattr(ax, "_axis_mpl", getattr(ax, "_ax", ax))
# 
#     return _fr_draw_graph(
#         ax_mpl,
#         G,
#         layout=layout,
#         pos=pos,
#         seed=seed,
#         node_size=node_size,
#         node_color=node_color,
#         node_alpha=node_alpha,
#         edge_width=edge_width,
#         edge_color=edge_color,
#         edge_alpha=edge_alpha,
#         arrows=arrows,
#         labels=labels,
#         font_size=font_size,
#         colormap=colormap,
#         **kwargs,
#     )
# 
# 
# def edit(source=None, style=None, port: int = 5050, **kwargs):
#     """Launch interactive GUI editor for figure styling.
# 
#     Parameters
#     ----------
#     source : RecordingFigure, str, Path, or None
#         Figure object or path to .yaml recipe.
#     style : str or dict, optional
#         Style preset or dict.
#     port : int
#         Server port (default: 5050).
#     **kwargs
#         Additional args (host, open_browser, desktop, etc.)
# 
#     Returns
#     -------
#     dict
#         Style overrides after editing session.
#     """
#     if not _AVAILABLE:
#         raise ImportError("figrecipe >= 0.13.0 required: pip install figrecipe")
# 
#     return fr.edit(source, style=style, port=port, **kwargs)
# 
# 
# __all__ = ["draw_graph", "edit"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_figrecipe_integration.py
# --------------------------------------------------------------------------------
