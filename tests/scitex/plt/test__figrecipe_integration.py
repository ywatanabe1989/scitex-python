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
    pytest.main([__file__, "-v"])

# EOF
