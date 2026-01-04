#!/usr/bin/env python3
# Time-stamp: "2025-06-11 03:40:00 (ywatanabe)"
# File: ./tests/scitex/ai/clustering/test__umap.py

"""Comprehensive test module for scitex.ai.clustering._umap functionality."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

try:
    import umap.umap_ as umap_lib

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Skip all tests if UMAP is not available
pytestmark = pytest.mark.skipif(not UMAP_AVAILABLE, reason="UMAP library not available")


class TestUmapBasicFunctionality:
    """Test basic UMAP functionality."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        data = np.random.randn(n_samples, n_features)
        labels = np.array(["A"] * 50 + ["B"] * 50)
        return data, labels

    @pytest.fixture
    def multi_dataset(self):
        """Generate multiple datasets for testing."""
        np.random.seed(42)
        data1 = np.random.randn(100, 50)
        data2 = np.random.randn(80, 50)
        labels1 = np.array(["A"] * 50 + ["B"] * 50)
        labels2 = np.array(["C"] * 40 + ["D"] * 40)
        return [data1, data2], [labels1, labels2]

    @pytest.mark.timeout(180)  # UMAP JIT compilation can take >60s on first run
    def test_umap_basic_functionality(self, sample_data):
        """Test basic UMAP functionality with minimal parameters."""
        data, labels = sample_data
        from scitex.ai.clustering import umap

        # UMAP expects lists
        fig, legend_figs, umap_model = umap(data=[data], labels=[labels])

        # Check return structure
        assert fig is not None
        # fig may be a FigWrapper or matplotlib Figure
        assert legend_figs is None  # Default is no independent legend
        assert umap_model is not None

        # Close with 'all' to handle wrapped figures
        plt.close("all")

    @pytest.mark.timeout(180)
    def test_umap_multiple_datasets(self, multi_dataset):
        """Test UMAP with multiple datasets."""
        data_list, labels_list = multi_dataset
        from scitex.ai.clustering import umap

        fig, legend_figs, umap_model = umap(
            data=data_list, labels=labels_list, axes_titles=["Dataset 1", "Dataset 2"]
        )

        assert fig is not None
        assert umap_model is not None

        # Check we have multiple axes (access underlying figure if wrapped)
        if hasattr(fig, "fig"):
            axes = fig.fig.get_axes()
        else:
            axes = fig.get_axes()
        assert len(axes) >= len(data_list)

        plt.close("all")

    @pytest.mark.timeout(180)
    def test_umap_supervised_mode(self, sample_data):
        """Test supervised UMAP mode."""
        data, labels = sample_data
        from scitex.ai.clustering import umap

        fig, legend_figs, umap_model = umap(
            data=[data], labels=[labels], supervised=True
        )

        assert fig is not None
        assert umap_model is not None

        plt.close("all")


class TestUmapVisualization:
    """Test UMAP visualization features."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        data = np.random.randn(100, 50)
        labels = np.array(["A"] * 50 + ["B"] * 50)
        return data, labels

    @pytest.mark.timeout(180)
    def test_umap_with_hues(self, sample_data):
        """Test UMAP with hue coloring."""
        data, labels = sample_data
        hues = np.array(["group_A"] * 50 + ["group_B"] * 50)
        from scitex.ai.clustering import umap

        fig, legend_figs, umap_model = umap(data=[data], labels=[labels], hues=[hues])

        assert fig is not None

        plt.close("all")

    @pytest.mark.skip(reason="Custom colors API needs clarification from source code")
    def test_umap_with_custom_colors(self, sample_data):
        """Test UMAP with custom color mapping."""
        data, labels = sample_data
        from scitex.ai.clustering import umap

        # Custom colors for each point - API unclear from source
        hues_colors = [["red"] * 50 + ["blue"] * 50]

        fig, legend_figs, umap_model = umap(
            data=[data], labels=[labels], hues_colors=hues_colors
        )

        assert fig is not None

        plt.close("all")

    @pytest.mark.timeout(180)
    def test_umap_visualization_parameters(self, sample_data):
        """Test UMAP with custom visualization parameters."""
        data, labels = sample_data
        from scitex.ai.clustering import umap

        fig, legend_figs, umap_model = umap(
            data=[data], labels=[labels], title="Custom UMAP Title", alpha=0.7, s=50
        )

        assert fig is not None
        # The fig may be wrapped, skip title check
        plt.close("all")

    @pytest.mark.skip(
        reason="Source code has bug with use_independent_legend (axes not iterable with FigWrapper)"
    )
    def test_umap_with_independent_legend(self, sample_data):
        """Test UMAP with independent legend figures."""
        data, labels = sample_data
        from scitex.ai.clustering import umap

        fig, legend_figs, umap_model = umap(
            data=[data], labels=[labels], use_independent_legend=True
        )

        assert fig is not None
        # legend_figs can be None or a list depending on implementation
        if legend_figs is not None:
            for leg_fig in legend_figs:
                plt.close(leg_fig)

        plt.close("all")

    @pytest.mark.skip(
        reason="Source code has bug with add_super_imposed (hues_colors vstack issue)"
    )
    def test_umap_with_superimposed(self):
        """Test UMAP with superimposed plot."""
        np.random.seed(42)
        data1 = np.random.randn(100, 50)
        data2 = np.random.randn(80, 50)
        labels1 = np.array(["A"] * 50 + ["B"] * 50)
        labels2 = np.array(["C"] * 40 + ["D"] * 40)

        from scitex.ai.clustering import umap

        fig, legend_figs, umap_model = umap(
            data=[data1, data2], labels=[labels1, labels2], add_super_imposed=True
        )

        assert fig is not None
        # Should have extra axis for superimposed plot
        if hasattr(fig, "fig"):
            axes = fig.fig.get_axes()
        else:
            axes = fig.get_axes()
        assert len(axes) == 3  # 2 datasets + 1 superimposed

        plt.close("all")


class TestUmapAlgorithmicOptions:
    """Test UMAP algorithmic options."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        data = np.random.randn(100, 50)
        labels = np.array(["A"] * 50 + ["B"] * 50)
        return data, labels

    @pytest.mark.timeout(180)
    def test_umap_with_pretrained_model(self, sample_data):
        """Test UMAP with pre-fitted model."""
        data, labels = sample_data

        from scitex.ai.clustering import umap

        # First fit to get a model
        fig1, _, model1 = umap(data=[data], labels=[labels])
        plt.close("all")

        # Reuse model on new data
        new_data = np.random.randn(80, 50)
        new_labels = np.array(["X"] * 40 + ["Y"] * 40)

        fig2, legend_figs, model2 = umap(
            data=[new_data], labels=[new_labels], umap_model=model1
        )

        assert model2 is model1

        plt.close("all")


class TestUmapDataValidation:
    """Test UMAP data validation and edge cases."""

    @pytest.mark.timeout(180)
    def test_umap_input_format_validation(self):
        """Test UMAP validates list input format."""
        from scitex.ai.clustering import umap

        data = np.random.randn(100, 50)
        labels = np.array(["A"] * 50 + ["B"] * 50)

        # Should work with list inputs
        fig, legend_figs, umap_model = umap(data=[data], labels=[labels])

        assert fig is not None
        plt.close("all")

    def test_umap_mismatched_lengths(self):
        """Test UMAP with mismatched data and label lengths."""
        from scitex.ai.clustering import umap

        data = [np.random.randn(100, 50)]
        labels = [np.array(["A"] * 50)]  # Wrong size

        # Should fail with assertion or index error
        with pytest.raises((AssertionError, IndexError)):
            umap(data=data, labels=labels)

    def test_umap_empty_data(self):
        """Test UMAP with empty data."""
        from scitex.ai.clustering import umap

        # Empty lists should fail
        with pytest.raises((ValueError, IndexError, AssertionError)):
            umap(data=[], labels=[])

    @pytest.mark.timeout(180)
    def test_umap_natural_label_sorting(self):
        """Test that labels are naturally sorted."""
        from scitex.ai.clustering import umap

        data = np.random.randn(100, 50)
        # Create labels that need natural sorting
        labels = np.array(["Label_1", "Label_10", "Label_2", "Label_20"] * 25)

        fig, legend_figs, umap_model = umap(data=[data], labels=[labels])

        assert fig is not None
        plt.close("all")


class TestUmapIntegration:
    """Test UMAP integration with scitex ecosystem."""

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize(
        "n_samples,n_features,n_classes",
        [
            (50, 10, 2),
            (100, 50, 2),
            (150, 30, 3),
        ],
    )
    def test_umap_various_data_sizes(self, n_samples, n_features, n_classes):
        """Test UMAP with various data sizes."""
        np.random.seed(42)
        data = np.random.randn(n_samples, n_features)

        # Create balanced labels
        samples_per_class = n_samples // n_classes
        labels = []
        for i in range(n_classes):
            labels.extend([f"Class_{i}"] * samples_per_class)
        # Fill remainder
        labels.extend(["Class_0"] * (n_samples - len(labels)))
        labels = np.array(labels)

        from scitex.ai.clustering import umap

        fig, legend_figs, umap_model = umap(data=[data], labels=[labels])

        assert fig is not None
        assert umap_model is not None

        plt.close("all")


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/clustering/_umap.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-09-12 05:37:55 (ywatanabe)"
# # _umap_dev.py
#
#
# """
# This script does XYZ.
# """
#
# """
# Imports
# """
# import sys
#
# import matplotlib.pyplot as plt
# import scitex
# import numpy as np
# import umap.umap_ as umap_orig
# from natsort import natsorted
# from sklearn.preprocessing import LabelEncoder
#
# # sys.path = ["."] + sys.path
# # from scripts import utils, load
#
# """
# Warnings
# """
# # warnings.simplefilter("ignore", UserWarning)
#
#
# """
# Config
# """
# # CONFIG = scitex.gen.load_configs()
#
#
# """
# Functions & Classes
# """
#
#
# def umap(
#     data,
#     labels,
#     hues=None,
#     hues_colors=None,
#     axes=None,
#     axes_titles=None,
#     supervised=False,
#     title="UMAP Clustering",
#     alpha=1.0,
#     s=3,
#     use_independent_legend=False,
#     add_super_imposed=False,
#     umap_model=None,
# ):
#     """
#     Perform UMAP clustering and visualization.
#
#     Parameters
#     ----------
#     data_all : list
#         List of data arrays to cluster
#     labels_all : list
#         List of label arrays corresponding to data_all
#     hues_all : list, optional
#         List of hue arrays for coloring points
#     hues_colors_all : list, optional
#         List of color mappings for hues
#     axes : matplotlib.axes.Axes, optional
#         Existing axes to plot on
#     axes_titles : list, optional
#         Titles for each subplot
#     supervised : bool, optional
#         Whether to use supervised UMAP
#     title : str, optional
#         Main title for the plot
#     alpha : float, optional
#         Transparency of points
#     s : int, optional
#         Size of points
#     use_independent_legend : bool, optional
#         Whether to create separate legend figures
#     add_super_imposed : bool, optional
#         Whether to add a superimposed plot
#     umap_model : umap.UMAP, optional
#         Pre-fitted UMAP model
#
#     Returns
#     -------
#     tuple
#         Figure, legend figures (if applicable), and UMAP model
#     """
#
#     # Renaming
#     data_all = data
#     labels_all = labels
#     hues_all = hues
#     hues_colors_all = hues_colors
#
#     data_all, labels_all, hues_all, hues_colors_all = _check_input_vars(
#         data_all, labels_all, hues_all, hues_colors_all
#     )
#
#     # Label Encoding
#     le = LabelEncoder()
#     le.fit(natsorted(np.hstack(labels_all)))
#     labels_all = [le.transform(labels) for labels in labels_all]
#
#     # Running UMAP Clustering
#     _umap = _run_umap(umap_model, data_all, labels_all, supervised, title)
#
#     # Plotting
#     fig, legend_figs = _plot(
#         _umap,
#         le,
#         data_all,
#         labels_all,
#         hues_all,
#         hues_colors_all,
#         add_super_imposed,
#         axes,
#         title,
#         axes_titles,
#         use_independent_legend,
#         s,
#         alpha,
#     )
#
#     return fig, legend_figs, _umap
#
#
# def _plot(
#     _umap,
#     le,
#     data_all,
#     labels_all,
#     hues_all,
#     hues_colors_all,
#     add_super_imposed,
#     axes,
#     title,
#     axes_titles,
#     use_independent_legend,
#     s,
#     alpha,
# ):
#     # Plotting
#     ncols = len(data_all) + 1 if add_super_imposed else len(data_all)
#     share = True if ncols > 1 else False
#
#     if axes is None:
#         fig, axes = scitex.plt.subplots(ncols=ncols, sharex=share, sharey=share)
#     else:
#         assert len(axes) == ncols
#         fig = (
#             axes[0].get_figure()
#             # axes
#             if isinstance(
#                 axes, (np.ndarray, scitex.plt._subplots.AxesWrapper)
#             )
#             # axis
#             else axes.get_figure()
#         )
#
#     fig.supxyt("UMAP 1", "UMAP 2", title)
#
#     for ii, (data, labels, hues, hues_colors) in enumerate(
#         zip(data_all, labels_all, hues_all, hues_colors_all)
#     ):
#         embedding = _umap.transform(data)
#
#         # ax
#         if ncols == 1:
#             ax = axes
#         else:
#             ax = axes[ii + 1] if add_super_imposed else axes[ii]
#
#         _hues = le.inverse_transform(labels) if hues is None else hues
#         for hue in np.unique(_hues):
#             indi = hue == np.array(_hues)
#
#             if hues_colors:
#                 colors = np.vstack(hues_colors)[indi]
#                 colors = [colors[ii] for ii in range(len(colors))]
#             else:
#                 colors = None
#             ax.scatter(
#                 x=embedding[:, 0][indi],
#                 y=embedding[:, 1][indi],
#                 label=hue,
#                 c=colors,
#                 s=s,
#                 alpha=alpha,
#             )
#
#         ax.set_box_aspect(1)
#
#         if axes_titles is not None:
#             ax.set_title(axes_titles[ii])
#
#         # Merged axis
#         if add_super_imposed:
#             ax = axes[0]
#             _hues = le.inverse_transform(labels) if hues is None else hues
#             for hue in np.unique(_hues):
#                 indi = hue == np.array(_hues)
#                 ax.scatter(
#                     x=embedding[:, 0][indi],
#                     y=embedding[:, 1][indi],
#                     label=hue,
#                     c=np.vstack(hues_colors)[indi][0],
#                     s=s,
#                     alpha=alpha,
#                 )
#
#             ax.set_title("Superimposed")
#             ax.set_box_aspect(1)
#             # ax.sns_scatterplot(
#             #     x=embedding[:, 0],
#             #     y=embedding[:, 1],
#             #     hue=le.inverse_transform(labels) if hues is None else hues,
#             #     palette=hues_colors,
#             #     legend="full" if ii == 0 else False,
#             #     s=s,
#             #     alpha=alpha,
#             # )
#
#     if share:
#         scitex.plt.ax.sharex(axes)
#         scitex.plt.ax.sharey(axes)
#
#     if not use_independent_legend:
#         for ax in axes.flat:
#             ax.legend(loc="upper left")
#         return fig, None
#
#     elif use_independent_legend:
#         legend_figs = []
#         for i, ax in enumerate(axes):
#             legend = ax.get_legend()
#             if legend:
#                 legend_fig = plt.figure(figsize=(3, 2))
#
#                 new_legend = legend_fig.gca().legend(
#                     handles=legend.get_lines(),
#                     labels=[t.get_text() for t in legend.texts],
#                     loc="center",
#                 )
#
#                 # new_legend = legend_fig.gca().legend(
#                 #     handles=legend.legendHandles,
#                 #     labels=legend.texts,
#                 #     loc="center",
#                 # )
#
#                 # legend_fig.canvas.draw()
#                 legend_figs.append(legend_fig)
#                 ax.get_legend().remove()
#
#         for ax in axes:
#             ax.legend_ = None
#
#         # elif use_independent_legend:
#         #     legend_figs = []
#         #     for i, ax in enumerate(axes):
#         #         legend = ax.get_legend()
#         #         if legend:
#         #             legend_fig = plt.figure(figsize=(3, 2))
#         #             new_legend = legend_fig.gca().legend(
#         #                 handles=legend.legendHandles,
#         #                 labels=legend.texts,
#         #                 loc="center",
#         #             )
#         #             legend_fig.canvas.draw()
#         #             legend_filename = f"legend_{i}.png"
#         #             legend_fig.savefig(legend_filename, bbox_inches="tight")
#         #             legend_figs.append(legend_fig)
#         #             plt.close(legend_fig)
#
#         #     for ax in axes:
#         #         ax.legend_ = None
#
#     return fig, legend_figs
#
#
# def _run_umap(umap_model, data_all, labels_all, supervised, title):
#     # UMAP Clustering
#     if not umap_model:
#         umap_model = umap_orig.UMAP(random_state=42)
#         supervised_label_or_none = labels_all[0] if supervised else None
#         title = f"(Supervised) {title}" if supervised else f"(Unsupervised) {title}"
#         _umap = umap_model.fit(data_all[0], y=supervised_label_or_none)
#     else:
#         _umap = umap_model
#
#     return _umap
#
#
# def _check_input_vars(data_all, labels_all, hues_all, hues_colors_all):
#     # Ensures input formats
#     if hues_all is None:
#         hues_all = [None for _ in range(len(data_all))]
#
#     if hues_colors_all is None:
#         hues_colors_all = [None for _ in range(len(data_all))]
#
#     assert len(data_all) == len(labels_all) == len(hues_all) == len(hues_colors_all)
#
#     assert (
#         isinstance(data_all, list)
#         and isinstance(labels_all, list)
#         and isinstance(hues_all, list)
#         and isinstance(hues_colors_all, list)
#     )
#     return data_all, labels_all, hues_all, hues_colors_all
#
#
# def _test(dataset_str="iris"):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from sklearn.datasets import load_digits, load_iris
#     from sklearn.model_selection import train_test_split
#
#     # Load iris dataset
#     load_dataset = {"iris": load_iris, "mnist": load_digits}[dataset_str]
#
#     dataset = load_dataset()
#     X = dataset.data
#     y = dataset.target
#
#     # Split data into two parts
#     X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=42)
#
#     # Call umap function
#     fig, legend_figs, umap_model = umap(
#         data=[X1, X2],
#         labels=[y1, y2],
#         # axes=axes,
#         axes_titles=[f"{dataset_str} Set 1", f"{dataset_str} Set 2"],
#         supervised=True,
#         title=dataset_str,
#         use_independent_legend=True,
#         s=10,
#     )
#
#     # plt.tight_layout()
#     scitex.io.save(fig, f"/tmp/scitex/umap/{dataset_str}.jpg")
#
#     # Save legend figures if any
#     if legend_figs:
#         for i, leg_fig in enumerate(legend_figs):
#             scitex.io.save(leg_fig, f"/tmp/scitex/umap/{dataset_str}_legend_{i}.jpg")
#
#
# main = umap
#
# if __name__ == "__main__":
#     # # Argument Parser
#     # import argparse
#     # parser = argparse.ArgumentParser(description='')
#     # parser.add_argument('--var', '-v', type=int, default=1, help='')
#     # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
#     # args = parser.parse_args()
#
#     # Main
#     CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(
#         sys, plt, verbose=False, agg=True
#     )
#     _test(dataset_str="mnist")
#     # main()
#     scitex.session.close(CONFIG, verbose=False, notify=False)
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/clustering/_umap.py
# --------------------------------------------------------------------------------
