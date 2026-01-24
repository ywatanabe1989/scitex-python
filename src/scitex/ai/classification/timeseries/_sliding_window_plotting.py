#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/ai/classification/timeseries/_sliding_window_plotting.py

"""Plotting mixin for TimeSeriesSlidingWindowSplit visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import scitex as stx

if TYPE_CHECKING:
    from ._sliding_window_core import TimeSeriesSlidingWindowSplitCore

__all__ = ["SlidingWindowPlottingMixin"]

COLORS = stx.plt.color.PARAMS


class SlidingWindowPlottingMixin:
    """Mixin class providing plot_splits visualization for sliding window splitters."""

    def plot_splits(
        self: TimeSeriesSlidingWindowSplitCore,
        X,
        y=None,
        timestamps=None,
        figsize=(12, 6),
        save_path=None,
    ):
        """Visualize the sliding window splits as rectangles.

        Shows train (blue), validation (green), and test (red) sets.
        When val_ratio=0, only shows train and test.
        When undersampling is enabled, shows dropped samples in gray.

        Parameters
        ----------
        X : array-like
            Training data
        y : array-like, optional
            Target variable (required for undersampling visualization)
        timestamps : array-like, optional
            Timestamps (if None, uses sample indices)
        figsize : tuple, default (12, 6)
            Figure size
        save_path : str, optional
            Path to save the plot

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        """
        if timestamps is None:
            timestamps = np.arange(len(X))

        time_order = np.argsort(timestamps)

        # Get splits WITH undersampling (if enabled)
        if self.val_ratio > 0:
            splits = list(self.split_with_val(X, y, timestamps))[:10]
            split_type = "train-val-test"
        else:
            splits = list(self.split(X, y, timestamps))[:10]
            split_type = "train-test"

        if not splits:
            raise ValueError("No splits generated")

        # Get splits WITHOUT undersampling to show dropped samples
        splits_no_undersample = None
        if self.undersample and y is not None:
            original_undersample = self.undersample
            self.undersample = False
            if self.val_ratio > 0:
                splits_no_undersample = list(self.split_with_val(X, y, timestamps))[:10]
            else:
                splits_no_undersample = list(self.split(X, y, timestamps))[:10]
            self.undersample = original_undersample

        fig, ax = stx.plt.subplots(figsize=figsize)

        # Plot each fold
        for fold, split_indices in enumerate(splits):
            y_pos = fold
            self._plot_fold_rectangles(ax, split_indices, time_order, y_pos, fold)

        # Plot dropped samples if undersampling
        if splits_no_undersample is not None:
            self._plot_dropped_samples(ax, splits, splits_no_undersample, time_order, y)

        # Plot kept samples
        self._plot_kept_samples(ax, splits, time_order, y)

        # Format plot
        self._format_plot(ax, X, splits, split_type, y)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def _plot_fold_rectangles(self, ax, split_indices, time_order, y_pos, fold):
        """Plot train/val/test rectangles for a fold."""
        if len(split_indices) == 3:
            train_idx, val_idx, test_idx = split_indices
            self._plot_train_rect(ax, train_idx, time_order, y_pos, fold)
            if len(val_idx) > 0:
                self._plot_val_rect(ax, val_idx, time_order, y_pos, fold)
            self._plot_test_rect(ax, test_idx, time_order, y_pos, fold, is_3way=True)
        else:
            train_idx, test_idx = split_indices
            self._plot_train_rect(ax, train_idx, time_order, y_pos, fold)
            self._plot_test_rect(ax, test_idx, time_order, y_pos, fold, is_3way=False)

    def _plot_train_rect(self, ax, train_idx, time_order, y_pos, fold):
        """Plot training window rectangle."""
        train_positions = [np.where(time_order == idx)[0][0] for idx in train_idx]
        if train_positions:
            train_start = min(train_positions)
            train_end = max(train_positions)
            train_rect = patches.Rectangle(
                (train_start, y_pos - 0.3),
                train_end - train_start + 1,
                0.6,
                linewidth=1,
                edgecolor="blue",
                facecolor="lightblue",
                alpha=0.7,
                label="Train" if fold == 0 else "",
            )
            ax.add_patch(train_rect)

    def _plot_val_rect(self, ax, val_idx, time_order, y_pos, fold):
        """Plot validation window rectangle."""
        val_positions = [np.where(time_order == idx)[0][0] for idx in val_idx]
        if val_positions:
            val_start = min(val_positions)
            val_end = max(val_positions)
            val_rect = patches.Rectangle(
                (val_start, y_pos - 0.3),
                val_end - val_start + 1,
                0.6,
                linewidth=1,
                edgecolor="green",
                facecolor="lightgreen",
                alpha=0.7,
                label="Validation" if fold == 0 else "",
            )
            ax.add_patch(val_rect)

    def _plot_test_rect(self, ax, test_idx, time_order, y_pos, fold, is_3way):
        """Plot test window rectangle."""
        test_positions = [np.where(time_order == idx)[0][0] for idx in test_idx]
        if test_positions:
            test_start = min(test_positions)
            test_end = max(test_positions)
            if is_3way:
                edgecolor = COLORS["RGBA_NORM"]["red"]
                facecolor = COLORS["RGBA_NORM"]["red"]
            else:
                edgecolor = "red"
                facecolor = "lightcoral"
            test_rect = patches.Rectangle(
                (test_start, y_pos - 0.3),
                test_end - test_start + 1,
                0.6,
                linewidth=1,
                edgecolor=edgecolor,
                facecolor=facecolor,
                alpha=0.7,
                label="Test" if fold == 0 else "",
            )
            ax.add_patch(test_rect)

    def _plot_dropped_samples(self, ax, splits, splits_no_undersample, time_order, y):
        """Plot dropped samples from undersampling."""
        np.random.seed(42)
        jitter_strength = 0.15

        for fold, split_indices_no_us in enumerate(splits_no_undersample):
            y_pos = fold
            split_indices_us = splits[fold]

            if len(split_indices_no_us) == 3:
                train_idx_no_us, val_idx_no_us, _ = split_indices_no_us
                train_idx_us, val_idx_us, _ = split_indices_us

                dropped_train = np.setdiff1d(train_idx_no_us, train_idx_us)
                if len(dropped_train) > 0:
                    positions = [
                        np.where(time_order == idx)[0][0] for idx in dropped_train
                    ]
                    jitter = np.random.normal(0, jitter_strength, len(positions))
                    ax.scatter(
                        positions,
                        y_pos + jitter,
                        c="gray",
                        s=15,
                        alpha=0.3,
                        marker="x",
                        label="Dropped (train)" if fold == 0 else "",
                        zorder=2,
                    )

                dropped_val = np.setdiff1d(val_idx_no_us, val_idx_us)
                if len(dropped_val) > 0:
                    positions = [
                        np.where(time_order == idx)[0][0] for idx in dropped_val
                    ]
                    jitter = np.random.normal(0, jitter_strength, len(positions))
                    ax.scatter(
                        positions,
                        y_pos + jitter,
                        c="gray",
                        s=15,
                        alpha=0.3,
                        marker="x",
                        label="Dropped (val)" if fold == 0 else "",
                        zorder=2,
                    )
            else:
                train_idx_no_us, _ = split_indices_no_us
                train_idx_us, _ = split_indices_us

                dropped_train = np.setdiff1d(train_idx_no_us, train_idx_us)
                if len(dropped_train) > 0:
                    positions = [
                        np.where(time_order == idx)[0][0] for idx in dropped_train
                    ]
                    jitter = np.random.normal(0, jitter_strength, len(positions))
                    ax.scatter(
                        positions,
                        y_pos + jitter,
                        c="gray",
                        s=15,
                        alpha=0.3,
                        marker="x",
                        label="Dropped samples" if fold == 0 else "",
                        zorder=2,
                    )

    def _plot_kept_samples(self, ax, splits, time_order, y):
        """Plot kept samples with colors."""
        np.random.seed(42)
        jitter_strength = 0.15

        for fold, split_indices in enumerate(splits):
            y_pos = fold

            if len(split_indices) == 3:
                train_idx, val_idx, test_idx = split_indices
                self._scatter_indices(
                    ax, train_idx, time_order, y_pos, y, fold, "train"
                )
                if len(val_idx) > 0:
                    self._scatter_indices(
                        ax, val_idx, time_order, y_pos, y, fold, "val"
                    )
                self._scatter_indices(ax, test_idx, time_order, y_pos, y, fold, "test")
            else:
                train_idx, test_idx = split_indices
                self._scatter_indices(
                    ax, train_idx, time_order, y_pos, y, fold, "train"
                )
                self._scatter_indices(ax, test_idx, time_order, y_pos, y, fold, "test")

    def _scatter_indices(self, ax, indices, time_order, y_pos, y, fold, set_type):
        """Scatter plot indices with jittering."""
        jitter_strength = 0.15
        positions = [np.where(time_order == idx)[0][0] for idx in indices]

        if not positions:
            return

        jitter = np.random.normal(0, jitter_strength, len(positions))

        color_map = {
            "train": ("darkblue", "o", "Train"),
            "val": ("darkgreen", "^", "Val"),
            "test": ("darkred", "s", "Test"),
        }

        if y is not None:
            class_colors = {
                "train": (
                    COLORS["RGBA_NORM"]["blue"],
                    COLORS["RGBA_NORM"]["lightblue"],
                ),
                "val": (COLORS["RGBA_NORM"]["yellow"], COLORS["RGBA_NORM"]["orange"]),
                "test": (COLORS["RGBA_NORM"]["red"], COLORS["RGBA_NORM"]["brown"]),
            }
            colors = [
                class_colors[set_type][0] if y[idx] == 0 else class_colors[set_type][1]
                for idx in indices
            ]
            ax.scatter(
                positions,
                y_pos + jitter,
                c=colors,
                s=20,
                alpha=0.7,
                marker=color_map[set_type][1],
                label=f"{color_map[set_type][2]} (class 0)" if fold == 0 else "",
                zorder=3,
            )
        else:
            ax.scatter(
                positions,
                y_pos + jitter,
                c=color_map[set_type][0],
                s=20,
                alpha=0.7,
                marker=color_map[set_type][1],
                label=f"{color_map[set_type][2]} points" if fold == 0 else "",
                zorder=3,
            )

    def _format_plot(self, ax, X, splits, split_type, y):
        """Format the plot with labels and legend."""
        ax.set_ylim(-0.5, len(splits) - 0.5)
        ax.set_xlim(0, len(X))
        ax.set_xlabel("Temporal Position (sorted by timestamp)")
        ax.set_ylabel("Fold")

        gap_text = f", Gap: {self.gap}" if self.gap > 0 else ""
        val_text = f", Val ratio: {self.val_ratio:.1%}" if self.val_ratio > 0 else ""
        ax.set_title(
            f"Sliding Window Split Visualization ({split_type})\\n"
            f"Window: {self.window_size}, Step: {self.step_size}, Test: {self.test_size}{gap_text}{val_text}\\n"
            f"Rectangles show windows, dots show actual data points"
        )

        ax.set_yticks(range(len(splits)))
        ax.set_yticklabels([f"Fold {i}" for i in range(len(splits))])

        if y is not None:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            total_class_info = ", ".join(
                [
                    f"Class {cls}: n={count}"
                    for cls, count in zip(unique_classes, class_counts)
                ]
            )

            first_split = splits[0]
            if len(first_split) == 3:
                train_idx, val_idx, test_idx = first_split
                fold_info = f"Fold 0: Train n={len(train_idx)}, Val n={len(val_idx)}, Test n={len(test_idx)}"
            else:
                train_idx, test_idx = first_split
                fold_info = f"Fold 0: Train n={len(train_idx)}, Test n={len(test_idx)}"

            handles, labels = ax.get_legend_handles_labels()
            legend_title = f"Total: {total_class_info}\\n{fold_info}"
            ax.legend(handles, labels, loc="upper right", title=legend_title)
        else:
            ax.legend(loc="upper right")


# EOF
