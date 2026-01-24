#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/ai/classification/reporters/_mixins/_plotting.py

"""
Plotting mixin for classification reporter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from scitex.logging import getLogger

from ._constants import FILENAME_PATTERNS, FOLD_DIR_PREFIX_PATTERN

logger = getLogger(__name__)


class PlottingMixin:
    """Mixin providing plotting methods."""

    def _create_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        labels: List[str],
        fold: int,
        metrics: Dict[str, Any],
    ) -> None:
        """Create and save plots with metric-based filenames."""
        fold_dir = self._create_subdir_if_needed(
            FOLD_DIR_PREFIX_PATTERN.format(fold=fold)
        )
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix plot
        if "confusion_matrix" in metrics:
            self._create_confusion_matrix_plot(metrics, labels, fold, fold_dir)

        # ROC and PR curves
        if y_proba is not None:
            self._create_roc_curve_plot(
                y_true, y_proba, labels, fold, fold_dir, metrics
            )
            self._create_pr_curve_plot(y_true, y_proba, labels, fold, fold_dir, metrics)

        # Metrics dashboard
        summary_filename = FILENAME_PATTERNS["metrics_summary"].format(fold=fold)
        self.plotter.create_metrics_visualization(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            labels=labels,
            save_path=fold_dir / summary_filename,
            title="Classification Metrics Dashboard",
            fold=fold,
            verbose=False,
        )

    def _create_confusion_matrix_plot(self, metrics, labels, fold, fold_dir):
        """Create confusion matrix plot."""
        cm_data = metrics["confusion_matrix"]
        if isinstance(cm_data, dict) and "value" in cm_data:
            cm_data = cm_data["value"]

        balanced_acc = metrics.get("balanced-accuracy", {})
        if isinstance(balanced_acc, dict) and "value" in balanced_acc:
            balanced_acc = balanced_acc["value"]
        elif isinstance(balanced_acc, (float, np.floating)):
            balanced_acc = float(balanced_acc)
        else:
            balanced_acc = None

        if balanced_acc is not None:
            title = (
                f"Confusion Matrix (Fold {fold:02d}) - Balanced Acc: {balanced_acc:.3f}"
            )
            filename = FILENAME_PATTERNS["confusion_matrix_jpg"].format(
                fold=fold, bacc=balanced_acc
            )
        else:
            title = f"Confusion Matrix (Fold {fold:02d})"
            filename = FILENAME_PATTERNS["confusion_matrix_jpg_no_bacc"].format(
                fold=fold
            )

        self.plotter.create_confusion_matrix_plot(
            cm_data, labels=labels, save_path=fold_dir / filename, title=title
        )

    def _create_roc_curve_plot(self, y_true, y_proba, labels, fold, fold_dir, metrics):
        """Create ROC curve plot."""
        roc_auc = metrics.get("roc-auc", {})
        if isinstance(roc_auc, dict) and "value" in roc_auc:
            roc_auc_val = roc_auc["value"]
            roc_filename = FILENAME_PATTERNS["roc_curve_jpg"].format(
                fold=fold, auc=roc_auc_val
            )
        else:
            roc_filename = FILENAME_PATTERNS["roc_curve_jpg_no_auc"].format(fold=fold)

        self.plotter.create_roc_curve(
            y_true,
            y_proba,
            labels=labels,
            save_path=fold_dir / roc_filename,
            title=f"ROC Curve (Fold {fold:02d})",
        )

    def _create_pr_curve_plot(self, y_true, y_proba, labels, fold, fold_dir, metrics):
        """Create precision-recall curve plot."""
        pr_auc = metrics.get("pr-auc", {})
        if isinstance(pr_auc, dict) and "value" in pr_auc:
            pr_auc_val = pr_auc["value"]
            pr_filename = FILENAME_PATTERNS["pr_curve_jpg"].format(
                fold=fold, ap=pr_auc_val
            )
        else:
            pr_filename = FILENAME_PATTERNS["pr_curve_jpg_no_ap"].format(fold=fold)

        self.plotter.create_precision_recall_curve(
            y_true,
            y_proba,
            labels=labels,
            save_path=fold_dir / pr_filename,
            title=f"Precision-Recall Curve (Fold {fold:02d})",
        )

    def create_cv_aggregation_visualizations(
        self,
        output_dir: Optional[Path] = None,
        show_individual_folds: bool = True,
        fold_alpha: float = 0.15,
    ) -> None:
        """Create CV aggregation visualizations with faded individual fold lines."""
        if not self.all_predictions:
            logger.warning("No predictions stored for CV aggregation visualizations")
            return

        if output_dir is None:
            output_dir = self._create_subdir_if_needed("cv_summary")
            output_dir.mkdir(parents=True, exist_ok=True)

        n_folds = len(self.all_predictions)

        # ROC curve
        roc_save_path = output_dir / f"roc_cv_aggregation_n{n_folds}.jpg"
        self.plotter.create_cv_aggregation_plot(
            fold_predictions=self.all_predictions,
            curve_type="roc",
            save_path=roc_save_path,
            show_individual_folds=show_individual_folds,
            fold_alpha=fold_alpha,
            title=f"ROC Curves - Cross Validation (n={n_folds} folds)",
            verbose=True,
        )
        logger.info("Created CV aggregation ROC plot with faded fold lines")

        # PR curve
        pr_save_path = output_dir / f"pr_cv_aggregation_n{n_folds}.jpg"
        self.plotter.create_cv_aggregation_plot(
            fold_predictions=self.all_predictions,
            curve_type="pr",
            save_path=pr_save_path,
            show_individual_folds=show_individual_folds,
            fold_alpha=fold_alpha,
            title=f"Precision-Recall Curves - Cross Validation (n={n_folds} folds)",
            verbose=True,
        )
        logger.info("Created CV aggregation PR plot with faded fold lines")


# EOF
