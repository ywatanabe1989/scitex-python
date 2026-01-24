#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/ai/classification/reporters/_mixins/_metrics.py

"""
Metrics calculation mixin for classification reporter.
"""

from __future__ import annotations

from pprint import pprint
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scitex.logging import getLogger

from ._constants import FILENAME_PATTERNS, FOLD_DIR_PREFIX_PATTERN

logger = getLogger(__name__)


class MetricsMixin:
    """Mixin providing metrics calculation methods."""

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        fold: Optional[int] = None,
        verbose: bool = True,
        store_y_true: bool = True,
        store_y_pred: bool = True,
        store_y_proba: bool = True,
        model: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Calculate and save classification metrics using unified API."""
        from ..reporter_utils import (
            calc_bacc,
            calc_clf_report,
            calc_conf_mat,
            calc_mcc,
        )

        if verbose:
            if fold:
                print()
                logger.info(f"Calculating metrics for fold #{fold:02d}...")
            else:
                logger.info("Calculating metrics...")

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        if y_proba is not None and len(y_true) != len(y_proba):
            raise ValueError("y_true and y_proba must have same length")

        if fold is None:
            fold = 0

        if labels is None:
            unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
            labels = [f"Class_{i}" for i in unique_labels]

        metrics = {}
        metrics["balanced-accuracy"] = calc_bacc(y_true, y_pred, fold=fold)
        metrics["mcc"] = calc_mcc(y_true, y_pred, fold=fold)
        metrics["confusion_matrix"] = calc_conf_mat(
            y_true=y_true, y_pred=y_pred, labels=labels, fold=fold
        )
        metrics["classification_report"] = calc_clf_report(
            y_true, y_pred, labels, fold=fold
        )

        if y_proba is not None:
            try:
                from scitex.ai.metrics import calc_pre_rec_auc, calc_roc_auc

                metrics["roc-auc"] = calc_roc_auc(
                    y_true, y_proba, labels=labels, fold=fold, return_curve=False
                )
                metrics["pr-auc"] = calc_pre_rec_auc(
                    y_true, y_proba, labels=labels, fold=fold, return_curve=False
                )
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")

        metrics = self._round_numeric(metrics)
        metrics["labels"] = labels

        if verbose:
            logger.info("Metrics calculated:")
            pprint(metrics)

        self.fold_metrics[fold] = metrics.copy()

        if y_proba is not None:
            self.all_predictions.append(
                {
                    "fold": fold,
                    "y_true": y_true.copy(),
                    "y_proba": y_proba.copy(),
                }
            )

        self._save_fold_metrics(metrics, fold, labels)
        self._create_plots(y_true, y_pred, y_proba, labels, fold, metrics)

        if model is not None and feature_names is not None:
            self._extract_and_save_feature_importance(
                model, feature_names, fold, metrics, verbose
            )

        if store_y_true or store_y_pred or store_y_proba:
            self._store_raw_predictions(
                y_true, y_pred, y_proba, fold, store_y_true, store_y_pred, store_y_proba
            )

        return metrics

    def _extract_and_save_feature_importance(
        self, model, feature_names, fold, metrics, verbose
    ):
        """Extract and save feature importance from model."""
        try:
            from scitex.ai.feature_selection import extract_feature_importance

            importance_dict = extract_feature_importance(
                model, feature_names, method="auto"
            )
            if importance_dict:
                metrics["feature-importance"] = importance_dict
                self.fold_metrics[fold]["feature-importance"] = importance_dict

                fold_dir = FOLD_DIR_PREFIX_PATTERN.format(fold=fold)
                filename = FILENAME_PATTERNS["feature_importance_json"].format(
                    fold=fold
                )
                self.storage.save(importance_dict, f"{fold_dir}/{filename}")

                if verbose:
                    logger.info("  Feature importance extracted and saved")
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")

    def _store_raw_predictions(
        self, y_true, y_pred, y_proba, fold, store_y_true, store_y_pred, store_y_proba
    ):
        """Store raw prediction data as CSV files."""
        fold_dir = FOLD_DIR_PREFIX_PATTERN.format(fold=fold)
        sample_indices = np.arange(len(y_true))

        n_samples = len(y_true)
        estimated_size_mb = 0
        if store_y_true:
            estimated_size_mb += n_samples * 0.0001
        if store_y_pred:
            estimated_size_mb += n_samples * 0.0001
        if store_y_proba and y_proba is not None:
            n_classes = 1 if y_proba.ndim == 1 else y_proba.shape[1]
            estimated_size_mb += n_samples * n_classes * 0.0001

        if estimated_size_mb > 10:
            logger.warning(
                f"Storing raw predictions for fold {fold} will create "
                f"~{estimated_size_mb:.1f}MB of CSV files."
            )

        if store_y_true:
            filename = FILENAME_PATTERNS["y_true"].format(fold=fold)
            df_y_true = pd.DataFrame(
                {"sample_index": sample_indices, "fold": fold, "y_true": y_true}
            )
            self.storage.save(df_y_true, f"{fold_dir}/{filename}")

        if store_y_pred:
            filename = FILENAME_PATTERNS["y_pred"].format(fold=fold)
            df_y_pred = pd.DataFrame(
                {"sample_index": sample_indices, "fold": fold, "y_pred": y_pred}
            )
            self.storage.save(df_y_pred, f"{fold_dir}/{filename}")

        if store_y_proba and y_proba is not None:
            filename = FILENAME_PATTERNS["y_proba"].format(fold=fold)
            if y_proba.ndim == 1:
                df_y_proba = pd.DataFrame(
                    {"sample_index": sample_indices, "fold": fold, "y_proba": y_proba}
                )
            else:
                data = {"sample_index": sample_indices, "fold": fold}
                for i in range(y_proba.shape[1]):
                    data[f"proba_class_{i}"] = y_proba[:, i]
                df_y_proba = pd.DataFrame(data)
            self.storage.save(df_y_proba, f"{fold_dir}/{filename}")

    def _extract_metric_value(self, metric_data: Any) -> Optional[float]:
        """Extract numeric value from metric data."""
        if metric_data is None:
            return None
        if isinstance(metric_data, dict) and "value" in metric_data:
            return float(metric_data["value"])
        if isinstance(metric_data, (int, float, np.number)):
            return float(metric_data)
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all calculated metrics across folds."""
        if not self.fold_metrics:
            return {"error": "No metrics calculated yet"}

        summary = {
            "output_dir": str(self.output_dir),
            "total_folds": len(self.fold_metrics),
            "metrics_summary": {},
        }

        confusion_matrices = []
        for fold_metrics in self.fold_metrics.values():
            if "confusion_matrix" in fold_metrics:
                cm_data = fold_metrics["confusion_matrix"]
                if isinstance(cm_data, dict) and "value" in cm_data:
                    cm_data = cm_data["value"]
                if cm_data is not None:
                    confusion_matrices.append(cm_data)

        if confusion_matrices:
            overall_cm = np.sum(confusion_matrices, axis=0)
            summary["overall_confusion_matrix"] = overall_cm.tolist()
            overall_cm_normalized = overall_cm / overall_cm.sum()
            summary["overall_confusion_matrix_normalized"] = self._round_numeric(
                overall_cm_normalized.tolist()
            )

        scalar_metrics = ["balanced-accuracy", "mcc", "roc-auc", "pr-auc"]
        for metric_name in scalar_metrics:
            values = []
            for fold_metrics in self.fold_metrics.values():
                if metric_name in fold_metrics:
                    metric_val = fold_metrics[metric_name]
                    if isinstance(metric_val, dict) and "value" in metric_val:
                        values.append(metric_val["value"])
                    else:
                        values.append(metric_val)

            if values:
                values = np.array(values)
                summary["metrics_summary"][metric_name] = {
                    "mean": self._round_numeric(np.mean(values)),
                    "std": self._round_numeric(np.std(values)),
                    "min": self._round_numeric(np.min(values)),
                    "max": self._round_numeric(np.max(values)),
                    "values": self._round_numeric(values.tolist()),
                }

        feature_importances_list = []
        for fold_metrics in self.fold_metrics.values():
            if "feature-importance" in fold_metrics:
                feature_importances_list.append(fold_metrics["feature-importance"])

        if feature_importances_list:
            from scitex.ai.feature_selection import aggregate_feature_importances

            aggregated_importances = aggregate_feature_importances(
                feature_importances_list
            )
            summary["feature-importance"] = aggregated_importances

        return summary


# EOF
