#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/ai/classification/reporters/_mixins/_storage.py

"""
Storage mixin for classification reporter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from scitex.logging import getLogger

from ..reporter_utils.storage import save_metric
from ._constants import (
    FILENAME_PATTERNS,
    FOLD_DIR_PREFIX_PATTERN,
    FOLD_FILE_PREFIX_PATTERN,
)

logger = getLogger(__name__)


class StorageMixin:
    """Mixin providing storage methods."""

    def _save_fold_metrics(
        self, metrics: Dict[str, Any], fold: int, labels: List[str]
    ) -> None:
        """Save metrics for a specific fold in shallow directory structure."""
        fold_dir = FOLD_DIR_PREFIX_PATTERN.format(fold=fold)

        balanced_acc = self._extract_metric_value(metrics.get("balanced-accuracy"))
        mcc_value = self._extract_metric_value(metrics.get("mcc"))
        roc_auc_value = self._extract_metric_value(metrics.get("roc-auc"))
        pr_auc_value = self._extract_metric_value(metrics.get("pr-auc"))

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, dict) and "value" in metric_value:
                actual_value = metric_value["value"]
            else:
                actual_value = metric_value

            if metric_name == "confusion_matrix":
                self._save_confusion_matrix(
                    actual_value, labels, fold, fold_dir, balanced_acc
                )
            elif metric_name == "classification_report":
                self._save_classification_report(actual_value, fold, fold_dir)
            elif metric_name == "balanced-accuracy" and balanced_acc is not None:
                self._save_scalar_metric(
                    actual_value, "balanced-accuracy", balanced_acc, fold, fold_dir
                )
            elif metric_name == "mcc" and mcc_value is not None:
                self._save_scalar_metric(actual_value, "mcc", mcc_value, fold, fold_dir)
            elif metric_name == "roc-auc" and roc_auc_value is not None:
                self._save_scalar_metric(
                    actual_value, "roc-auc", roc_auc_value, fold, fold_dir
                )
            elif metric_name == "pr-auc" and pr_auc_value is not None:
                self._save_scalar_metric(
                    actual_value, "pr-auc", pr_auc_value, fold, fold_dir
                )

    def _save_confusion_matrix(
        self, actual_value, labels, fold, fold_dir, balanced_acc
    ):
        """Save confusion matrix as CSV."""
        try:
            if isinstance(actual_value, pd.DataFrame):
                cm_df = actual_value.copy()
                cm_df.index = [f"True_{label}" for label in labels]
                cm_df.columns = [f"Pred_{label}" for label in labels]
            else:
                cm_df = pd.DataFrame(
                    actual_value,
                    index=[f"True_{label}" for label in labels],
                    columns=[f"Pred_{label}" for label in labels],
                )
        except Exception as e:
            logger.error(f"Error formatting confusion matrix: {e}")
            return

        if balanced_acc is not None:
            cm_filename = FILENAME_PATTERNS["confusion_matrix_csv"].format(
                fold=fold, bacc=balanced_acc
            )
        else:
            cm_filename = FILENAME_PATTERNS["confusion_matrix_csv_no_bacc"].format(
                fold=fold
            )

        self.storage.save(cm_df, f"{fold_dir}/{cm_filename}", index=True)

    def _save_classification_report(self, actual_value, fold, fold_dir):
        """Save classification report."""
        report_filename = FILENAME_PATTERNS["classification_report"].format(fold=fold)

        if isinstance(actual_value, pd.DataFrame):
            report_df = actual_value.reset_index()
            report_df = report_df.rename(columns={"index": "class"})
            self.storage.save(report_df, f"{fold_dir}/{report_filename}")
        elif isinstance(actual_value, dict):
            try:
                report_df = pd.DataFrame(actual_value).transpose()
                self.storage.save(report_df, f"{fold_dir}/{report_filename}")
            except Exception:
                report_filename = FILENAME_PATTERNS[
                    "classification_report_json"
                ].format(fold=fold)
                self.storage.save(actual_value, f"{fold_dir}/{report_filename}")
        else:
            report_filename = FILENAME_PATTERNS["classification_report_txt"].format(
                fold=fold
            )
            self.storage.save(actual_value, f"{fold_dir}/{report_filename}")

    def _save_scalar_metric(self, actual_value, metric_name, value, fold, fold_dir):
        """Save scalar metric with value in filename."""
        filename = FILENAME_PATTERNS["fold_metric_with_value"].format(
            fold=fold, metric_name=metric_name, value=value
        )
        save_metric(
            actual_value,
            self.output_dir / f"{fold_dir}/{filename}",
            fold=fold,
            precision=self.precision,
        )

    def save(
        self,
        data: Any,
        relative_path: Union[str, Path],
        fold: Optional[int] = None,
    ) -> Path:
        """Save custom data with automatic fold organization and filename prefixing."""
        if fold is not None:
            path_obj = Path(relative_path)
            filename = path_obj.name
            parent = path_obj.parent

            prefixed_filename = (
                f"{FOLD_FILE_PREFIX_PATTERN.format(fold=fold)}_{filename}"
            )

            if parent and str(parent) != ".":
                relative_path = f"{FOLD_DIR_PREFIX_PATTERN.format(fold=fold)}/{parent}/{prefixed_filename}"
            else:
                relative_path = (
                    f"{FOLD_DIR_PREFIX_PATTERN.format(fold=fold)}/{prefixed_filename}"
                )

        return self.storage.save(data, relative_path)


# EOF
