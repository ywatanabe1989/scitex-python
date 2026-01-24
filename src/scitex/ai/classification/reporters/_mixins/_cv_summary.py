#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/ai/classification/reporters/_mixins/_cv_summary.py

"""
CV summary mixin for classification reporter.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from scitex.logging import getLogger

from ._constants import FILENAME_PATTERNS

logger = getLogger(__name__)


class CVSummaryMixin:
    """Mixin providing CV summary methods."""

    def create_cv_summary_curves(self, summary: Dict[str, Any]) -> None:
        """Create CV summary ROC and PR curves from aggregated predictions."""
        if not self.all_predictions:
            logger.warning("No predictions stored for CV summary curves")
            return

        all_y_true = np.concatenate([p["y_true"] for p in self.all_predictions])
        all_y_proba = np.concatenate([p["y_proba"] for p in self.all_predictions])

        roc_values = []
        pr_values = []
        for metrics in self.fold_metrics.values():
            if "roc-auc" in metrics:
                val = metrics["roc-auc"]
                if isinstance(val, dict) and "value" in val:
                    roc_values.append(val["value"])
                else:
                    roc_values.append(val)
            if "pr-auc" in metrics:
                val = metrics["pr-auc"]
                if isinstance(val, dict) and "value" in val:
                    pr_values.append(val["value"])
                else:
                    pr_values.append(val)

        n_folds = len(self.fold_metrics)
        if roc_values:
            roc_mean = np.mean(roc_values)
            roc_std = np.std(roc_values)
        else:
            from ..reporter_utils.metrics import calc_roc_auc

            overall_roc = calc_roc_auc(all_y_true, all_y_proba)
            roc_mean = overall_roc["value"]
            roc_std = 0.0

        if pr_values:
            pr_mean = np.mean(pr_values)
            pr_std = np.std(pr_values)
        else:
            from ..reporter_utils.metrics import calc_pre_rec_auc

            overall_pr = calc_pre_rec_auc(all_y_true, all_y_proba)
            pr_mean = overall_pr["value"]
            pr_std = 0.0

        cv_summary_dir = self._create_subdir_if_needed("cv_summary")
        cv_summary_dir.mkdir(parents=True, exist_ok=True)

        self._save_cv_summary_curve_data(
            all_y_true, all_y_proba, roc_mean, roc_std, pr_mean, pr_std, n_folds
        )

        from scitex.ai.metrics import _normalize_labels

        all_y_true_norm, _, label_names, _ = _normalize_labels(all_y_true, all_y_true)

        roc_title = f"ROC Curve (CV Summary) - AUC: {roc_mean:.3f} ± {roc_std:.3f} (n={n_folds})"
        roc_filename = FILENAME_PATTERNS["cv_summary_roc_curve_jpg"].format(
            mean=roc_mean, std=roc_std, n_folds=n_folds
        )
        self.plotter.create_overall_roc_curve(
            all_y_true_norm,
            all_y_proba,
            labels=label_names,
            save_path=cv_summary_dir / roc_filename,
            title=roc_title,
            auc_mean=roc_mean,
            auc_std=roc_std,
            verbose=True,
        )

        pr_title = f"Precision-Recall Curve (CV Summary) - AP: {pr_mean:.3f} ± {pr_std:.3f} (n={n_folds})"
        pr_filename = FILENAME_PATTERNS["cv_summary_pr_curve_jpg"].format(
            mean=pr_mean, std=pr_std, n_folds=n_folds
        )
        self.plotter.create_overall_pr_curve(
            all_y_true_norm,
            all_y_proba,
            labels=label_names,
            save_path=cv_summary_dir / pr_filename,
            title=pr_title,
            ap_mean=pr_mean,
            ap_std=pr_std,
            verbose=True,
        )

        logger.info(
            f"Created CV summary ROC curve: AUC = {roc_mean:.3f} ± {roc_std:.3f} (n={n_folds})"
        )
        logger.info(
            f"Created CV summary PR curve: AP = {pr_mean:.3f} ± {pr_std:.3f} (n={n_folds})"
        )

    def _save_cv_summary_curve_data(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        roc_mean: float,
        roc_std: float,
        pr_mean: float,
        pr_std: float,
        n_folds: int,
    ) -> None:
        """Save CV summary ROC and PR curve data as CSV files."""
        from sklearn.metrics import (
            auc,
            average_precision_score,
            precision_recall_curve,
            roc_curve,
        )

        cv_summary_dir = "cv_summary"

        if y_proba.ndim == 1 or y_proba.shape[1] == 2:
            if y_proba.ndim == 2:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba

            from scitex.ai.metrics import _normalize_labels

            y_true_norm, _, _, _ = _normalize_labels(y_true, y_true)

            fpr, tpr, _ = roc_curve(y_true_norm, y_proba_pos)
            roc_auc = auc(fpr, tpr)
            roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
            roc_filename = FILENAME_PATTERNS["cv_summary_roc_curve_csv"].format(
                mean=roc_mean, std=roc_std, n_folds=n_folds
            )
            self.storage.save(roc_df, f"{cv_summary_dir}/{roc_filename}")

            precision, recall, _ = precision_recall_curve(y_true_norm, y_proba_pos)
            avg_precision = average_precision_score(y_true_norm, y_proba_pos)
            pr_df = pd.DataFrame({"Recall": recall, "Precision": precision})
            pr_filename = FILENAME_PATTERNS["cv_summary_pr_curve_csv"].format(
                mean=pr_mean, std=pr_std, n_folds=n_folds
            )
            self.storage.save(pr_df, f"{cv_summary_dir}/{pr_filename}")

    def save_cv_summary_confusion_matrix(self, summary: Dict[str, Any]) -> None:
        """Save and plot the CV summary confusion matrix."""
        confusion_matrices = []
        for fold_metrics in self.fold_metrics.values():
            if "confusion_matrix" in fold_metrics:
                cm_data = fold_metrics["confusion_matrix"]
                if isinstance(cm_data, dict) and "value" in cm_data:
                    cm_data = cm_data["value"]
                if cm_data is not None:
                    confusion_matrices.append(cm_data)

        if not confusion_matrices:
            return

        overall_cm = np.sum(confusion_matrices, axis=0)

        labels = None
        for fold_metrics in self.fold_metrics.values():
            if "labels" in fold_metrics:
                labels = fold_metrics["labels"]
                break
            elif "confusion_matrix" in fold_metrics:
                cm_data = fold_metrics["confusion_matrix"]
                if isinstance(cm_data, dict) and "labels" in cm_data:
                    labels = cm_data["labels"]
                    break

        cv_summary_dir = self._create_subdir_if_needed("cv_summary")
        cv_summary_dir.mkdir(parents=True, exist_ok=True)

        balanced_acc_mean = None
        balanced_acc_std = None
        n_folds = len(self.fold_metrics)
        if "metrics_summary" in summary:
            if "balanced-accuracy" in summary["metrics_summary"]:
                balanced_acc_stats = summary["metrics_summary"]["balanced-accuracy"]
                balanced_acc_mean = balanced_acc_stats.get("mean")
                balanced_acc_std = balanced_acc_stats.get("std")

        if balanced_acc_mean is not None and balanced_acc_std is not None:
            cm_filename = FILENAME_PATTERNS["cv_summary_confusion_matrix_csv"].format(
                mean=balanced_acc_mean, std=balanced_acc_std, n_folds=n_folds
            )
        else:
            cm_filename = FILENAME_PATTERNS[
                "cv_summary_confusion_matrix_csv_no_bacc"
            ].format(n_folds=n_folds)

        if labels:
            cm_df = pd.DataFrame(
                overall_cm,
                index=[f"True_{label}" for label in labels],
                columns=[f"Pred_{label}" for label in labels],
            )
        else:
            cm_df = pd.DataFrame(overall_cm)

        self.storage.save(cm_df, f"cv_summary/{cm_filename}", index=True)

        cv_summary_dir = self._create_subdir_if_needed("cv_summary")
        cv_summary_dir.mkdir(parents=True, exist_ok=True)

        balanced_acc_mean = None
        balanced_acc_std = None
        if "metrics_summary" in self.get_summary():
            metrics_summary = self.get_summary()["metrics_summary"]
            if "balanced-accuracy" in metrics_summary:
                balanced_acc_stats = metrics_summary["balanced-accuracy"]
                balanced_acc_mean = balanced_acc_stats.get("mean")
                balanced_acc_std = balanced_acc_stats.get("std")

        if balanced_acc_mean is not None and balanced_acc_std is not None:
            title = f"Confusion Matrix (CV Summary) - Balanced Acc: {balanced_acc_mean:.3f} ± {balanced_acc_std:.3f} (n={n_folds})"
            filename = FILENAME_PATTERNS["cv_summary_confusion_matrix_jpg"].format(
                mean=balanced_acc_mean, std=balanced_acc_std, n_folds=n_folds
            )
        else:
            title = f"Confusion Matrix (CV Summary) (n={n_folds})"
            filename = FILENAME_PATTERNS[
                "cv_summary_confusion_matrix_jpg_no_bacc"
            ].format(n_folds=n_folds)

        self.plotter.create_confusion_matrix_plot(
            overall_cm,
            labels=labels,
            save_path=cv_summary_dir / filename,
            title=title,
        )

    def _save_cv_summary_metrics(self, summary: Dict[str, Any]) -> None:
        """Save individual CV summary metrics with mean/std/n_folds in filenames."""
        if "metrics_summary" not in summary:
            return

        n_folds = len(self.fold_metrics)
        cv_summary_dir = "cv_summary"

        for metric_name, stats in summary["metrics_summary"].items():
            if isinstance(stats, dict) and "mean" in stats:
                mean_val = stats.get("mean", 0)
                std_val = stats.get("std", 0)

                filename = FILENAME_PATTERNS["cv_summary_metric"].format(
                    metric_name=metric_name,
                    mean=mean_val,
                    std=std_val,
                    n_folds=n_folds,
                )

                self.storage.save(stats, f"{cv_summary_dir}/{filename}")

    def _save_cv_summary_classification_report(self, summary: Dict[str, Any]) -> None:
        """Save CV summary classification report with mean ± std (n_folds=X) format."""
        n_folds = len(self.fold_metrics)
        cv_summary_dir = "cv_summary"

        all_reports = []
        for fold_num, fold_metrics in self.fold_metrics.items():
            if "classification_report" in fold_metrics:
                report = fold_metrics["classification_report"]
                if isinstance(report, dict) and "value" in report:
                    report = report["value"]

                if isinstance(report, pd.DataFrame):
                    if "class" in report.columns:
                        report_dict = {}
                        for _, row in report.iterrows():
                            class_name = row["class"]
                            report_dict[class_name] = {
                                col: row[col]
                                for col in report.columns
                                if col != "class"
                            }
                        report = report_dict
                    else:
                        report = report.to_dict("index")

                if isinstance(report, dict):
                    all_reports.append(report)

        if not all_reports:
            return

        summary_report = {}

        all_classes = set()
        for report in all_reports:
            all_classes.update(
                [
                    k
                    for k in report.keys()
                    if k not in ["accuracy", "macro avg", "weighted avg"]
                ]
            )

        for cls in sorted(all_classes):
            cls_metrics = {
                "precision": [],
                "recall": [],
                "f1-score": [],
                "support": [],
            }

            for report in all_reports:
                if cls in report:
                    for metric in ["precision", "recall", "f1-score", "support"]:
                        if metric in report[cls]:
                            cls_metrics[metric].append(report[cls][metric])

            summary_report[cls] = {}
            for metric, values in cls_metrics.items():
                if values:
                    if metric == "support":
                        total_support = int(np.sum(values))
                        mean_support = np.mean(values)
                        std_support = np.std(values)
                        if std_support > 0:
                            summary_report[cls][metric] = (
                                f"{mean_support:.1f} ± {std_support:.1f} (total={total_support})"
                            )
                        else:
                            summary_report[cls][metric] = (
                                f"{int(mean_support)} per fold (total={total_support})"
                            )
                    else:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        summary_report[cls][metric] = (
                            f"{mean_val:.3f} ± {std_val:.3f} (n={n_folds})"
                        )

        for avg_type in ["macro avg", "weighted avg"]:
            avg_metrics = {"precision": [], "recall": [], "f1-score": []}

            for report in all_reports:
                if avg_type in report:
                    for metric in ["precision", "recall", "f1-score"]:
                        if metric in report[avg_type]:
                            avg_metrics[metric].append(report[avg_type][metric])

            if any(avg_metrics.values()):
                summary_report[avg_type] = {}
                for metric, values in avg_metrics.items():
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        summary_report[avg_type][metric] = (
                            f"{mean_val:.3f} ± {std_val:.3f} (n={n_folds})"
                        )

        if summary_report:
            report_df = pd.DataFrame(summary_report).T
            report_df = report_df.reset_index()
            report_df = report_df.rename(columns={"index": "class"})

            filename = FILENAME_PATTERNS["cv_summary_classification_report"].format(
                n_folds=n_folds
            )
            self.storage.save(report_df, f"{cv_summary_dir}/{filename}")


# EOF
