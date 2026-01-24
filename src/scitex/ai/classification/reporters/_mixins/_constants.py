#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/ai/classification/reporters/_mixins/_constants.py

"""
Constants for classification reporter file naming.
"""

# Fold directory and filename prefixes for consistent naming
FOLD_DIR_PREFIX_PATTERN = "fold_{fold:02d}"  # Directory: fold_00, fold_01, ...
FOLD_FILE_PREFIX_PATTERN = "fold-{fold:02d}"  # Filename prefix: fold-00_, fold-01_, ...

# Filename patterns for consistent naming across the reporter
# Note: fold-{fold:02d} comes first to group files by fold when sorted
# Convention: hyphens within chunks, underscores between chunks
FILENAME_PATTERNS = {
    # Individual fold metrics (with metric value in filename)
    "fold_metric_with_value": f"{FOLD_FILE_PREFIX_PATTERN}_{{metric_name}}-{{value:.3f}}.json",
    "fold_metric": f"{FOLD_FILE_PREFIX_PATTERN}_{{metric_name}}.json",
    # Confusion matrix
    "confusion_matrix_csv": f"{FOLD_FILE_PREFIX_PATTERN}_confusion-matrix_bacc-{{bacc:.3f}}.csv",
    "confusion_matrix_csv_no_bacc": f"{FOLD_FILE_PREFIX_PATTERN}_confusion-matrix.csv",
    "confusion_matrix_jpg": f"{FOLD_FILE_PREFIX_PATTERN}_confusion-matrix_bacc-{{bacc:.3f}}.jpg",
    "confusion_matrix_jpg_no_bacc": f"{FOLD_FILE_PREFIX_PATTERN}_confusion-matrix.jpg",
    # Classification report
    "classification_report": f"{FOLD_FILE_PREFIX_PATTERN}_classification-report.csv",
    # ROC curve
    "roc_curve_csv": f"{FOLD_FILE_PREFIX_PATTERN}_roc-curve_auc-{{auc:.3f}}.csv",
    "roc_curve_csv_no_auc": f"{FOLD_FILE_PREFIX_PATTERN}_roc-curve.csv",
    "roc_curve_jpg": f"{FOLD_FILE_PREFIX_PATTERN}_roc-curve_auc-{{auc:.3f}}.jpg",
    "roc_curve_jpg_no_auc": f"{FOLD_FILE_PREFIX_PATTERN}_roc-curve.jpg",
    # PR curve
    "pr_curve_csv": f"{FOLD_FILE_PREFIX_PATTERN}_pr-curve_ap-{{ap:.3f}}.csv",
    "pr_curve_csv_no_ap": f"{FOLD_FILE_PREFIX_PATTERN}_pr-curve.csv",
    "pr_curve_jpg": f"{FOLD_FILE_PREFIX_PATTERN}_pr-curve_ap-{{ap:.3f}}.jpg",
    "pr_curve_jpg_no_ap": f"{FOLD_FILE_PREFIX_PATTERN}_pr-curve.jpg",
    # Raw prediction data
    "y_true": f"{FOLD_FILE_PREFIX_PATTERN}_y-true.csv",
    "y_pred": f"{FOLD_FILE_PREFIX_PATTERN}_y-pred.csv",
    "y_proba": f"{FOLD_FILE_PREFIX_PATTERN}_y-proba.csv",
    # Metrics dashboard
    "metrics_summary": f"{FOLD_FILE_PREFIX_PATTERN}_metrics-summary.jpg",
    # Feature importance
    "feature_importance_json": f"{FOLD_FILE_PREFIX_PATTERN}_feature-importance.json",
    "feature_importance_jpg": f"{FOLD_FILE_PREFIX_PATTERN}_feature-importance.jpg",
    # Classification report edge cases
    "classification_report_json": f"{FOLD_FILE_PREFIX_PATTERN}_classification-report.json",
    "classification_report_txt": f"{FOLD_FILE_PREFIX_PATTERN}_classification-report.txt",
    # CV summary
    "cv_summary_metric": "cv-summary_{metric_name}_mean-{mean:.3f}_std-{std:.3f}_n-{n_folds}.json",
    "cv_summary_confusion_matrix_csv": "cv-summary_confusion-matrix_bacc-{mean:.3f}_{std:.3f}_n-{n_folds}.csv",
    "cv_summary_confusion_matrix_jpg": "cv-summary_confusion-matrix_bacc-{mean:.3f}_{std:.3f}_n-{n_folds}.jpg",
    "cv_summary_classification_report": "cv-summary_classification-report_n-{n_folds}.csv",
    "cv_summary_roc_curve_csv": "cv-summary_roc-curve_auc-{mean:.3f}_{std:.3f}_n-{n_folds}.csv",
    "cv_summary_roc_curve_jpg": "cv-summary_roc-curve_auc-{mean:.3f}_{std:.3f}_n-{n_folds}.jpg",
    "cv_summary_pr_curve_csv": "cv-summary_pr-curve_ap-{mean:.3f}_{std:.3f}_n-{n_folds}.csv",
    "cv_summary_pr_curve_jpg": "cv-summary_pr-curve_ap-{mean:.3f}_{std:.3f}_n-{n_folds}.jpg",
    "cv_summary_feature_importance_json": "cv-summary_feature-importance_n-{n_folds}.json",
    "cv_summary_feature_importance_jpg": "cv-summary_feature-importance_n-{n_folds}.jpg",
    "cv_summary_summary": "cv-summary_summary.json",
    # CV summary edge cases
    "cv_summary_confusion_matrix_csv_no_bacc": "cv-summary_confusion-matrix_n-{n_folds}.csv",
    "cv_summary_confusion_matrix_jpg_no_bacc": "cv-summary_confusion-matrix_n-{n_folds}.jpg",
}


# EOF
