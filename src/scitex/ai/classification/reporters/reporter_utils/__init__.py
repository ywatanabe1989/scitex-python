#!/usr/bin/env python3
"""
Classification reporter utilities for modular metric calculation and reporting.

This module provides separated, focused utilities for:
- Metric calculations
- File organization
- Validation
- Report generation
"""

# Import from centralized metrics module
from scitex.ai.metrics import (
    calc_bacc,
    calc_clf_report,
    calc_conf_mat,
    calc_mcc,
    calc_pre_rec_auc,
    calc_roc_auc,
)

from .aggregation import (
    aggregate_confusion_matrices,
    aggregate_fold_metrics,
    calculate_mean_std,
    create_summary_table,
)
from .reporting import (
    create_summary_statistics,
    export_for_paper,
    generate_latex_report,
    generate_markdown_report,
)
from .storage import MetricStorage
from .storage import create_directory_structure_lazy as organize_outputs
from .storage import save_metric
from .validation import MetricValidator, check_required_metrics, validate_completeness

__all__ = [
    # Metrics
    "calc_bacc",
    "calc_mcc",
    "calc_conf_mat",
    "calc_clf_report",
    "calc_roc_auc",
    "calc_pre_rec_auc",
    # Storage
    "MetricStorage",
    "save_metric",
    "save_figure",
    "save_dataframe",
    "organize_outputs",
    # Validation
    "MetricValidator",
    "validate_completeness",
    "check_required_metrics",
    # Reporting
    "generate_markdown_report",
    "generate_latex_report",
    "create_summary_statistics",
    "export_for_paper",
    # Aggregation
    "aggregate_fold_metrics",
    "calculate_mean_std",
    "create_summary_table",
    "aggregate_confusion_matrices",
]
