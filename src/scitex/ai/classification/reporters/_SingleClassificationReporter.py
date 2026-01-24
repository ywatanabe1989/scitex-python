#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/ai/classification/reporters/_SingleClassificationReporter.py

"""
Improved Single Classification Reporter with unified API.

This module provides a comprehensive classification reporter that:
- Uses unified API interface
- Supports lazy directory creation
- Provides numerical precision control
- Creates visualizations with proper error handling
- Maintains consistent parameter naming

The main class inherits from multiple mixins for modular functionality:
- MetricsMixin: Metrics calculation and aggregation
- StorageMixin: File storage and organization
- PlottingMixin: Visualization generation
- FeatureImportanceMixin: Feature importance analysis
- CVSummaryMixin: Cross-validation summary generation
- ReportsMixin: Multi-format report generation
"""

from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Union

from scitex.logging import getLogger

from ._BaseClassificationReporter import BaseClassificationReporter, ReporterConfig
from ._mixins import (
    CVSummaryMixin,
    FeatureImportanceMixin,
    MetricsMixin,
    PlottingMixin,
    ReportsMixin,
    StorageMixin,
)
from .reporter_utils._Plotter import Plotter
from .reporter_utils.storage import MetricStorage

logger = getLogger(__name__)


class SingleTaskClassificationReporter(
    MetricsMixin,
    StorageMixin,
    PlottingMixin,
    FeatureImportanceMixin,
    CVSummaryMixin,
    ReportsMixin,
    BaseClassificationReporter,
):
    """
    Improved single-task classification reporter with unified API.

    Key improvements:
    - Inherits from BaseClassificationReporter for consistent API
    - Lazy directory creation (no empty folders)
    - Numerical precision control
    - Graceful plotting with proper error handling
    - Consistent parameter names across all methods

    Features:
    - Comprehensive metrics calculation (balanced accuracy, MCC, ROC-AUC, PR-AUC, etc.)
    - Automated visualization generation:
      * Confusion matrices
      * ROC and Precision-Recall curves
      * Feature importance plots
      * CV aggregation plots with faded fold lines
      * Comprehensive metrics dashboard
    - Multi-format report generation (Org, Markdown, LaTeX, HTML, DOCX, PDF)
    - Cross-validation support with automatic fold aggregation

    Parameters
    ----------
    output_dir : Union[str, Path]
        Base directory for outputs. If None, creates timestamped directory.
    config : ReporterConfig, optional
        Configuration object for advanced settings
    verbose : bool, default True
        Print initialization message
    **kwargs
        Additional arguments passed to base class

    Examples
    --------
    >>> # Basic usage
    >>> reporter = SingleTaskClassificationReporter("./results")
    >>> metrics = reporter.calculate_metrics(y_true, y_pred, y_proba, labels=['A', 'B'])
    >>> reporter.save_summary()

    >>> # Cross-validation with automatic CV aggregation plots
    >>> for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    ...     metrics = reporter.calculate_metrics(
    ...         y_test, y_pred, y_proba, fold=fold
    ...     )
    >>> reporter.save_summary()  # Automatically creates CV aggregation visualizations

    >>> # Feature importance visualization
    >>> reporter.plotter.create_feature_importance_plot(
    ...     feature_importance=importances,
    ...     feature_names=feature_names,
    ...     save_path=output_dir / "feature_importance.png"
    ... )
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        config: Optional[ReporterConfig] = None,
        verbose: bool = True,
        **kwargs,
    ):
        if config is None:
            config = ReporterConfig()

        super().__init__(
            output_dir=output_dir,
            precision=config.precision,
            **kwargs,
        )

        self.config = config
        self.session_config = None
        self.storage = MetricStorage(self.output_dir, precision=self.precision)
        self.plotter = Plotter(enable_plotting=True)

        self.fold_metrics: Dict[int, Dict[str, Any]] = {}
        self.all_predictions: List[Dict[str, Any]] = []

        if verbose:
            logger.info(
                f"{self.__class__.__name__} initialized with output directory: {self.output_dir}"
            )

    def set_session_config(self, config: Any) -> None:
        """
        Set the SciTeX session CONFIG object for inclusion in reports.

        Parameters
        ----------
        config : Any
            The SciTeX session CONFIG object
        """
        self.session_config = config

    def save_summary(
        self, filename: str = "cv_summary/summary.json", verbose: bool = True
    ) -> Path:
        """
        Save summary to file, create CV summary visualizations, and generate reports.

        Parameters
        ----------
        filename : str, default "cv_summary/summary.json"
            Filename for summary (now in cv_summary directory)
        verbose : bool, default True
            Print summary to console

        Returns
        -------
        Path
            Path to saved summary file
        """
        summary = self.get_summary()

        try:
            possible_paths = [
                self.output_dir.parent / "CONFIGS" / "CONFIG.yaml",
                self.output_dir.parent.parent / "CONFIGS" / "CONFIG.yaml",
                self.output_dir / "CONFIGS" / "CONFIG.yaml",
            ]

            config_path = None
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break

            if config_path and config_path.exists():
                import yaml

                with open(config_path) as config_file:
                    config_data = yaml.safe_load(config_file)
                    summary["experiment_configuration"] = config_data
        except Exception as e:
            logger.warning(f"Could not load CONFIG.yaml: {e}")

        self._save_cv_summary_metrics(summary)
        self.save_cv_summary_confusion_matrix(summary)
        self.create_cv_summary_curves(summary)
        self.create_cv_aggregation_visualizations(
            show_individual_folds=True, fold_alpha=0.15
        )
        self._save_cv_summary_classification_report(summary)
        self.generate_reports()

        cv_summary_dir = self._create_subdir_if_needed("cv_summary")
        cv_summary_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print()
            logger.info("Summary:")
            pprint(summary)

        return self.storage.save(summary, "cv_summary/summary.json")

    def __repr__(self) -> str:
        fold_count = len(self.fold_metrics)
        return (
            f"SingleTaskClassificationReporter("
            f"folds={fold_count}, "
            f"output_dir='{self.output_dir}')"
        )


# EOF
