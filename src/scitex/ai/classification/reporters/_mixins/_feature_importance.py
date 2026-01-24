#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/ai/classification/reporters/_mixins/_feature_importance.py

"""
Feature importance mixin for classification reporter.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from scitex.logging import getLogger

from ._constants import FILENAME_PATTERNS, FOLD_DIR_PREFIX_PATTERN

logger = getLogger(__name__)


class FeatureImportanceMixin:
    """Mixin providing feature importance methods."""

    def save_feature_importance(
        self,
        model,
        feature_names: List[str],
        fold: Optional[int] = None,
    ) -> Dict[str, float]:
        """Calculate and save feature importance for tree-based models."""
        from scitex.ai.metrics import calc_feature_importance

        try:
            importance_dict, importances = calc_feature_importance(model, feature_names)
        except ValueError as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}

        sorted_importances = list(importance_dict.items())

        fold_subdir = (
            FOLD_DIR_PREFIX_PATTERN.format(fold=fold)
            if fold is not None
            else "cv_summary"
        )
        json_filename = FILENAME_PATTERNS["feature_importance_json"].format(fold=fold)
        self.storage.save(dict(sorted_importances), f"{fold_subdir}/{json_filename}")

        jpg_filename = FILENAME_PATTERNS["feature_importance_jpg"].format(fold=fold)
        save_path = self.output_dir / fold_subdir / jpg_filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.plotter.create_feature_importance_plot(
            feature_importance=importances,
            feature_names=feature_names,
            save_path=save_path,
            title=(
                f"Feature Importance (Fold {fold:02d})"
                if fold is not None
                else "Feature Importance (CV Summary)"
            ),
        )

        logger.info(
            "Saved feature importance"
            + (f" for fold {fold}" if fold is not None else "")
        )

        return importance_dict

    def save_feature_importance_summary(
        self,
        all_importances: List[Dict[str, float]],
    ) -> None:
        """Create summary visualization of feature importances across all folds."""
        if not all_importances:
            return

        all_features = set()
        for imp_dict in all_importances:
            all_features.update(imp_dict.keys())

        feature_stats = {}
        for feature in all_features:
            values = [imp_dict.get(feature, 0) for imp_dict in all_importances]
            feature_stats[feature] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(v) for v in values],
            }

        sorted_features = sorted(
            feature_stats.items(), key=lambda x: x[1]["mean"], reverse=True
        )

        n_folds = len(all_importances)
        json_filename = FILENAME_PATTERNS["cv_summary_feature_importance_json"].format(
            n_folds=n_folds
        )
        self.storage.save(dict(sorted_features), f"cv_summary/{json_filename}")

        from scitex.ai.plt import plot_feature_importance_cv_summary

        jpg_filename = FILENAME_PATTERNS["cv_summary_feature_importance_jpg"].format(
            n_folds=n_folds
        )
        save_path = self.output_dir / "cv_summary" / jpg_filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plot_feature_importance_cv_summary(
            all_importances=all_importances,
            spath=save_path,
        )

        logger.info("Saved feature importance summary")


# EOF
