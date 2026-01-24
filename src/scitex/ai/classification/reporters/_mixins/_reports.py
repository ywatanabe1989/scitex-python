#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/ai/classification/reporters/_mixins/_reports.py

"""
Reports generation mixin for classification reporter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from scitex.logging import getLogger

from ._constants import FOLD_DIR_PREFIX_PATTERN

logger = getLogger(__name__)


class ReportsMixin:
    """Mixin providing report generation methods."""

    def generate_reports(self) -> Dict[str, Path]:
        """Generate comprehensive reports in multiple formats."""
        from ..reporter_utils.reporting import generate_org_report

        results = {
            "config": {
                "n_folds": len(self.fold_metrics),
                "output_dir": str(self.output_dir),
            },
            "session_config": self.session_config,
            "summary": {},
            "folds": [],
            "plots": {},
        }

        summary = self.get_summary()

        if "metrics_summary" in summary:
            results["summary"] = summary["metrics_summary"]

        if "feature-importance" in summary:
            results["summary"]["feature-importance"] = summary["feature-importance"]

        for fold, fold_data in self.fold_metrics.items():
            fold_result = {"fold_id": fold}
            fold_result.update(fold_data)

            try:
                calling_file_dir = Path(__file__).parent.parent / "reporter_utils"
                storage_out_path = (
                    calling_file_dir
                    / "storage_out"
                    / self.output_dir
                    / FOLD_DIR_PREFIX_PATTERN.format(fold=fold)
                    / "features.json"
                )

                regular_path = (
                    self.output_dir
                    / FOLD_DIR_PREFIX_PATTERN.format(fold=fold)
                    / "features.json"
                )

                features_json = None
                if storage_out_path.exists():
                    features_json = storage_out_path
                elif regular_path.exists():
                    features_json = regular_path

                if features_json:
                    with open(features_json) as f:
                        features_data = json.load(f)
                        for key in [
                            "n_train",
                            "n_test",
                            "n_train_seizure",
                            "n_train_interictal",
                            "n_test_seizure",
                            "n_test_interictal",
                        ]:
                            if key in features_data:
                                fold_result[key] = int(features_data[key])
            except Exception:
                pass

            results["folds"].append(fold_result)

        cv_summary_dir = self.output_dir / "cv_summary"
        if cv_summary_dir.exists():
            for plot_file in cv_summary_dir.glob("*.jpg"):
                plot_key = f"cv_summary_{plot_file.stem}"
                results["plots"][plot_key] = str(plot_file.relative_to(self.output_dir))

        for fold_dir in sorted(self.output_dir.glob("fold_*")):
            fold_num = fold_dir.name.replace("fold_", "")
            for plot_file in fold_dir.glob("*.jpg"):
                plot_key = f"fold_{fold_num}_{plot_file.stem}"
                results["plots"][plot_key] = str(plot_file.relative_to(self.output_dir))

        reports_dir = self._create_subdir_if_needed("reports")
        generated_files = {}

        org_path = reports_dir / "classification_report.org"
        generate_org_report(results, org_path, include_plots=True, convert_formats=True)
        generated_files["org"] = org_path
        logger.info(f"Generated org-mode report: {org_path}")

        try:
            import os
            import shutil
            import subprocess

            if shutil.which("pdflatex"):
                original_dir = Path.cwd()
                try:
                    os.chdir(reports_dir)

                    for _ in range(2):
                        result = subprocess.run(
                            [
                                "pdflatex",
                                "-interaction=nonstopmode",
                                "classification_report.tex",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )

                    pdf_path = reports_dir / "classification_report.pdf"
                    if pdf_path.exists():
                        generated_files["pdf"] = pdf_path
                        logger.info(f"Generated PDF report: {pdf_path}")

                        for ext in [".aux", ".log", ".out", ".toc"]:
                            aux_file = reports_dir / f"classification_report{ext}"
                            if aux_file.exists():
                                aux_file.unlink()
                finally:
                    os.chdir(original_dir)
            else:
                logger.warning("pdflatex not found. Skipping PDF generation.")
        except Exception as e:
            logger.warning(f"Could not generate PDF report: {e}")

        return generated_files


# EOF
