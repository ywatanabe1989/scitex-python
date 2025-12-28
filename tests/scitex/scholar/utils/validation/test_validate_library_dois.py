# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/utils/validation/validate_library_dois.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-11-12 14:00:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/scholar/utils/validation/validate_library_dois.py
# # ----------------------------------------
# """
# Validate all DOIs in a scitex.scholar library.
# 
# Usage:
#     python -m scitex.scholar.utils.validation.validate_library_dois
#     python -m scitex.scholar.utils.validation.validate_library_dois --fix
#     python -m scitex.scholar.utils.validation.validate_library_dois --project my-project
# """
# 
# import json
# import time
# from pathlib import Path
# from typing import Dict, Optional
# 
# from scitex import logging
# from scitex.scholar.config import ScholarConfig
# from .DOIValidator import DOIValidator
# 
# logger = logging.getLogger(__name__)
# 
# 
# def validate_library_dois(
#     library_path: Optional[Path] = None,
#     project: Optional[str] = None,
#     delay_between_requests: float = 0.5,
#     fix_invalid: bool = False,
#     output_file: Optional[Path] = None,
# ) -> Dict:
#     """Validate all DOIs in the scholar library.
# 
#     Args:
#         library_path: Path to MASTER directory (default: from ScholarConfig)
#         project: Specific project to validate (default: validate all in MASTER)
#         delay_between_requests: Delay in seconds between DOI checks (be polite)
#         fix_invalid: If True, remove invalid DOIs from metadata.json files
#         output_file: Optional path to save validation report as JSON
# 
#     Returns:
#         Dictionary with validation results
#     """
#     # Get library path
#     if library_path is None:
#         config = ScholarConfig()
#         library_path = config.path_manager.get_library_master_dir()
# 
#     logger.info(f"Validating DOIs in: {library_path}")
#     logger.info(f"Delay between requests: {delay_between_requests}s")
#     if fix_invalid:
#         logger.warning("Fix mode enabled - invalid DOIs will be removed!")
# 
#     # Initialize validator
#     validator = DOIValidator()
# 
#     results = {
#         "total_papers": 0,
#         "papers_with_doi": 0,
#         "valid_dois": 0,
#         "invalid_dois": 0,
#         "empty_dois": 0,
#         "invalid_details": [],
#     }
# 
#     # Find all metadata.json files
#     metadata_files = sorted(library_path.glob("*/metadata.json"))
#     results["total_papers"] = len(metadata_files)
# 
#     logger.info(f"Found {len(metadata_files)} papers to validate")
# 
#     print("\n" + "=" * 80)
#     print("DOI VALIDATION REPORT")
#     print("=" * 80)
#     print(f"Library: {library_path}")
#     print(f"Papers: {len(metadata_files)}")
#     print("=" * 80)
# 
#     for i, metadata_file in enumerate(metadata_files, 1):
#         paper_id = metadata_file.parent.name
# 
#         # Load metadata
#         with open(metadata_file, "r") as f:
#             metadata = json.load(f)
# 
#         doi = metadata.get("metadata", {}).get("id", {}).get("doi", "")
#         title = (
#             metadata.get("metadata", {}).get("basic", {}).get("title", "No title")[:60]
#         )
# 
#         if not doi:
#             results["empty_dois"] += 1
#             continue
# 
#         results["papers_with_doi"] += 1
# 
#         print(f"\n[{i}/{len(metadata_files)}] {title}...")
#         print(f"  Paper ID: {paper_id}")
#         print(f"  DOI: {doi}")
# 
#         # Check DOI accessibility
#         is_valid, message, status_code, resolved_url = validator.validate_doi(doi)
# 
#         if is_valid:
#             results["valid_dois"] += 1
#             logger.success(f"  ✓ {message}")
#             if resolved_url:
#                 print(f"  Resolved to: {resolved_url[:70]}...")
#         else:
#             results["invalid_dois"] += 1
#             logger.error(f"  ✗ {message}")
# 
#             results["invalid_details"].append(
#                 {
#                     "paper_id": paper_id,
#                     "title": title,
#                     "doi": doi,
#                     "reason": message,
#                     "status_code": status_code,
#                     "metadata_file": str(metadata_file),
#                 }
#             )
# 
#             # Fix invalid DOI if requested
#             if fix_invalid:
#                 logger.warning(f"  Removing invalid DOI from {metadata_file}")
#                 metadata["metadata"]["id"]["doi"] = ""
#                 metadata["metadata"]["id"]["doi_engines"] = []
#                 metadata["metadata"]["url"]["doi"] = None
#                 metadata["metadata"]["url"]["doi_engines"] = []
# 
#                 with open(metadata_file, "w") as f:
#                     json.dump(metadata, f, indent=2, ensure_ascii=False)
# 
#                 logger.info(f"  ✓ Removed invalid DOI from metadata")
# 
#         # Be polite to DOI service
#         if i < len(metadata_files):
#             time.sleep(delay_between_requests)
# 
#     # Summary
#     print("\n" + "=" * 80)
#     print("VALIDATION SUMMARY")
#     print("=" * 80)
#     print(f"Total papers: {results['total_papers']}")
#     print(f"Papers with DOI: {results['papers_with_doi']}")
#     if results["papers_with_doi"] > 0:
#         print(
#             f"  ✓ Valid DOIs: {results['valid_dois']} ({results['valid_dois'] * 100 / results['papers_with_doi']:.1f}%)"
#         )
#         print(
#             f"  ✗ Invalid DOIs: {results['invalid_dois']} ({results['invalid_dois'] * 100 / results['papers_with_doi']:.1f}%)"
#         )
#     print(f"Papers without DOI: {results['empty_dois']}")
# 
#     # Invalid DOI details
#     if results["invalid_details"]:
#         print("\n" + "=" * 80)
#         print(f"INVALID DOI DETAILS ({len(results['invalid_details'])} papers)")
#         print("=" * 80)
#         for item in results["invalid_details"]:
#             print(f"\nTitle: {item['title']}")
#             print(f"  Paper ID: {item['paper_id']}")
#             print(f"  DOI: {item['doi']}")
#             print(f"  Reason: {item['reason']}")
#             print(f"  File: {item['metadata_file']}")
# 
#     # Save results to file if requested
#     if output_file:
#         with open(output_file, "w") as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
#         logger.success(f"\n✓ Validation results saved to: {output_file}")
# 
#     return results
# 
# 
# if __name__ == "__main__":
#     import argparse
# 
#     parser = argparse.ArgumentParser(
#         description="Validate DOIs in scitex.scholar library"
#     )
#     parser.add_argument(
#         "--library-path",
#         type=Path,
#         help="Path to MASTER library directory (default: from config)",
#     )
#     parser.add_argument("--project", type=str, help="Validate specific project only")
#     parser.add_argument(
#         "--delay",
#         type=float,
#         default=0.5,
#         help="Delay between DOI checks in seconds (default: 0.5)",
#     )
#     parser.add_argument(
#         "--fix",
#         action="store_true",
#         help="Remove invalid DOIs from metadata.json files",
#     )
#     parser.add_argument(
#         "--output", type=Path, help="Save validation report to JSON file"
#     )
# 
#     args = parser.parse_args()
# 
#     # Validate DOIs
#     results = validate_library_dois(
#         library_path=args.library_path,
#         project=args.project,
#         delay_between_requests=args.delay,
#         fix_invalid=args.fix,
#         output_file=args.output,
#     )
# 
#     logger.success("\n✓ DOI validation complete!")
# 
#     # Exit with error code if invalid DOIs found
#     if results["invalid_dois"] > 0:
#         exit(1)
#     else:
#         exit(0)
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/utils/validation/validate_library_dois.py
# --------------------------------------------------------------------------------
