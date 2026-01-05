# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/utils/validation/DOIValidator.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-11-12 13:45:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/scholar/utils/validation/DOIValidator.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/scholar/utils/validation/DOIValidator.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# DOI validation utilities for scitex.scholar.
# 
# Validates DOI accessibility by checking https://doi.org/<DOI> resolution.
# """
# 
# import time
# from typing import Optional, Tuple
# import requests
# 
# from scitex import logging
# 
# logger = logging.getLogger(__name__)
# 
# 
# class DOIValidator:
#     """Validator for DOI accessibility and resolution."""
# 
#     def __init__(
#         self,
#         timeout: int = 10,
#         user_agent: str = "SciTeX-Scholar/2.0 (DOI Validator)",
#         retry_on_timeout: bool = True,
#         max_retries: int = 2,
#     ):
#         """Initialize DOI validator.
# 
#         Args:
#             timeout: Request timeout in seconds
#             user_agent: User agent string for requests
#             retry_on_timeout: Whether to retry on timeout
#             max_retries: Maximum number of retries on timeout
#         """
#         self.timeout = timeout
#         self.user_agent = user_agent
#         self.retry_on_timeout = retry_on_timeout
#         self.max_retries = max_retries
# 
#     def validate_doi(self, doi: str) -> Tuple[bool, str, int, Optional[str]]:
#         """Validate DOI by checking accessibility at https://doi.org/<DOI>.
# 
#         Args:
#             doi: DOI string (e.g., "10.1038/s41598-023-12345-6")
# 
#         Returns:
#             Tuple of (is_valid, message, status_code, resolved_url)
#             - is_valid: True if DOI resolves successfully
#             - message: Human-readable status message
#             - status_code: HTTP status code (0 if request failed)
#             - resolved_url: Final URL after redirects (None if invalid)
#         """
#         if not doi:
#             return False, "Empty DOI", 0, None
# 
#         # Clean DOI (remove URL prefix if present)
#         doi_clean = self._clean_doi(doi)
# 
#         # Validate DOI format
#         if not self._is_valid_doi_format(doi_clean):
#             return False, f"Invalid DOI format: {doi_clean}", 0, None
# 
#         url = f"https://doi.org/{doi_clean}"
# 
#         # Try validation with retries
#         for attempt in range(1, self.max_retries + 1):
#             try:
#                 # Use HEAD request first (faster, less bandwidth)
#                 response = requests.head(
#                     url,
#                     timeout=self.timeout,
#                     allow_redirects=True,
#                     headers={"User-Agent": self.user_agent},
#                 )
# 
#                 # DOI service returns 404 for invalid DOIs
#                 if response.status_code == 404:
#                     return False, "DOI Not Found (404)", 404, None
# 
#                 # Some publishers don't support HEAD, try GET
#                 if response.status_code == 405:  # Method Not Allowed
#                     response = requests.get(
#                         url,
#                         timeout=self.timeout,
#                         allow_redirects=True,
#                         headers={"User-Agent": self.user_agent},
#                     )
# 
#                 # Success codes (200-399, including redirects)
#                 if 200 <= response.status_code < 400:
#                     resolved_url = response.url
#                     if resolved_url != url:
#                         return (
#                             True,
#                             "Valid (resolved)",
#                             response.status_code,
#                             resolved_url,
#                         )
#                     return True, "Valid", response.status_code, resolved_url
# 
#                 # Other error codes
#                 return False, f"HTTP {response.status_code}", response.status_code, None
# 
#             except requests.exceptions.Timeout:
#                 if self.retry_on_timeout and attempt < self.max_retries:
#                     logger.warning(
#                         f"DOI validation timeout (attempt {attempt}/{self.max_retries}), retrying..."
#                     )
#                     time.sleep(1)  # Brief delay before retry
#                     continue
#                 return False, "Timeout", 0, None
# 
#             except requests.exceptions.ConnectionError:
#                 return False, "Connection Error", 0, None
# 
#             except Exception as e:
#                 logger.debug(f"DOI validation error for {doi}: {e}")
#                 return False, f"Error: {str(e)[:50]}", 0, None
# 
#         return False, f"Failed after {self.max_retries} retries", 0, None
# 
#     def _clean_doi(self, doi: str) -> str:
#         """Clean DOI string by removing URL prefixes.
# 
#         Args:
#             doi: Raw DOI string
# 
#         Returns:
#             Cleaned DOI string
#         """
#         doi = doi.strip()
# 
#         # Remove common prefixes
#         prefixes = [
#             "https://doi.org/",
#             "http://doi.org/",
#             "https://dx.doi.org/",
#             "http://dx.doi.org/",
#             "doi:",
#             "DOI:",
#         ]
# 
#         for prefix in prefixes:
#             if doi.startswith(prefix):
#                 doi = doi[len(prefix) :]
# 
#         return doi.strip()
# 
#     def _is_valid_doi_format(self, doi: str) -> bool:
#         """Check if DOI string has valid format.
# 
#         DOI format: 10.XXXX/suffix
#         - Must start with "10."
#         - Must contain at least one "/"
#         - Prefix must be numeric after "10."
# 
#         Args:
#             doi: Cleaned DOI string
# 
#         Returns:
#             True if DOI format is valid
#         """
#         if not doi.startswith("10."):
#             return False
# 
#         if "/" not in doi:
#             return False
# 
#         # Split into prefix and suffix
#         parts = doi.split("/", 1)
#         if len(parts) != 2:
#             return False
# 
#         prefix, suffix = parts
# 
#         # Validate prefix format (10.XXXX where XXXX is numeric)
#         prefix_parts = prefix.split(".")
#         if len(prefix_parts) < 2:
#             return False
# 
#         # Check that prefix is "10.XXXX" format
#         if prefix_parts[0] != "10":
#             return False
# 
#         # Check that registrant code is numeric
#         try:
#             int(prefix_parts[1])
#         except ValueError:
#             return False
# 
#         # Suffix must not be empty
#         if not suffix.strip():
#             return False
# 
#         return True
# 
#     def validate_batch(
#         self,
#         dois: list[str],
#         delay: float = 0.5,
#         progress_callback: Optional[callable] = None,
#     ) -> dict:
#         """Validate multiple DOIs with rate limiting.
# 
#         Args:
#             dois: List of DOI strings to validate
#             delay: Delay between requests in seconds (be polite to DOI service)
#             progress_callback: Optional callback function(current, total, doi, is_valid)
# 
#         Returns:
#             Dictionary with validation results:
#             {
#                 'total': int,
#                 'valid': int,
#                 'invalid': int,
#                 'results': [{'doi': str, 'is_valid': bool, 'message': str, ...}, ...]
#             }
#         """
#         results = {"total": len(dois), "valid": 0, "invalid": 0, "results": []}
# 
#         for i, doi in enumerate(dois, 1):
#             is_valid, message, status_code, resolved_url = self.validate_doi(doi)
# 
#             result_entry = {
#                 "doi": doi,
#                 "is_valid": is_valid,
#                 "message": message,
#                 "status_code": status_code,
#                 "resolved_url": resolved_url,
#             }
# 
#             results["results"].append(result_entry)
# 
#             if is_valid:
#                 results["valid"] += 1
#             else:
#                 results["invalid"] += 1
# 
#             # Call progress callback if provided
#             if progress_callback:
#                 progress_callback(i, len(dois), doi, is_valid)
# 
#             # Rate limiting (except for last item)
#             if i < len(dois):
#                 time.sleep(delay)
# 
#         return results
# 
# 
# if __name__ == "__main__":
#     # Demo usage
#     validator = DOIValidator()
# 
#     # Test cases
#     test_dois = [
#         "10.1038/s41598-023-12345-6",  # Invalid (example)
#         "10.1186/1751-0473-8-7",  # Valid (Git reproducibility paper)
#         "10.1371/journal.pcbi.1007128",  # Valid (Manubot paper)
#         "",  # Empty
#         "invalid-doi",  # Invalid format
#     ]
# 
#     print("=" * 80)
#     print("DOI Validator Demo")
#     print("=" * 80)
# 
#     for doi in test_dois:
#         print(f"\nTesting: {doi or '(empty)'}")
#         is_valid, message, status_code, resolved_url = validator.validate_doi(doi)
# 
#         print(f"  Valid: {is_valid}")
#         print(f"  Message: {message}")
#         print(f"  Status Code: {status_code}")
#         if resolved_url:
#             print(f"  Resolved URL: {resolved_url[:80]}...")
# 
#     print("\n" + "=" * 80)
#     print("Demo complete")
#     print("=" * 80)
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/utils/validation/DOIValidator.py
# --------------------------------------------------------------------------------
