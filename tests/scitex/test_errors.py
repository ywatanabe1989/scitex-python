# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/errors.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-21"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/errors.py
# 
# """Backwards-compatible re-export of errors from scitex.logging.
# 
# DEPRECATED: Import from scitex.logging instead.
# 
#     # Old (deprecated)
#     from scitex.errors import SciTeXError, SaveError
# 
#     # New (recommended)
#     from scitex.logging import SciTeXError, SaveError
# """
# 
# from __future__ import annotations
# import warnings
# 
# # Issue deprecation warning on import
# # stacklevel=1 so warning appears from scitex.errors (matches scitex.* filter)
# warnings.warn(
#     "scitex.errors is deprecated. Import from scitex.logging instead. "
#     "Example: from scitex.logging import SciTeXError, UnitWarning",
#     DeprecationWarning,
#     stacklevel=1,
# )
# 
# # Re-export everything from scitex.logging for backwards compatibility
# from scitex.logging import (
#     # Warnings
#     SciTeXWarning,
#     UnitWarning,
#     StyleWarning,
#     SciTeXDeprecationWarning,
#     PerformanceWarning,
#     DataLossWarning,
#     warn_deprecated,
#     warn_performance,
#     warn_data_loss,
#     # Errors
#     SciTeXError,
#     ConfigurationError,
#     ConfigFileNotFoundError,
#     ConfigKeyError,
#     IOError,
#     FileFormatError,
#     SaveError,
#     LoadError,
#     ScholarError,
#     SearchError,
#     EnrichmentError,
#     PDFDownloadError,
#     DOIResolutionError,
#     PDFExtractionError,
#     BibTeXEnrichmentError,
#     TranslatorError,
#     AuthenticationError,
#     PlottingError,
#     FigureNotFoundError,
#     AxisError,
#     DataError,
#     ShapeError,
#     DTypeError,
#     PathError,
#     InvalidPathError,
#     PathNotFoundError,
#     TemplateError,
#     TemplateViolationError,
#     NNError,
#     ModelError,
#     StatsError,
#     TestError,
#     # Validation helpers
#     check_path,
#     check_file_exists,
#     check_shape_compatibility,
# )
# 
# __all__ = [
#     # Warnings
#     "SciTeXWarning",
#     "UnitWarning",
#     "StyleWarning",
#     "SciTeXDeprecationWarning",
#     "PerformanceWarning",
#     "DataLossWarning",
#     "warn_deprecated",
#     "warn_performance",
#     "warn_data_loss",
#     # Errors
#     "SciTeXError",
#     "ConfigurationError",
#     "ConfigFileNotFoundError",
#     "ConfigKeyError",
#     "IOError",
#     "FileFormatError",
#     "SaveError",
#     "LoadError",
#     "ScholarError",
#     "SearchError",
#     "EnrichmentError",
#     "PDFDownloadError",
#     "DOIResolutionError",
#     "PDFExtractionError",
#     "BibTeXEnrichmentError",
#     "TranslatorError",
#     "AuthenticationError",
#     "PlottingError",
#     "FigureNotFoundError",
#     "AxisError",
#     "DataError",
#     "ShapeError",
#     "DTypeError",
#     "PathError",
#     "InvalidPathError",
#     "PathNotFoundError",
#     "TemplateError",
#     "TemplateViolationError",
#     "NNError",
#     "ModelError",
#     "StatsError",
#     "TestError",
#     # Validation helpers
#     "check_path",
#     "check_file_exists",
#     "check_shape_compatibility",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/errors.py
# --------------------------------------------------------------------------------
