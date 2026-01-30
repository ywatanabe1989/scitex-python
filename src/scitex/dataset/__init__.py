#!/usr/bin/env python3
"""
SciTeX Dataset - Thin wrapper delegating to scitex-dataset package.

This module provides access to scientific dataset discovery functionality
by delegating to the scitex-dataset package.

Usage:
    >>> from scitex import dataset
    >>> datasets = dataset.fetch_all_datasets(max_datasets=10)
    >>> results = dataset.search_datasets(datasets, modality="eeg")

    >>> # Or use the database for fast local search
    >>> dataset.db.build()
    >>> results = dataset.db.search("alzheimer EEG")
"""

try:
    from scitex_dataset import (
        OPENNEURO_API,
        __version__,
        database,
        fetch_all_datasets,
        fetch_datasets,
        format_dataset,
        general,
        neuroscience,
        search_datasets,
        sort_datasets,
    )

    # Alias for consistency
    db = database

    __all__ = [
        "__version__",
        # Domains
        "neuroscience",
        "general",
        # Database
        "database",
        "db",
        # Convenience (OpenNeuro)
        "fetch_datasets",
        "fetch_all_datasets",
        "format_dataset",
        "OPENNEURO_API",
        # Search
        "search_datasets",
        "sort_datasets",
    ]

except ImportError as e:
    import warnings

    warnings.warn(
        f"scitex-dataset package not installed. "
        f"Install with: pip install scitex-dataset\n{e}",
        ImportWarning,
        stacklevel=2,
    )

    # Provide stub for error messages
    def _not_installed(*args, **kwargs):
        raise ImportError(
            "scitex-dataset package not installed. "
            "Install with: pip install scitex-dataset"
        )

    fetch_datasets = _not_installed
    fetch_all_datasets = _not_installed
    format_dataset = _not_installed
    search_datasets = _not_installed
    sort_datasets = _not_installed
    database = None
    db = None
    neuroscience = None
    general = None
    OPENNEURO_API = None
    __version__ = "0.0.0"

    __all__ = []


# EOF
