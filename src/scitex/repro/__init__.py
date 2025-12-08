#!/usr/bin/env python3
"""
SciTeX Repro Module - Reproducibility utilities.

Provides tools for reproducible scientific computing:
- Random state management (RandomStateManager)
- ID generation (gen_ID)
- Timestamp generation (gen_timestamp)
- Array hashing (hash_array)
"""

# ID and timestamp utilities
from ._gen_ID import gen_ID, gen_id
from ._gen_timestamp import gen_timestamp, timestamp

# Hash utilities
from ._hash_array import hash_array

# Random state management (moved from scitex.rng)
from ._RandomStateManager import RandomStateManager, get, reset


# Legacy function for backward compatibility
def fix_seeds(
    seed=42,
    os=True,
    random=True,
    np=True,
    torch=True,
    tf=False,
    jax=False,
    verbose=False,
    **kwargs,
):
    """
    Deprecated: Use stx.repro.RandomStateManager instead.

    This function maintains backward compatibility with the old fix_seeds API.
    """
    import warnings

    warnings.warn(
        "fix_seeds is deprecated. Use stx.repro.RandomStateManager instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Create RandomStateManager with seed and verbose
    # It automatically handles all available modules
    return RandomStateManager(seed=seed, verbose=verbose)


__all__ = [
    # ID and timestamp utilities
    "gen_ID",
    "gen_id",
    "gen_timestamp",
    "timestamp",
    # Hash utilities
    "hash_array",
    # Random state management
    "RandomStateManager",
    "get",
    "reset",
    # Legacy (deprecated)
    "fix_seeds",
]
