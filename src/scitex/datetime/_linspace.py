#!/usr/bin/env python3
# Timestamp: "2026-01-05 14:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/datetime/_linspace.py

"""
Datetime linspace utility for creating evenly spaced datetime arrays.
"""

import datetime
from datetime import timedelta
from typing import Optional

import numpy as np


def linspace(
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
    n_samples: Optional[int] = None,
    sampling_rate: Optional[float] = None,
) -> np.ndarray:
    """
    Create a linearly spaced array between two datetime objects.

    Parameters
    ----------
    start_dt : datetime.datetime
        Starting datetime object
    end_dt : datetime.datetime
        Ending datetime object
    n_samples : int, optional
        Number of samples to create (mutually exclusive with sampling_rate)
    sampling_rate : float, optional
        Sampling rate in Hz (mutually exclusive with n_samples)

    Returns
    -------
    np.ndarray
        Array of datetime objects evenly spaced between start_dt and end_dt

    Raises
    ------
    TypeError
        If start_dt or end_dt is not a datetime object
    ValueError
        If start_dt >= end_dt, or if both/neither n_samples and sampling_rate provided

    Examples
    --------
    >>> import datetime
    >>> start = datetime.datetime(2023, 1, 1, 0, 0, 0)
    >>> end = datetime.datetime(2023, 1, 1, 0, 0, 10)
    >>> result = linspace(start, end, n_samples=11)
    >>> len(result)
    11
    """
    # Type checking
    if not isinstance(start_dt, datetime.datetime):
        raise TypeError(f"start_dt must be a datetime object, got {type(start_dt)}")

    if not isinstance(end_dt, datetime.datetime):
        raise TypeError(f"end_dt must be a datetime object, got {type(end_dt)}")

    if n_samples is not None and not isinstance(n_samples, (int, float)):
        raise TypeError(f"n_samples must be a number, got {type(n_samples)}")

    if sampling_rate is not None and not isinstance(sampling_rate, (int, float)):
        raise TypeError(f"sampling_rate must be a number, got {type(sampling_rate)}")

    if start_dt >= end_dt:
        raise ValueError("start_dt must be earlier than end_dt")

    duration_seconds = (end_dt - start_dt).total_seconds()

    if n_samples is not None and sampling_rate is not None:
        raise ValueError("Provide either n_samples or sampling_rate, not both")

    if n_samples is None and sampling_rate is None:
        raise ValueError("Either n_samples or sampling_rate must be provided")

    if sampling_rate is not None:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        n_samples = int(duration_seconds * sampling_rate) + 1
    else:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

    # Create linear space in seconds
    seconds_array = np.linspace(0, duration_seconds, n_samples)

    # Convert to datetime objects
    datetime_array = np.array(
        [start_dt + timedelta(seconds=float(sec)) for sec in seconds_array]
    )

    return datetime_array


# EOF
