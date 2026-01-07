#!/usr/bin/env python3
# Time-stamp: "2024-11-14 07:51:45 (ywatanabe)"
# File: ./scitex_repo/src/scitex/io/_load_modules/_con.py

from typing import Any

try:
    import mne

    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    mne = None


def _load_con(lpath: str, **kwargs) -> Any:
    if not MNE_AVAILABLE:
        raise ImportError(
            "MNE-Python is not installed. Please install with: pip install mne"
        )

    if not lpath.endswith(".con"):
        raise ValueError("File must have .con extension")
    raw = mne.io.read_raw_fif(lpath, preload=True, **kwargs)
    sfreq = raw.info["sfreq"]
    df = raw.to_data_frame()
    df["samp_rate"] = sfreq
    return df


# EOF
