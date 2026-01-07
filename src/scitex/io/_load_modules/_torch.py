#!/usr/bin/env python3
# Time-stamp: "2024-11-14 07:41:34 (ywatanabe)"
# File: ./scitex_repo/src/scitex/io/_load_modules/_torch.py

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _load_torch(lpath, **kwargs):
    """Load PyTorch model/checkpoint file."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. Please install with: pip install torch"
        )

    if not lpath.endswith((".pth", ".pt")):
        raise ValueError("File must have .pth or .pt extension")
    return torch.load(lpath, **kwargs)


# EOF
