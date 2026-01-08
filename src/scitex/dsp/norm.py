#!/usr/bin/env python3
# Time-stamp: "2024-04-05 12:15:42 (ywatanabe)"

try:
    import torch as _torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    _torch = None

from scitex.decorators import signal_fn as _signal_fn


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. Please install with: pip install torch"
        )


@_signal_fn
def z(x, dim=-1):
    _check_torch()
    return (x - x.mean(dim=dim, keepdim=True)) / x.std(dim=dim, keepdim=True)


@_signal_fn
def minmax(x, amp=1.0, dim=-1, fn="mean"):
    _check_torch()
    MM = x.max(dim=dim, keepdims=True)[0].abs()
    mm = x.min(dim=dim, keepdims=True)[0].abs()
    return amp * x / _torch.maximum(MM, mm)
