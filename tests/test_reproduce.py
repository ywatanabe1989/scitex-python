#!/usr/bin/env python3
"""Test figrecipe reproduce functionality."""

import sys

sys.path.insert(0, "/home/ywatanabe/proj/figrecipe/src")

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

# Skip if test_basic.yaml doesn't exist
if not Path("test_basic.yaml").exists():
    pytest.skip("test_basic.yaml not found", allow_module_level=True)

import figrecipe as fr  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Test 2: Reproduce from recipe
print("=" * 60)
print("TEST 2: fr.reproduce() from recipe")
print("=" * 60)

fig = fr.reproduce("test_basic.yaml")
fr.save(fig, "test_reproduced.png")
plt.close(fig.fig)

import os  # noqa: E402

print("\nFiles after reproduce:")
for f in sorted(os.listdir(".")):
    if "reproduced" in f or "basic" in f:
        size = os.path.getsize(f) if not os.path.isdir(f) else 0
        print(f"  {f} ({size} bytes)")
