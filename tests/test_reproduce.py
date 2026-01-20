#!/usr/bin/env python3
"""Test figrecipe reproduce functionality."""
import sys
sys.path.insert(0, "/home/ywatanabe/proj/figrecipe/src")

import figrecipe as fr
import matplotlib.pyplot as plt

# Test 2: Reproduce from recipe
print("=" * 60)
print("TEST 2: fr.reproduce() from recipe")
print("=" * 60)

fig = fr.reproduce("test_basic.yaml")
fr.save(fig, "test_reproduced.png")
plt.close(fig.fig)

import os
print("\nFiles after reproduce:")
for f in sorted(os.listdir(".")):
    if "reproduced" in f or "basic" in f:
        size = os.path.getsize(f) if not os.path.isdir(f) else 0
        print(f"  {f} ({size} bytes)")
