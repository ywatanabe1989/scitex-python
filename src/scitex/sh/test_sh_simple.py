#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 07:24:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/sh/test_sh_simple.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = (
    "./src/scitex/sh/test_sh_simple.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

__FILE__ = __file__

import sys

sys.path.insert(0, "/home/ywatanabe/proj/scitex_repo/src")

from scitex.sh import quote, sh, sh_run

# Test 1: Basic list command
print("Test 1: Basic list command")
result = sh(["echo", "Hello World"], verbose=True)
print(f"Result: {result}\n")

# Test 2: String command should be rejected
print("Test 2: String command rejection")
try:
    result = sh_run("echo test")
    print("ERROR: Should have raised TypeError")
except TypeError as ee:
    print(f"Correctly rejected: {ee}\n")

# Test 3: Error handling
print("Test 3: Error handling")
result = sh_run(["cat", "/nonexistent/file"], verbose=False)
print(f"Success: {result['success']}")
print(f"Exit code: {result['exit_code']}")
print(f"Stderr: {result['stderr'][:50]}\n")

# Test 4: Security - quote function
print("Test 4: Security - quote function")
dangerous_input = "file; rm -rf /"
safe_quoted = quote(dangerous_input)
print(f"Original: {dangerous_input}")
print(f"Quoted: {safe_quoted}\n")

# Test 5: Security - null byte rejection
print("Test 5: Security - null byte rejection")
try:
    sh(["echo", "test\0malicious"])
    print("ERROR: Should have raised ValueError")
except ValueError as ee:
    print(f"Correctly rejected: {ee}\n")

print("All tests passed!")

# EOF
