#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 07:23:59 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/sh/test_sh.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = (
    "./src/scitex/sh/test_sh.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

__FILE__ = __file__

import sys

import matplotlib.pyplot as plt

import scitex

CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(
    sys, plt, verbose=False
)

# Test 1: Basic list command
print("Test 1: Basic list command")
result = scitex.sh(["echo", "Hello World"], verbose=True)
print(f"Result: {result}\n")

# Test 2: String command should be rejected
print("Test 2: String command rejection")
try:
    result = scitex.sh("echo 'Hello'")
    print("ERROR: Should have raised TypeError")
except TypeError as ee:
    print(f"Correctly rejected: {ee}\n")

# Test 3: sh_run convenience function
print("Test 3: sh_run convenience function")
result = scitex.sh_run(["ls", "-la"])
print(f"Success: {result['success']}, Exit code: {result['exit_code']}\n")

# Test 4: Error handling
print("Test 4: Error handling")
result = scitex.sh_run(["cat", "/nonexistent/file"], verbose=False)
print(f"Success: {result['success']}")
print(f"Exit code: {result['exit_code']}")
print(f"Stderr: {result['stderr']}\n")

# Test 5: Security - quote function
print("Test 5: Security - quote function")
from scitex.sh import quote

dangerous_input = "file; rm -rf /"
safe_quoted = quote(dangerous_input)
print(f"Original: {dangerous_input}")
print(f"Quoted: {safe_quoted}\n")

# Test 6: Security - null byte rejection
print("Test 6: Security - null byte rejection")
try:
    scitex.sh(["echo", "test\0malicious"])
    print("ERROR: Should have raised ValueError")
except ValueError as ee:
    print(f"Correctly rejected: {ee}\n")

scitex.session.close(CONFIG, verbose=False, notify=False)

# EOF
