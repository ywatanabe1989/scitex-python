#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/git/result.py

"""
Result dataclasses for git operations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GitResult:
    """
    Result of a git operation.

    Attributes
    ----------
    success : bool
        Whether operation succeeded
    error : Optional[str]
        Error message if operation failed
    stdout : Optional[str]
        Standard output from git command
    stderr : Optional[str]
        Standard error from git command
    """

    success: bool
    error: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


@dataclass
class CommitResult(GitResult):
    """
    Result of a git commit operation.

    Attributes
    ----------
    commit_hash : Optional[str]
        Hash of created commit if successful
    """

    commit_hash: Optional[str] = None


@dataclass
class BranchResult(GitResult):
    """
    Result of a git branch operation.

    Attributes
    ----------
    branch_name : Optional[str]
        Name of created or renamed branch if successful
    """

    branch_name: Optional[str] = None


__all__ = [
    "GitResult",
    "CommitResult",
    "BranchResult",
]

# EOF
