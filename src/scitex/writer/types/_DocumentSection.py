#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-28 16:41:11 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/types/document_section.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/writer/types/document_section.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
DocumentSection - wrapper for document file with git-backed version control.

Provides intuitive version control API while leveraging git internally.
"""

from pathlib import Path
from typing import Optional
import subprocess

from scitex.logging import getLogger

logger = getLogger(__name__)


class DocumentSection:
    """
    Wrapper for document section file with git-backed version control.

    Provides simple version control API while leveraging git internally:
    - Users get intuitive .read(), .write(), .save(), .history(), .diff()
    - We maintain clean separation from git complexity
    - Enables advanced users to use git directly when needed
    """

    def __init__(self, path: Path, git_root: Optional[Path] = None):
        """
        Initialize with file path and optional git root.

        Args:
            path: Path to the document file
            git_root: Path to git repository root (for efficiency)
        """
        self.path = path
        self._git_root = git_root
        self._cached_git_root = None

    @property
    def git_root(self) -> Optional[Path]:
        """Get cached git root, finding it if needed."""
        if self._git_root is not None:
            return self._git_root
        if self._cached_git_root is None:
            self._cached_git_root = self._find_git_root()
        return self._cached_git_root

    @staticmethod
    def _find_git_root(start_path: Path = None) -> Optional[Path]:
        """Find git root by walking up directory tree."""
        if start_path is None:
            start_path = Path.cwd()
        current = start_path.absolute()
        while current != current.parent:
            if (current / ".git").exists():
                return current
            current = current.parent
        return None

    def read(self):
        """Read file contents with intelligent fallback strategy."""
        if not self.path.exists():
            logger.warning(f"File does not exist: {self.path}")
            return None

        try:
            import scitex.io as stx_io

            return stx_io.load(str(self.path))
        except ImportError:
            logger.debug("scitex.io not available, using plain text reader")
            return self._read_plain_text()
        except ValueError as e:
            logger.warning(
                f"scitex.io could not parse {self.path} ({e}), "
                "falling back to plain text"
            )
            return self._read_plain_text()
        except Exception as e:
            logger.error(
                f"Unexpected error reading {self.path}: {e}", exc_info=True
            )
            return None

    def _read_plain_text(self):
        """Read file as plain text with proper encoding handling."""
        try:
            return self.path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(
                f"UTF-8 decode failed for {self.path}, trying latin-1"
            )
            return self.path.read_text(encoding="latin-1")
        except Exception as e:
            logger.error(f"Failed to read {self.path} as text: {e}")
            return None

    def write(self, content) -> bool:
        """Write content to file."""
        try:
            if isinstance(content, (list, tuple)):
                # Join lines if content is a list
                text = "\n".join(str(line) for line in content)
            else:
                text = str(content)
            self.path.write_text(text)
            return True
        except Exception as e:
            logger.error(f"Failed to write {self.path}: {e}")
            return False

    def history(self) -> list:
        """Get version history (uses git log internally)."""
        if not self.git_root:
            logger.debug(f"No git repository for {self.path}")
            return []

        try:
            rel_path = self.path.relative_to(self.git_root)

            result = subprocess.run(
                ["git", "log", "--oneline", str(rel_path)],
                cwd=self.git_root,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                logger.debug(f"Git log failed: {result.stderr}")
                return []

            return (
                result.stdout.strip().split("\n")
                if result.stdout.strip()
                else []
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"Git log timed out for {self.path}")
            return []
        except Exception as e:
            logger.error(f"Error getting history for {self.path}: {e}")
            return []

    def diff(self, ref: str = "HEAD") -> str:
        """Get diff against git reference (default: HEAD)."""
        if not self.git_root:
            logger.debug(f"No git repository for {self.path}")
            return ""

        try:
            rel_path = self.path.relative_to(self.git_root)

            result = subprocess.run(
                ["git", "diff", ref, str(rel_path)],
                cwd=self.git_root,
                capture_output=True,
                text=True,
                timeout=5,
            )

            return result.stdout if result.returncode == 0 else ""
        except subprocess.TimeoutExpired:
            logger.warning(f"Git diff timed out for {self.path}")
            return ""
        except Exception as e:
            logger.error(f"Error getting diff for {self.path}: {e}")
            return ""

    def commit(self, message: str) -> bool:
        """Commit this file to project's git repo with retry logic."""
        from ..git_utils import git_retry

        if not self.git_root:
            logger.warning(f"No git repository found for {self.path}")
            return False

        def _do_commit():
            rel_path = self.path.relative_to(self.git_root)
            subprocess.run(
                ["git", "add", str(rel_path)],
                cwd=self.git_root,
                check=True,
                timeout=5,
            )
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.git_root,
                check=True,
                timeout=5,
            )

        try:
            git_retry(_do_commit)
            logger.info(f"Committed {self.path}: {message}")
            return True
        except TimeoutError as e:
            logger.error(f"Git lock timeout for {self.path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to commit {self.path}: {e}")
            return False

    def checkout(self, ref: str = "HEAD") -> bool:
        """Checkout file from git reference."""
        if not self.git_root:
            logger.warning(f"No git repository found for {self.path}")
            return False

        try:
            rel_path = self.path.relative_to(self.git_root)

            result = subprocess.run(
                ["git", "checkout", ref, str(rel_path)],
                cwd=self.git_root,
                capture_output=True,
                timeout=5,
            )

            if result.returncode == 0:
                logger.info(f"Checked out {self.path} from {ref}")
                return True
            else:
                logger.error(f"Git checkout failed: {result.stderr.decode()}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Git checkout timed out for {self.path}")
            return False
        except Exception as e:
            logger.error(f"Error checking out {self.path}: {e}")
            return False

    def __repr__(self) -> str:
        """String representation."""
        return f"DocumentSection({self.path.name})"


__all__ = ["DocumentSection"]

# EOF
