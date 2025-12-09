#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:08:40 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/_DocumentSection.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/dataclasses/_DocumentSection.py"
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
            logger.error(f"Unexpected error reading {self.path}: {e}", exc_info=True)
            return None

    def _read_plain_text(self):
        """Read file as plain text with proper encoding handling."""
        try:
            return self.path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decode failed for {self.path}, trying latin-1")
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

            return result.stdout.strip().split("\n") if result.stdout.strip() else []
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

    def diff_between(self, ref1: str, ref2: str) -> str:
        """
        Compare two arbitrary git references.

        Args:
            ref1: First git reference (commit, branch, tag, or human-readable spec)
            ref2: Second git reference (commit, branch, tag, or human-readable spec)

        Returns:
            Diff output string, or "" if error or no differences.

        Examples:
            section.diff_between("HEAD~2", "HEAD")
            section.diff_between("v1.0", "v2.0")
            section.diff_between("main", "develop")
            section.diff_between("2 days ago", "now")
        """
        if not self.git_root:
            logger.debug(f"No git repository for {self.path}")
            return ""

        try:
            # Resolve human-readable refs to commit hashes
            resolved_ref1 = self._resolve_ref(ref1)
            resolved_ref2 = self._resolve_ref(ref2)

            if not resolved_ref1 or not resolved_ref2:
                logger.error(f"Failed to resolve references: {ref1} or {ref2}")
                return ""

            rel_path = self.path.relative_to(self.git_root)

            # Use git diff ref1..ref2 file (three-dot shows what changed on ref2 since ref1 diverged)
            result = subprocess.run(
                [
                    "git",
                    "diff",
                    f"{resolved_ref1}..{resolved_ref2}",
                    str(rel_path),
                ],
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
            logger.error(f"Error getting diff_between for {self.path}: {e}")
            return ""

    def _resolve_ref(self, spec: str) -> Optional[str]:
        """
        Resolve human-readable reference specification to git reference.

        Handles:
        - Standard git refs: HEAD, HEAD~N, branch, tag, commit hash
        - Relative time: "N days ago", "N weeks ago", "N hours ago", "now"
        - Absolute dates: "2025-10-28", "2025-10-28 14:30"

        Args:
            spec: Reference specification

        Returns:
            Git reference (commit hash or ref name), or None if invalid.
        """
        if not self.git_root:
            return None

        spec = spec.strip()

        # Direct git reference (HEAD, branch, tag, hash)
        if self._is_valid_git_ref(spec):
            return spec

        # Handle "now" as HEAD
        if spec.lower() == "now":
            return "HEAD"

        # Handle "today" as start of day
        if spec.lower() == "today":
            from datetime import datetime, timedelta

            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            return self._find_commit_at_timestamp(today)

        # Handle relative time like "2 days ago", "1 week ago", "24 hours ago"
        time_ref = self._parse_relative_time(spec)
        if time_ref:
            return self._find_commit_at_timestamp(time_ref)

        # Handle absolute date like "2025-10-28" or "2025-10-28 14:30"
        date_ref = self._parse_absolute_date(spec)
        if date_ref:
            return self._find_commit_at_timestamp(date_ref)

        logger.warning(f"Could not resolve reference: {spec}")
        return None

    def _is_valid_git_ref(self, ref: str) -> bool:
        """Check if reference exists in git repository."""
        if not self.git_root:
            return False

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", ref],
                cwd=self.git_root,
                capture_output=True,
                timeout=2,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _parse_relative_time(self, spec: str):
        """
        Parse relative time specification like "2 days ago".

        Returns:
            datetime object or None if not a valid time spec.
        """
        import re
        from datetime import datetime, timedelta

        # Pattern: "N <unit> ago"
        match = re.match(r"(\d+)\s*(day|week|hour|minute)s?\s*ago", spec, re.IGNORECASE)
        if not match:
            return None

        amount = int(match.group(1))
        unit = match.group(2).lower()

        now = datetime.now()
        if unit == "day":
            return now - timedelta(days=amount)
        elif unit == "week":
            return now - timedelta(weeks=amount)
        elif unit == "hour":
            return now - timedelta(hours=amount)
        elif unit == "minute":
            return now - timedelta(minutes=amount)

        return None

    def _parse_absolute_date(self, spec: str):
        """
        Parse absolute date specification like "2025-10-28" or "2025-10-28 14:30".

        Returns:
            datetime object or None if not a valid date spec.
        """
        from datetime import datetime

        # Try YYYY-MM-DD HH:MM format
        try:
            return datetime.strptime(spec, "%Y-%m-%d %H:%M")
        except ValueError:
            pass

        # Try YYYY-MM-DD format
        try:
            return datetime.strptime(spec, "%Y-%m-%d")
        except ValueError:
            pass

        return None

    def _find_commit_at_timestamp(self, target_datetime) -> Optional[str]:
        """
        Find commit closest to (before) given timestamp.

        Args:
            target_datetime: datetime object

        Returns:
            Commit hash or None if not found.
        """
        if not self.git_root:
            return None

        try:
            # Format timestamp for git
            timestamp_str = target_datetime.strftime("%Y-%m-%d %H:%M:%S")

            # Find commit at or before this timestamp
            result = subprocess.run(
                [
                    "git",
                    "log",
                    "--format=%H",
                    "--before=" + timestamp_str,
                    "-1",  # Get only the most recent one
                ],
                cwd=self.git_root,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            else:
                logger.warning(f"No commit found before {timestamp_str}")
                return None
        except subprocess.TimeoutExpired:
            logger.warning(f"Git log timed out looking for commit at {target_datetime}")
            return None
        except Exception as e:
            logger.error(f"Error finding commit at timestamp: {e}")
            return None

    def commit(self, message: str) -> bool:
        """Commit this file to project's git repo with retry logic."""
        from scitex.git import git_retry

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


def run_session() -> None:
    """Initialize scitex framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt, rng
    import sys
    import matplotlib.pyplot as plt
    import scitex as stx

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = stx.session.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        sdir_suffix=None,
        verbose=False,
        agg=True,
    )

    exit_status = main(args)

    stx.session.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


def main(args):
    section = DocumentSection(Path(args.file))

    if args.action == "read":
        content = section.read()
        print(content if content else "File not found or empty")

    elif args.action == "history":
        history = section.history()
        print(f"History ({len(history)} commits):")
        for entry in history:
            print(f"  {entry}")

    elif args.action == "diff":
        diff = section.diff(args.ref)
        print(diff if diff else "No differences")

    return 0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstrate DocumentSection version control"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="Path to document section file",
    )
    parser.add_argument(
        "--action",
        "-a",
        type=str,
        choices=["read", "history", "diff"],
        default="read",
        help="Action to perform (default: read)",
    )
    parser.add_argument(
        "--ref",
        "-r",
        type=str,
        default="HEAD",
        help="Git reference for diff (default: HEAD)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_session()


__all__ = ["DocumentSection"]

# python -m scitex.writer.dataclasses.core._DocumentSection --file ./01_manuscript/contents/introduction.tex --action history

# EOF
