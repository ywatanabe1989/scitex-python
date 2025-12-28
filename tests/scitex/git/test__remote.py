#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
pytest.importorskip("git")

from scitex.git._remote import (
    _normalize_git_url,
    _validate_git_url,
    get_remote_url,
    is_cloned_from,
)


class TestGitRemote:
    def test_validate_github_https(self):
        url = "https://github.com/user/repo.git"
        assert _validate_git_url(url) is True

    def test_validate_github_ssh(self):
        url = "git@github.com:user/repo.git"
        assert _validate_git_url(url) is True

    def test_validate_gitlab_https(self):
        url = "https://gitlab.com/user/repo.git"
        assert _validate_git_url(url) is True

    def test_validate_bitbucket_https(self):
        url = "https://bitbucket.org/user/repo.git"
        assert _validate_git_url(url) is True

    def test_validate_invalid_url(self):
        url = "https://invalid.com/user/repo.git"
        assert _validate_git_url(url) is False

    def test_normalize_https_url(self):
        url = "https://github.com/user/repo.git"
        expected = "https://github.com/user/repo"
        assert _normalize_git_url(url) == expected

    def test_normalize_ssh_url(self):
        url = "git@github.com:user/repo.git"
        expected = "https://github.com/user/repo"
        assert _normalize_git_url(url) == expected

    def test_normalize_url_without_git_suffix(self):
        url = "https://github.com/user/repo/"
        expected = "https://github.com/user/repo"
        assert _normalize_git_url(url) == expected

    def test_get_remote_url_no_repo(self, tmp_path):
        non_repo = tmp_path / "not_repo"
        non_repo.mkdir()
        result = get_remote_url(non_repo, verbose=False)
        assert result is None

    def test_is_cloned_from_non_git(self, tmp_path):
        non_repo = tmp_path / "not_repo"
        non_repo.mkdir()
        result = is_cloned_from(non_repo, "https://github.com/user/repo.git")
        assert result is False


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_remote.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/git/remote.py
# 
# """
# Git remote operations.
# """
# 
# from pathlib import Path
# from typing import Optional
# from scitex.logging import getLogger
# from scitex.sh import sh
# from ._utils import _in_directory
# from ._constants import EXIT_SUCCESS, EXIT_FAILURE
# 
# logger = getLogger(__name__)
# 
# 
# def get_remote_url(
#     repo_path: Path, remote_name: str = "origin", verbose: bool = False
# ) -> Optional[str]:
#     """
#     Get remote URL for a git repository.
# 
#     Parameters
#     ----------
#     repo_path : Path
#         Git repository path
#     remote_name : str
#         Remote name (default: origin)
#     verbose : bool
#         Enable verbose output
# 
#     Returns
#     -------
#     Optional[str]
#         Remote URL if found, None otherwise
# 
#     Notes
#     -----
#     Returns None if remote doesn't exist or repo is not a git repository.
#     Check stderr via sh() directly if detailed error info is needed.
#     """
#     if not (repo_path / ".git").exists():
#         logger.warning(f"Not a git repository: {repo_path}")
#         return None
# 
#     with _in_directory(repo_path):
#         result = sh(
#             ["git", "config", "--get", f"remote.{remote_name}.url"],
#             verbose=verbose,
#             return_as="dict",
#         )
#         if result["success"]:
#             return result["stdout"].strip()
# 
#         logger.debug(f"Remote '{remote_name}' not found in {repo_path}")
#         return None
# 
# 
# def _validate_git_url(url: str) -> bool:
#     """
#     Validate git URL format.
# 
#     Parameters
#     ----------
#     url : str
#         Git URL to validate
# 
#     Returns
#     -------
#     bool
#         True if valid, False otherwise
#     """
#     valid_hosts = ("github.com", "gitlab.com", "bitbucket.org")
# 
#     if url.startswith("https://"):
#         for host in valid_hosts:
#             if f"https://{host}/" in url:
#                 return True
# 
#     elif url.startswith("git@"):
#         for host in valid_hosts:
#             if url.startswith(f"git@{host}:"):
#                 return True
# 
#     return False
# 
# 
# def _normalize_git_url(url: str) -> str:
#     """
#     Normalize git URL for comparison.
#     Handles HTTPS and SSH formats for GitHub, GitLab, Bitbucket.
# 
#     Parameters
#     ----------
#     url : str
#         Git URL to normalize
# 
#     Returns
#     -------
#     str
#         Normalized URL
#     """
#     url = url.rstrip("/")
# 
#     if url.startswith("git@"):
#         url = url.replace(":", "/", 1).replace("git@", "https://")
# 
#     if url.endswith(".git"):
#         url = url[:-4]
# 
#     return url
# 
# 
# def is_cloned_from(
#     repo_path: Path, expected_url: str, remote_name: str = "origin"
# ) -> bool:
#     """
#     Check if directory is a git repository cloned from specific URL.
#     Handles both HTTPS and SSH URL formats.
# 
#     Parameters
#     ----------
#     repo_path : Path
#         Directory to check
#     expected_url : str
#         Expected remote URL
#     remote_name : str
#         Remote name to check (default: origin)
# 
#     Returns
#     -------
#     bool
#         True if directory is cloned from expected URL, False otherwise
#     """
#     if not (repo_path / ".git").exists():
#         return False
#     actual_url = get_remote_url(repo_path, remote_name)
#     if actual_url is None:
#         return False
#     return _normalize_git_url(actual_url) == _normalize_git_url(expected_url)
# 
# 
# def main(args):
#     if args.action == "get-url":
#         url = get_remote_url(args.repo_path, args.remote_name, args.verbose)
#         if url:
#             print(url)
#             return EXIT_SUCCESS
#         return EXIT_FAILURE
#     elif args.action == "check-origin":
#         if not args.expected_url:
#             logger.error("Expected URL required for check-origin action")
#             return EXIT_FAILURE
#         result = is_cloned_from(args.repo_path, args.expected_url, args.remote_name)
#         print(result)
#         return EXIT_SUCCESS if result else EXIT_FAILURE
# 
# 
# def parse_args():
#     """Parse command line arguments."""
#     import argparse
# 
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--action", choices=["get-url", "check-origin"], required=True)
#     parser.add_argument("--repo-path", type=Path, required=True)
#     parser.add_argument("--expected-url", help="Expected URL for check-origin action")
#     parser.add_argument("--remote-name", default="origin")
#     parser.add_argument("--verbose", action="store_true")
#     return parser.parse_args()
# 
# 
# def run_session():
#     """Initialize scitex framework, run main function, and cleanup."""
#     from ._session import run_with_session
# 
#     run_with_session(parse_args, main)
# 
# 
# __all__ = [
#     "get_remote_url",
#     "is_cloned_from",
#     "_validate_git_url",
# ]
# 
# 
# if __name__ == "__main__":
#     run_session()
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_remote.py
# --------------------------------------------------------------------------------
