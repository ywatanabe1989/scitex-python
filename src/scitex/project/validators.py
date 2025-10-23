#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/project/validators.py

"""
Project validation and naming utilities.

This module provides validation logic for project names, slugs, and identifiers
following GitHub/Gitea repository naming conventions.
"""

import re
from typing import Tuple
from urllib.parse import unquote


class ProjectValidator:
    """Validator for SciTeX project names and identifiers."""

    # GitHub/Gitea compatible naming rules
    MAX_NAME_LENGTH = 100
    MIN_NAME_LENGTH = 1

    # Valid characters: alphanumeric, hyphen, underscore, period
    VALID_CHARS_PATTERN = r'^[a-zA-Z0-9._-]+$'

    # Cannot start or end with special characters
    INVALID_START_END_PATTERN = r'^[._-]|[._-]$'

    @classmethod
    def validate_repository_name(cls, name: str) -> Tuple[bool, str | None]:
        """
        Validate repository name according to GitHub/Gitea naming rules.

        Args:
            name: Project/repository name to validate

        Returns:
            Tuple of (is_valid: bool, error_message: str or None)

        Rules:
            - Cannot contain spaces
            - Must be 1-100 characters
            - Can only contain: alphanumeric, hyphens, underscores, periods
            - Cannot start or end with special characters (-, _, .)
            - Cannot be empty or whitespace only

        Examples:
            >>> ProjectValidator.validate_repository_name("my-project")
            (True, None)

            >>> ProjectValidator.validate_repository_name("my project")
            (False, "Repository name cannot contain spaces...")

            >>> ProjectValidator.validate_repository_name("-invalid")
            (False, "Repository name cannot start or end...")
        """
        # Check if empty or whitespace only
        if not name or not name.strip():
            return False, "Repository name cannot be empty"

        # Check length
        if len(name) < cls.MIN_NAME_LENGTH:
            return False, f"Repository name must be at least {cls.MIN_NAME_LENGTH} character"

        if len(name) > cls.MAX_NAME_LENGTH:
            return False, f"Repository name must be {cls.MAX_NAME_LENGTH} characters or less"

        # Check for spaces
        if ' ' in name:
            return False, "Repository name cannot contain spaces. Use hyphens (-) or underscores (_) instead."

        # Check for valid characters (alphanumeric, hyphens, underscores, periods)
        if not re.match(cls.VALID_CHARS_PATTERN, name):
            return False, "Repository name can only contain letters, numbers, hyphens (-), underscores (_), and periods (.)"

        # Check that it doesn't start or end with special characters
        if re.match(cls.INVALID_START_END_PATTERN, name):
            return False, "Repository name cannot start or end with hyphens, underscores, or periods"

        return True, None

    @classmethod
    def validate_name(cls, name: str) -> Tuple[bool, str | None]:
        """Alias for validate_repository_name for convenience."""
        return cls.validate_repository_name(name)

    @classmethod
    def generate_slug(cls, name: str) -> str:
        """
        Generate a URL-safe slug from project name.

        Follows GitHub repository naming rules:
        - Only alphanumeric, hyphens, underscores, periods
        - Cannot start or end with special characters
        - Max 100 characters
        - Lowercase

        Args:
            name: Project name

        Returns:
            URL-safe slug

        Examples:
            >>> ProjectValidator.generate_slug("My Research Project")
            'my-research-project'

            >>> ProjectValidator.generate_slug("Neural_Decoding-2025")
            'neural-decoding-2025'

            >>> ProjectValidator.generate_slug("  Project  ")
            'project'
        """
        # Convert to lowercase
        slug = name.lower().strip()

        # Replace spaces with hyphens
        slug = re.sub(r'\s+', '-', slug)

        # Replace multiple consecutive special chars with single hyphen
        slug = re.sub(r'[._-]+', '-', slug)

        # Remove invalid characters (keep only alphanumeric, hyphens, underscores, periods)
        slug = re.sub(r'[^a-z0-9._-]', '', slug)

        # Remove leading/trailing special chars
        slug = re.sub(r'^[._-]+|[._-]+$', '', slug)

        # Ensure not empty
        if not slug:
            slug = 'project'

        # Limit to 100 chars (GitHub limit)
        slug = slug[:cls.MAX_NAME_LENGTH]

        # Final cleanup: remove trailing special chars if truncation created them
        slug = re.sub(r'[._-]+$', '', slug)

        return slug or 'project'

    @classmethod
    def get_github_safe_name(cls, name: str) -> str:
        """
        Get a GitHub-safe repository name.

        This is an alias for generate_slug() but preserves more of the original name.

        Args:
            name: Original project name

        Returns:
            GitHub-compatible name

        Examples:
            >>> ProjectValidator.get_github_safe_name("My Project!")
            'my-project'
        """
        # Similar to generate_slug but preserve underscores and periods
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', name.lower())
        safe_name = re.sub(r'^[._-]+|[._-]+$', '', safe_name)
        safe_name = safe_name[:cls.MAX_NAME_LENGTH]
        return safe_name or 'scitex-project'

    @classmethod
    def get_filesystem_safe_name(cls, name: str) -> str:
        """
        Get a filesystem-safe directory name.

        Removes characters that are problematic for filesystems:
        < > : " / \ | ? *

        Args:
            name: Original project name

        Returns:
            Filesystem-safe name

        Examples:
            >>> ProjectValidator.get_filesystem_safe_name('Project: 2025')
            'Project_2025'

            >>> ProjectValidator.get_filesystem_safe_name('Data/Analysis')
            'Data_Analysis'
        """
        # Remove/replace characters that are problematic for filesystems
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)

        # Replace spaces with underscores
        safe_name = re.sub(r'\s+', '_', safe_name)

        # Filesystem limit (most systems support 255 chars)
        safe_name = safe_name[:255]

        return safe_name or 'scitex_project'

    @classmethod
    def extract_repo_name_from_url(cls, git_url: str) -> str:
        """
        Extract repository name from Git URL, preserving the original name.

        Args:
            git_url: Git repository URL

        Returns:
            Repository name extracted from URL (preserves original case and valid characters)

        Examples:
            >>> ProjectValidator.extract_repo_name_from_url(
            ...     "https://github.com/user/my-repo.git"
            ... )
            'my-repo'

            >>> ProjectValidator.extract_repo_name_from_url(
            ...     "git@github.com:user/awesome_project.git"
            ... )
            'awesome_project'

            >>> ProjectValidator.extract_repo_name_from_url(
            ...     "https://github.com/user/MyRepo"
            ... )
            'MyRepo'
        """
        git_url = git_url.strip()

        # Remove .git suffix if present
        if git_url.endswith('.git'):
            git_url = git_url[:-4]

        # Extract repo name (last part of path)
        # Works for both HTTPS and SSH formats
        repo_name = git_url.rstrip('/').split('/')[-1]

        # Only decode URL encoding if present, but keep original name otherwise
        try:
            repo_name = unquote(repo_name)
        except:
            pass

        return repo_name or 'imported-repo'

    @classmethod
    def generate_unique_slug_candidate(cls, base_slug: str, counter: int) -> str:
        """
        Generate a unique slug candidate by appending a counter.

        Args:
            base_slug: Base slug without counter
            counter: Counter to append

        Returns:
            Slug with counter appended

        Examples:
            >>> ProjectValidator.generate_unique_slug_candidate("my-project", 1)
            'my-project-1'

            >>> ProjectValidator.generate_unique_slug_candidate("my-project", 42)
            'my-project-42'
        """
        candidate = f"{base_slug}-{counter}"

        # Ensure it doesn't exceed max length
        if len(candidate) > cls.MAX_NAME_LENGTH:
            # Truncate base_slug to make room for counter
            suffix_len = len(f"-{counter}")
            max_base_len = cls.MAX_NAME_LENGTH - suffix_len
            candidate = f"{base_slug[:max_base_len]}-{counter}"

        return candidate


# Convenience functions at module level
def validate_name(name: str) -> Tuple[bool, str | None]:
    """Validate project name. See ProjectValidator.validate_repository_name()."""
    return ProjectValidator.validate_repository_name(name)


def generate_slug(name: str) -> str:
    """Generate slug from name. See ProjectValidator.generate_slug()."""
    return ProjectValidator.generate_slug(name)


def extract_repo_name(url: str) -> str:
    """Extract repo name from URL. See ProjectValidator.extract_repo_name_from_url()."""
    return ProjectValidator.extract_repo_name_from_url(url)


__all__ = [
    'ProjectValidator',
    'validate_name',
    'generate_slug',
    'extract_repo_name',
]

# EOF
