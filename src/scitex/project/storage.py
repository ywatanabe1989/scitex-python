#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/project/storage.py

"""
Project storage calculation utilities.

This module provides functions to calculate storage usage for SciTeX projects.
"""

from pathlib import Path
from typing import List, Optional


class ProjectStorageCalculator:
    """Calculator for project storage usage."""

    # Patterns to ignore in storage calculations
    DEFAULT_IGNORE_PATTERNS = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.Python',
        '*.so',
        '*.egg',
        '*.egg-info',
        'dist',
        'build',
        '.pytest_cache',
        '.mypy_cache',
        '.tox',
        '.coverage',
        'htmlcov',
        '.venv',
        'venv',
        'ENV',
        'env',
        '.DS_Store',
        'Thumbs.db',
        '*.swp',
        '*.swo',
        '*~',
        '.tmp',
        'tmp',
    ]

    # Metadata directory (always excluded from storage display)
    METADATA_DIR = 'scitex/.metadata'

    def __init__(
        self,
        ignore_patterns: Optional[List[str]] = None,
        include_git: bool = True,
        include_metadata: bool = False
    ):
        """
        Initialize storage calculator.

        Args:
            ignore_patterns: Additional patterns to ignore (extends defaults)
            include_git: Whether to include .git directory in calculations
            include_metadata: Whether to include .scitex directory in calculations
        """
        self.ignore_patterns = self.DEFAULT_IGNORE_PATTERNS.copy()
        if ignore_patterns:
            self.ignore_patterns.extend(ignore_patterns)

        self.include_git = include_git
        self.include_metadata = include_metadata

        # Add .git to ignore patterns if not including
        if not include_git:
            self.ignore_patterns.append('.git')

        # Always add scitex/.metadata to ignore unless explicitly including
        if not include_metadata:
            self.ignore_patterns.append(self.METADATA_DIR)

    def calculate(self, project_path: Path) -> int:
        """
        Calculate total storage used by project.

        Args:
            project_path: Path to project root directory

        Returns:
            Total storage in bytes

        Raises:
            ValueError: If project_path doesn't exist or isn't a directory
        """
        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")

        if not project_path.is_dir():
            raise ValueError(f"Project path is not a directory: {project_path}")

        total_size = 0

        try:
            for item in project_path.rglob('*'):
                if self._should_ignore(item, project_path):
                    continue

                if item.is_file():
                    try:
                        total_size += item.stat().st_size
                    except (OSError, PermissionError):
                        # Skip files we can't access
                        pass

        except Exception as e:
            # Log error but don't fail completely
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error calculating storage for {project_path}: {e}")

        return total_size

    def _should_ignore(self, path: Path, project_root: Path) -> bool:
        """
        Check if path should be ignored in storage calculations.

        Args:
            path: Path to check
            project_root: Project root directory (for relative path calculations)

        Returns:
            True if path should be ignored
        """
        # Get relative path from project root
        try:
            rel_path = path.relative_to(project_root)
        except ValueError:
            # Path is not relative to project_root
            return False

        # Check each part of the path against ignore patterns
        for part in rel_path.parts:
            for pattern in self.ignore_patterns:
                # Simple pattern matching (supports *.ext and directory names)
                if pattern.startswith('*'):
                    # Glob-style pattern (e.g., *.pyc)
                    if part.endswith(pattern[1:]):
                        return True
                elif part == pattern:
                    # Exact match
                    return True

        return False

    def calculate_by_category(self, project_path: Path) -> dict:
        """
        Calculate storage broken down by category.

        Args:
            project_path: Path to project root directory

        Returns:
            Dictionary with storage breakdown:
            {
                'total': int,
                'git': int,
                'scitex': int,
                'user_files': int,
                'breakdown': {
                    'data': int,
                    'scripts': int,
                    'docs': int,
                    'other': int
                }
            }
        """
        if not project_path.exists() or not project_path.is_dir():
            return {
                'total': 0,
                'git': 0,
                'scitex': 0,
                'user_files': 0,
                'breakdown': {}
            }

        result = {
            'total': 0,
            'git': 0,
            'scitex': 0,
            'user_files': 0,
            'breakdown': {}
        }

        # Calculate git directory size
        git_dir = project_path / '.git'
        if git_dir.exists():
            result['git'] = self._calculate_directory_size(git_dir)

        # Calculate scitex metadata directory size
        scitex_metadata_dir = project_path / 'scitex' / '.metadata'
        if scitex_metadata_dir.exists():
            result['scitex'] = self._calculate_directory_size(scitex_metadata_dir)

        # Calculate user files by category
        common_dirs = ['data', 'scripts', 'docs', 'src', 'tests', 'results']
        for dir_name in common_dirs:
            dir_path = project_path / dir_name
            if dir_path.exists():
                result['breakdown'][dir_name] = self._calculate_directory_size(dir_path)

        # Calculate other files (not in common dirs)
        other_size = 0
        for item in project_path.iterdir():
            if item.name.startswith('.'):
                continue
            if item.name in common_dirs:
                continue
            if item.is_file():
                try:
                    other_size += item.stat().st_size
                except (OSError, PermissionError):
                    pass
            elif item.is_dir():
                other_size += self._calculate_directory_size(item)

        result['breakdown']['other'] = other_size

        # Sum up user files
        result['user_files'] = sum(result['breakdown'].values())

        # Total (git + scitex + user_files)
        result['total'] = result['git'] + result['scitex'] + result['user_files']

        return result

    def _calculate_directory_size(self, directory: Path) -> int:
        """
        Calculate total size of a directory.

        Args:
            directory: Directory path

        Returns:
            Total size in bytes
        """
        total = 0
        try:
            for item in directory.rglob('*'):
                if item.is_file():
                    try:
                        total += item.stat().st_size
                    except (OSError, PermissionError):
                        pass
        except Exception:
            pass
        return total


def calculate_storage(
    project_path: Path,
    include_git: bool = True,
    include_metadata: bool = False
) -> int:
    """
    Calculate project storage usage.

    Args:
        project_path: Path to project root directory
        include_git: Whether to include .git directory
        include_metadata: Whether to include scitex/.metadata directory

    Returns:
        Total storage in bytes

    Examples:
        >>> from pathlib import Path
        >>> calculate_storage(Path("/path/to/project"))
        1048576

        >>> calculate_storage(Path("/path/to/project"), include_git=False)
        524288
    """
    calculator = ProjectStorageCalculator(
        include_git=include_git,
        include_metadata=include_metadata
    )
    return calculator.calculate(project_path)


def format_size(size_bytes: int) -> str:
    """
    Format byte size to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string

    Examples:
        >>> format_size(1024)
        '1.00 KB'

        >>> format_size(1048576)
        '1.00 MB'

        >>> format_size(1073741824)
        '1.00 GB'
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


__all__ = [
    'ProjectStorageCalculator',
    'calculate_storage',
    'format_size',
]

# EOF
