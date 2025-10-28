#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/project/core.py

"""
Core SciTeX project management.

This module provides the SciTeXProject dataclass for managing project metadata
and operations without external dependencies.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from .validators import ProjectValidator, validate_name, generate_slug
from .metadata import ProjectMetadataStore, generate_project_id
from .storage import ProjectStorageCalculator, format_size

logger = logging.getLogger(__name__)


@dataclass
class SciTeXProject:
    """
    SciTeX Project dataclass.

    Represents a self-contained research project with metadata stored in
    scitex/.metadata/ directory. Works standalone without Django or Gitea.

    Attributes:
        name: Project name (must follow repository naming rules)
              For standalone use, this is just a local identifier.
              When integrated with Django, Django will generate a globally unique slug.
        path: Path to project root directory
        slug: URL-safe project identifier (optional, defaults to name)
              In standalone mode, slug = name.
              When integrated with Django, Django manages slug uniqueness.
        owner: Project owner username/identifier
        description: Project description
        visibility: 'public' or 'private'
        template: Template used to create project (optional)
        tags: List of tags/keywords
        storage_used: Storage size in bytes
        project_id: Unique project identifier (generated automatically)
        created_at: Creation timestamp
        updated_at: Last update timestamp
        last_activity: Last activity timestamp
    """

    # Core fields
    name: str
    path: Path
    owner: str
    description: str = ""

    # Optional fields with defaults
    slug: str = ""
    visibility: str = "private"
    template: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    storage_used: int = 0

    # Metadata (auto-managed)
    project_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    last_activity: str = ""

    # Internal state (not serialized)
    _metadata_store: Optional[ProjectMetadataStore] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        """Initialize project after creation."""
        # Convert path to Path object
        self.path = Path(self.path)

        # Generate slug if not provided
        if not self.slug:
            self.slug = generate_slug(self.name)

        # Validate name
        is_valid, error = validate_name(self.name)
        if not is_valid:
            raise ValueError(f"Invalid project name: {error}")

        # Validate visibility
        if self.visibility not in ('public', 'private'):
            raise ValueError(f"Invalid visibility: {self.visibility}. Must be 'public' or 'private'")

        # Initialize timestamps if not set
        now = datetime.utcnow().isoformat() + "Z"
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
        if not self.last_activity:
            self.last_activity = now

        # Initialize metadata store
        self._metadata_store = ProjectMetadataStore(self.path)

    @classmethod
    def create(
        cls,
        name: str,
        path: Path,
        owner: str,
        description: str = "",
        visibility: str = "private",
        template: Optional[str] = None,
        tags: Optional[List[str]] = None,
        init_git: bool = True,
        scitex_version: str = "0.1.0"
    ) -> 'SciTeXProject':
        """
        Create a new SciTeX project.

        Args:
            name: Project name
            path: Path to project root directory
            owner: Project owner
            description: Project description
            visibility: 'public' or 'private'
            template: Template used to create project
            tags: List of tags
            init_git: Whether to initialize git repository
            scitex_version: SciTeX package version

        Returns:
            SciTeXProject instance

        Raises:
            ValueError: If project name is invalid
            FileExistsError: If scitex/.metadata/ already exists

        Examples:
            >>> project = SciTeXProject.create(
            ...     name="My Research",
            ...     path=Path("/path/to/project"),
            ...     owner="ywatanabe",
            ...     description="Neural decoding research"
            ... )
        """
        # Validate inputs
        is_valid, error = validate_name(name)
        if not is_valid:
            raise ValueError(f"Invalid project name: {error}")

        path = Path(path)

        # Generate project ID
        project_id = generate_project_id()

        # Create project instance
        project = cls(
            name=name,
            path=path,
            owner=owner,
            description=description,
            visibility=visibility,
            template=template,
            tags=tags or [],
            project_id=project_id
        )

        # Create directory if it doesn't exist
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created project directory: {path}")

        # Initialize scitex/.metadata/ directory
        project._metadata_store.initialize(project_id, scitex_version)

        # Save metadata
        project.save()

        # Initialize git if requested
        if init_git:
            project.init_git()

        # Log creation
        project._metadata_store.log_activity('created', user=owner)

        logger.info(f"Created project '{name}' at {path}")
        return project

    @classmethod
    def load_from_directory(cls, path: Path) -> 'SciTeXProject':
        """
        Load existing project from directory.

        Args:
            path: Path to project root directory

        Returns:
            SciTeXProject instance loaded from scitex/.metadata/

        Raises:
            FileNotFoundError: If scitex/.metadata/ directory doesn't exist
            ValueError: If metadata is invalid

        Examples:
            >>> project = SciTeXProject.load_from_directory(
            ...     Path("/path/to/project")
            ... )
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Project directory not found: {path}")

        store = ProjectMetadataStore(path)

        if not store.exists():
            raise FileNotFoundError(
                f"Not a SciTeX project (no scitex/.metadata/ directory): {path}"
            )

        # Load metadata
        config = store.read_config()
        metadata = store.read_metadata()

        # Create project instance
        project = cls(
            name=metadata['name'],
            path=path,
            slug=metadata['slug'],
            owner=metadata['owner'],
            description=metadata['description'],
            visibility=metadata['visibility'],
            template=metadata.get('template'),
            tags=metadata.get('tags', []),
            storage_used=metadata.get('storage_used', 0),
            project_id=config['project_id'],
            created_at=config['created_at'],
            updated_at=metadata['updated_at'],
            last_activity=metadata.get('last_activity', metadata['updated_at'])
        )

        logger.debug(f"Loaded project '{project.name}' from {path}")
        return project

    def save(self) -> None:
        """
        Save project metadata to scitex/.metadata/ directory.

        Updates metadata.json with current project state.
        """
        if not self._metadata_store.exists():
            raise RuntimeError(
                f"Project not initialized. Call create() or initialize scitex/.metadata/ directory first."
            )

        metadata = {
            'name': self.name,
            'slug': self.slug,
            'description': self.description,
            'owner': self.owner,
            'visibility': self.visibility,
            'template': self.template,
            'tags': self.tags,
            'storage_used': self.storage_used,
            'updated_at': datetime.utcnow().isoformat() + "Z",
            'last_activity': datetime.utcnow().isoformat() + "Z"
        }

        self._metadata_store.write_metadata(metadata)
        logger.debug(f"Saved project metadata for '{self.name}'")

    def update_storage_usage(self, include_git: bool = True) -> int:
        """
        Calculate and update storage usage.

        Args:
            include_git: Whether to include .git directory in calculation

        Returns:
            Updated storage size in bytes

        Examples:
            >>> project = SciTeXProject.load_from_directory(Path("/path"))
            >>> size = project.update_storage_usage()
            >>> print(f"Storage: {size / 1024**2:.2f} MB")
        """
        calculator = ProjectStorageCalculator(
            include_git=include_git,
            include_metadata=False  # Don't count .scitex/
        )

        self.storage_used = calculator.calculate(self.path)
        self._metadata_store.update_storage(self.storage_used)

        logger.info(f"Updated storage for '{self.name}': {format_size(self.storage_used)}")
        return self.storage_used

    def get_storage_breakdown(self) -> Dict[str, Any]:
        """
        Get detailed storage breakdown.

        Returns:
            Dictionary with storage categories and sizes

        Examples:
            >>> breakdown = project.get_storage_breakdown()
            >>> print(f"User files: {breakdown['user_files'] / 1024**2:.2f} MB")
        """
        calculator = ProjectStorageCalculator(include_git=True, include_metadata=False)
        return calculator.calculate_by_category(self.path)

    def validate(self) -> bool:
        """
        Validate project metadata and structure.

        Returns:
            True if project is valid

        Raises:
            ValueError: If validation fails
        """
        # Validate name
        is_valid, error = validate_name(self.name)
        if not is_valid:
            raise ValueError(f"Invalid project name: {error}")

        # Validate path exists
        if not self.path.exists():
            raise ValueError(f"Project path does not exist: {self.path}")

        # Validate scitex/.metadata/ structure
        if not self._metadata_store.validate_structure():
            raise ValueError(f"Invalid scitex/.metadata/ directory structure")

        # Validate visibility
        if self.visibility not in ('public', 'private'):
            raise ValueError(f"Invalid visibility: {self.visibility}")

        return True

    def init_git(self) -> bool:
        """
        Initialize git repository in project directory.

        Returns:
            True if successful, False otherwise
        """
        import subprocess

        git_dir = self.path / '.git'
        if git_dir.exists():
            logger.warning(f"Git repository already initialized at {self.path}")
            return True

        try:
            # Initialize git repository
            subprocess.run(
                ['git', 'init'],
                cwd=self.path,
                check=True,
                capture_output=True,
                text=True
            )

            # Create initial branches
            subprocess.run(
                ['git', 'checkout', '-b', 'main'],
                cwd=self.path,
                check=True,
                capture_output=True,
                text=True
            )

            logger.info(f"Initialized git repository at {self.path}")
            self._metadata_store.log_activity('git_initialized')
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize git: {e}")
            return False
        except FileNotFoundError:
            logger.error("Git not found. Please install git.")
            return False

    def is_git_repository(self) -> bool:
        """Check if project is a git repository."""
        return (self.path / '.git').exists()

    def get_scitex_directory(self, feature: str) -> Path:
        """
        Get path to SciTeX feature directory.

        Args:
            feature: Feature name ('scholar', 'writer', 'code', 'viz')

        Returns:
            Path to feature directory (creates if doesn't exist)

        Examples:
            >>> scholar_dir = project.get_scitex_directory('scholar')
            >>> bib_file = scholar_dir / 'bibliography.bib'
        """
        valid_features = ('scholar', 'writer', 'code', 'viz', 'metadata')
        if feature not in valid_features:
            raise ValueError(f"Invalid feature: {feature}. Must be one of {valid_features}")

        feature_dir = self.path / 'scitex' / feature
        feature_dir.mkdir(parents=True, exist_ok=True)

        return feature_dir

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert project to dictionary.

        Returns:
            Dictionary representation (excludes internal state)
        """
        data = asdict(self)
        # Remove internal fields
        data.pop('_metadata_store', None)
        # Convert Path to string
        data['path'] = str(self.path)
        return data

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get project activity history.

        Args:
            limit: Maximum number of entries (most recent first)

        Returns:
            List of activity entries
        """
        return self._metadata_store.read_history(limit=limit)

    def log_activity(self, action: str, **kwargs) -> None:
        """
        Log project activity.

        Args:
            action: Action name
            **kwargs: Additional context
        """
        self._metadata_store.log_activity(action, **kwargs)

    def __str__(self) -> str:
        """String representation."""
        return f"SciTeXProject(name='{self.name}', slug='{self.slug}', owner='{self.owner}')"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"SciTeXProject(name='{self.name}', slug='{self.slug}', "
            f"path='{self.path}', owner='{self.owner}', "
            f"visibility='{self.visibility}', storage={format_size(self.storage_used)})"
        )


__all__ = [
    'SciTeXProject',
]

# EOF
