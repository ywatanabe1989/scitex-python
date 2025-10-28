#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/project/metadata.py

"""
Project metadata storage and persistence.

This module handles reading and writing project metadata to the .scitex/ directory.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ProjectMetadataStore:
    """Manages project metadata persistence in scitex/ directory."""

    METADATA_DIR = 'scitex'
    METADATA_SUBDIR = '.metadata'  # Hidden subdir inside scitex/
    CONFIG_FILE = 'config.json'
    METADATA_FILE = 'metadata.json'
    INTEGRATIONS_FILE = 'integrations.json'
    HISTORY_FILE = 'history.jsonl'

    def __init__(self, project_path: Path):
        """
        Initialize metadata store.

        Args:
            project_path: Path to project root directory
        """
        self.project_path = Path(project_path)
        self.scitex_dir = self.project_path / self.METADATA_DIR
        self.metadata_dir = self.scitex_dir / self.METADATA_SUBDIR

    def initialize(self, project_id: str, scitex_version: str = "0.1.0") -> None:
        """
        Initialize scitex/ directory structure.

        Creates:
        - scitex/ (visible directory for features)
        - scitex/.metadata/ (hidden metadata files)

        Args:
            project_id: Unique project identifier
            scitex_version: SciTeX package version

        Raises:
            FileExistsError: If scitex/.metadata/ directory already exists
        """
        if self.metadata_dir.exists():
            raise FileExistsError(f"Metadata directory already exists: {self.metadata_dir}")

        # Create scitex/.metadata/ directory
        self.metadata_dir.mkdir(parents=True, exist_ok=False)

        # Initialize config
        config = {
            "project_id": project_id,
            "version": "1.0.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "scitex_version": scitex_version
        }
        self._write_json(self.CONFIG_FILE, config)

        # Initialize empty metadata
        metadata = {
            "name": "",
            "slug": "",
            "description": "",
            "owner": "",
            "visibility": "private",
            "template": None,
            "tags": [],
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "last_activity": datetime.utcnow().isoformat() + "Z",
            "storage_used": 0
        }
        self._write_json(self.METADATA_FILE, metadata)

        # Initialize empty integrations
        integrations = {
            "cloud": {"enabled": False},
            "gitea": {"enabled": False},
            "github": {"enabled": False}
        }
        self._write_json(self.INTEGRATIONS_FILE, integrations)

        # Create empty history file
        history_path = self.metadata_dir / self.HISTORY_FILE
        history_path.touch()

        logger.info(f"Initialized project metadata at {self.metadata_dir}")

    def exists(self) -> bool:
        """Check if scitex/.metadata/ directory exists."""
        return self.metadata_dir.exists()

    def read_config(self) -> Dict[str, Any]:
        """Read project configuration."""
        return self._read_json(self.CONFIG_FILE)

    def read_metadata(self) -> Dict[str, Any]:
        """Read project metadata."""
        return self._read_json(self.METADATA_FILE)

    def read_integrations(self) -> Dict[str, Any]:
        """Read integration configuration."""
        return self._read_json(self.INTEGRATIONS_FILE)

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Write project metadata.

        Args:
            metadata: Metadata dictionary to write
        """
        # Update last_activity timestamp
        metadata['updated_at'] = datetime.utcnow().isoformat() + "Z"
        self._write_json(self.METADATA_FILE, metadata)
        logger.debug(f"Updated project metadata: {self.METADATA_FILE}")

    def write_integrations(self, integrations: Dict[str, Any]) -> None:
        """
        Write integration configuration.

        Args:
            integrations: Integration configuration dictionary
        """
        self._write_json(self.INTEGRATIONS_FILE, integrations)
        logger.debug(f"Updated integrations: {self.INTEGRATIONS_FILE}")

    def update_metadata(self, **kwargs) -> None:
        """
        Update specific metadata fields.

        Args:
            **kwargs: Fields to update
        """
        metadata = self.read_metadata()
        metadata.update(kwargs)
        self.write_metadata(metadata)

    def update_storage(self, storage_bytes: int) -> None:
        """
        Update storage usage.

        Args:
            storage_bytes: Storage size in bytes
        """
        self.update_metadata(
            storage_used=storage_bytes,
            last_activity=datetime.utcnow().isoformat() + "Z"
        )

    def log_activity(self, action: str, **kwargs) -> None:
        """
        Log activity to history file.

        Args:
            action: Action name (e.g., 'created', 'updated', 'synced')
            **kwargs: Additional context fields
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": action,
            **kwargs
        }

        history_path = self.metadata_dir / self.HISTORY_FILE
        try:
            with open(history_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            logger.debug(f"Logged activity: {action}")
        except Exception as e:
            logger.error(f"Failed to log activity: {e}")

    def read_history(self, limit: Optional[int] = None) -> list:
        """
        Read activity history.

        Args:
            limit: Maximum number of entries to return (most recent first)

        Returns:
            List of activity entries
        """
        history_path = self.metadata_dir / self.HISTORY_FILE
        if not history_path.exists():
            return []

        entries = []
        try:
            with open(history_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read history: {e}")
            return []

        # Return most recent first
        entries.reverse()

        if limit:
            return entries[:limit]
        return entries

    def backup(self, backup_path: Optional[Path] = None) -> Path:
        """
        Create backup of scitex/.metadata/ directory.

        Args:
            backup_path: Optional backup location (default: scitex/.metadata.backup)

        Returns:
            Path to backup directory
        """
        if not self.metadata_dir.exists():
            raise FileNotFoundError(f"Metadata directory not found: {self.metadata_dir}")

        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.scitex_dir / f".metadata.backup.{timestamp}"

        shutil.copytree(self.metadata_dir, backup_path)
        logger.info(f"Created metadata backup at {backup_path}")
        return backup_path

    def restore(self, backup_path: Path) -> None:
        """
        Restore scitex/.metadata/ directory from backup.

        Args:
            backup_path: Path to backup directory

        Raises:
            FileNotFoundError: If backup doesn't exist
        """
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        # Remove existing metadata directory if present
        if self.metadata_dir.exists():
            shutil.rmtree(self.metadata_dir)

        # Restore from backup
        shutil.copytree(backup_path, self.metadata_dir)
        logger.info(f"Restored metadata from {backup_path}")

    def _read_json(self, filename: str) -> Dict[str, Any]:
        """Read JSON file from scitex/.metadata/ directory."""
        file_path = self.metadata_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filename}: {e}")
            raise

    def _write_json(self, filename: str, data: Dict[str, Any]) -> None:
        """Write JSON file to scitex/.metadata/ directory with atomic operation."""
        file_path = self.metadata_dir / filename
        temp_path = file_path.with_suffix('.tmp')

        try:
            # Write to temporary file first
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write('\n')  # Add trailing newline

            # Atomic replace
            temp_path.replace(file_path)

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Failed to write {filename}: {e}")
            raise

    def get_project_id(self) -> str:
        """Get project unique identifier."""
        config = self.read_config()
        return config['project_id']

    def validate_structure(self) -> bool:
        """
        Validate scitex/.metadata/ directory structure.

        Returns:
            True if structure is valid, False otherwise
        """
        if not self.metadata_dir.exists():
            return False

        required_files = [
            self.CONFIG_FILE,
            self.METADATA_FILE,
            self.INTEGRATIONS_FILE,
            self.HISTORY_FILE
        ]

        for filename in required_files:
            if not (self.metadata_dir / filename).exists():
                logger.error(f"Missing required file: {filename}")
                return False

        return True


def generate_project_id() -> str:
    """
    Generate unique project identifier.

    Returns:
        Unique project ID in format: proj_<random_string>

    Examples:
        >>> generate_project_id()
        'proj_abc123xyz'
    """
    import uuid
    # Use first 12 characters of UUID (hex)
    uid = uuid.uuid4().hex[:12]
    return f"proj_{uid}"


__all__ = [
    'ProjectMetadataStore',
    'generate_project_id',
]

# EOF
