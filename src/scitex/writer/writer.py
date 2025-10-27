#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/writer.py

"""
Writer class for manuscript LaTeX compilation.

Provides object-oriented interface to scitex-writer functionality.
"""

from pathlib import Path
from typing import Optional, Callable

from .compile import (
    compile_manuscript,
    compile_supplementary,
    compile_revision,
    CompilationResult,
)
from .watch import watch_manuscript
from .template import init_directory as _init_directory


class DocumentSection:
    """
    Wrapper for document section file with git-backed version control.

    Provides simple version control API while leveraging git internally:
    - Users get intuitive .read(), .write(), .save(), .history(), .diff()
    - We maintain clean separation from git complexity
    - Enables advanced users to use git directly when needed
    """

    def __init__(self, path: Path):
        """Initialize with file path."""
        self.path = path

    def read(self):
        """Read file contents (uses scitex.io.load internally)."""
        if not self.path.exists():
            return None
        try:
            import scitex.io as stx_io
            return stx_io.load(str(self.path))
        except Exception:
            # Fallback to plain text read
            return self.path.read_text()

    def write(self, content) -> bool:
        """Write content to file."""
        try:
            if isinstance(content, (list, tuple)):
                # Join lines if content is a list
                text = '\n'.join(str(line) for line in content)
            else:
                text = str(content)
            self.path.write_text(text)
            return True
        except Exception as e:
            from scitex.logging import getLogger
            getLogger(__name__).error(f"Failed to write {self.path}: {e}")
            return False

    def history(self) -> list:
        """Get version history (uses git log internally)."""
        import subprocess

        try:
            # Find project root (contains .git)
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.path.parent,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return []

            repo_root = Path(result.stdout.strip())
            rel_path = self.path.relative_to(repo_root)

            result = subprocess.run(
                ["git", "log", "--oneline", str(rel_path)],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return []

            lines = result.stdout.strip().split('\n')
            return [line for line in lines if line]
        except Exception:
            return []

    def diff(self) -> str:
        """Get changes vs last version (uses git diff internally)."""
        import subprocess

        try:
            result = subprocess.run(
                ["git", "diff", "HEAD", str(self.path)],
                cwd=self.path.parent,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return "No changes or file not tracked"

            return result.stdout
        except Exception as e:
            return f"Error computing diff: {e}"

    def commit(self, message: str) -> bool:
        """Commit this file to project's git repo."""
        import subprocess

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.path.parent,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return False

            repo_root = Path(result.stdout.strip())
            rel_path = self.path.relative_to(repo_root)

            # Stage this file
            subprocess.run(
                ["git", "add", str(rel_path)],
                cwd=repo_root,
                capture_output=True,
                timeout=5
            )

            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=repo_root,
                capture_output=True,
                timeout=5
            )

            return result.returncode == 0
        except Exception:
            return False

    def checkout(self, ref: str = "HEAD") -> bool:
        """Restore file from git revision (e.g., 'HEAD', 'HEAD~1')."""
        import subprocess

        try:
            result = subprocess.run(
                ["git", "show", f"{ref}:{self.path}"],
                cwd=self.path.parent,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return False

            self.path.write_text(result.stdout)
            return True
        except Exception:
            return False

    def save(self, message: str = "Save version") -> bool:
        """Save file to git (alias for commit())."""
        return self.commit(message)

    def __str__(self) -> str:
        """String representation."""
        return str(self.path)

    def __repr__(self) -> str:
        """String representation."""
        return f"DocumentSection({self.path})"


class Document:
    """Base document accessor."""

    def __init__(self, doc_dir: Path):
        """Initialize document accessor."""
        self.dir = doc_dir

    def __getattr__(self, name: str) -> DocumentSection:
        """Get file path by name (e.g., introduction -> introduction.tex)."""
        file_path = self.dir / "contents" / f"{name}.tex"
        return DocumentSection(file_path)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self.dir.name})"


class ManuscriptDocument(Document):
    """Manuscript document."""

    @property
    def abstract(self) -> DocumentSection:
        """Get abstract.tex."""
        return DocumentSection(self.dir / "contents" / "abstract.tex")

    @property
    def introduction(self) -> DocumentSection:
        """Get introduction.tex."""
        return DocumentSection(self.dir / "contents" / "introduction.tex")

    @property
    def methods(self) -> DocumentSection:
        """Get methods.tex."""
        return DocumentSection(self.dir / "contents" / "methods.tex")

    @property
    def results(self) -> DocumentSection:
        """Get results.tex."""
        return DocumentSection(self.dir / "contents" / "results.tex")

    @property
    def discussion(self) -> DocumentSection:
        """Get discussion.tex."""
        return DocumentSection(self.dir / "contents" / "discussion.tex")


class SupplementaryDocument(Document):
    """Supplementary materials document."""

    pass


class RevisionDocument(Document):
    """Revision response document."""

    pass


class Writer:
    """LaTeX manuscript compiler."""

    def __init__(
        self,
        project_dir: Path,
        name: Optional[str] = None,
        git_strategy: Optional[str] = 'child'
    ):
        """
        Initialize for project directory.

        If directory doesn't exist, creates new project.

        Args:
            project_dir: Path to project directory
            name: Project name (used if creating new project)
            git_strategy: Git initialization strategy
                - 'child': Create isolated git in project directory (default)
                - 'parent': Use parent git repository (error if not found)
                - None: Disable git initialization
        """
        self.project_dir = Path(project_dir)
        self.git_strategy = git_strategy

        # Create new project if directory doesn't exist
        if not self.project_dir.exists():
            project_name = name or self.project_dir.name
            target_dir = self.project_dir.parent

            # Choose creation method based on git_strategy
            if self.git_strategy == 'child':
                # Use full template (includes git initialization)
                _init_directory(project_name, str(target_dir))
            else:
                # Just create minimal directory structure
                # (parent or None strategy - no template needed)
                self.project_dir.mkdir(parents=True, exist_ok=True)
                for subdir in ["01_manuscript/contents", "02_supplementary/contents", "03_revision/contents", "shared"]:
                    (self.project_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Initialize git repo based on strategy
        self.git_root = self._init_git_repo()

        # Document accessors
        self.manuscript = ManuscriptDocument(self.project_dir / "01_manuscript")
        self.supplementary = SupplementaryDocument(self.project_dir / "02_supplementary")
        self.revision = RevisionDocument(self.project_dir / "03_revision")

    def _find_parent_git(self) -> Optional[Path]:
        """
        Find parent git repository by walking up directory tree.

        Returns:
            Path to parent git root, or None if not found
        """
        current = self.project_dir.absolute()
        while current != current.parent:
            if (current / '.git').exists():
                return current
            current = current.parent
        return None

    def _create_child_git(self) -> Optional[Path]:
        """
        Create isolated git repository in project directory.

        Returns:
            Path to git root (project_dir), or None on failure
        """
        import subprocess

        try:
            # Check if already a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.project_dir,
                capture_output=True,
                timeout=5
            )

            if result.returncode == 0:
                # Already a git repo
                return self.project_dir

            # Initialize new git repo
            subprocess.run(
                ["git", "init"],
                cwd=self.project_dir,
                capture_output=True,
                timeout=5
            )

            # Initial commit
            subprocess.run(
                ["git", "add", "."],
                cwd=self.project_dir,
                capture_output=True,
                timeout=5
            )

            subprocess.run(
                ["git", "commit", "-m", "Initial commit from scitex-writer"],
                cwd=self.project_dir,
                capture_output=True,
                timeout=5
            )

            return self.project_dir
        except Exception:
            return None

    def _init_git_repo(self) -> Optional[Path]:
        """
        Initialize or detect git repository based on git_strategy.

        Returns:
            Path to git repository root, or None if disabled

        Raises:
            ValueError: If strategy='parent' but no parent git found
        """
        # Strategy: disabled
        if self.git_strategy is None:
            return None

        # Strategy: parent (require existing parent repo)
        if self.git_strategy == 'parent':
            parent_git = self._find_parent_git()
            if not parent_git:
                raise ValueError(
                    f"git_strategy='parent' but no parent git repository found "
                    f"for {self.project_dir}"
                )
            return parent_git

        # Strategy: child (create isolated repo)
        if self.git_strategy == 'child':
            return self._create_child_git()

        # Unknown strategy
        raise ValueError(
            f"Unknown git_strategy: {self.git_strategy}. "
            f"Expected 'parent', 'child', or None"
        )

    def compile_manuscript(self, timeout: int = 300) -> CompilationResult:
        """Compile manuscript."""
        return compile_manuscript(self.project_dir, timeout=timeout)

    def compile_supplementary(self, timeout: int = 300) -> CompilationResult:
        """Compile supplementary materials."""
        return compile_supplementary(self.project_dir, timeout=timeout)

    def compile_revision(
        self,
        track_changes: bool = False,
        timeout: int = 300
    ) -> CompilationResult:
        """Compile revision."""
        return compile_revision(
            self.project_dir,
            track_changes=track_changes,
            timeout=timeout
        )

    def watch(self, on_compile: Optional[Callable] = None) -> None:
        """Auto-recompile on file changes."""
        watch_manuscript(self.project_dir, on_compile=on_compile)

    def get_pdf(self, doc_type: str = 'manuscript') -> Optional[Path]:
        """Get output PDF path (Read)."""
        doc_map = {
            'manuscript': '01_manuscript',
            'supplementary': '02_supplementary',
            'revision': '03_revision'
        }
        pdf = self.project_dir / doc_map[doc_type] / f"{doc_type}.pdf"
        return pdf if pdf.exists() else None

    def delete(self) -> bool:
        """Delete project directory (Delete)."""
        import shutil
        try:
            shutil.rmtree(self.project_dir)
            return True
        except Exception:
            return False


__all__ = ['Writer']

# EOF
