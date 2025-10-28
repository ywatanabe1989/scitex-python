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
            if (current / '.git').exists():
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
                f"Unexpected error reading {self.path}: {e}",
                exc_info=True
            )
            return None

    def _read_plain_text(self):
        """Read file as plain text with proper encoding handling."""
        try:
            return self.path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decode failed for {self.path}, trying latin-1")
            return self.path.read_text(encoding='latin-1')
        except Exception as e:
            logger.error(f"Failed to read {self.path} as text: {e}")
            return None

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
                timeout=5
            )

            if result.returncode != 0:
                logger.debug(f"No history found for {self.path}")
                return []

            lines = result.stdout.strip().split('\n')
            return [line for line in lines if line]
        except ValueError as e:
            logger.warning(f"File {self.path} not in git repository: {e}")
            return []
        except subprocess.TimeoutExpired:
            logger.error(f"Git log timed out for {self.path}")
            return []
        except Exception as e:
            logger.error(f"Error getting history for {self.path}: {e}", exc_info=True)
            return []

    def diff(self) -> str:
        """Get changes vs last version (uses git diff internally)."""
        import subprocess

        if not self.git_root:
            return "No git repository"

        try:
            result = subprocess.run(
                ["git", "diff", "HEAD", str(self.path)],
                cwd=self.git_root,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return "No changes or file not tracked"

            return result.stdout
        except subprocess.TimeoutExpired:
            return "Git diff timed out"
        except Exception as e:
            logger.error(f"Error computing diff for {self.path}: {e}", exc_info=True)
            return f"Error computing diff: {e}"

    def commit(self, message: str) -> bool:
        """Commit this file to project's git repo with retry logic."""
        import subprocess
        from .git_utils import git_retry

        if not self.git_root:
            logger.warning(f"No git repository found for {self.path}")
            return False

        if not self.path.exists():
            logger.error(f"File does not exist: {self.path}")
            return False

        def _do_commit():
            """Perform git add and commit."""
            rel_path = self.path.relative_to(self.git_root)

            # Stage the file
            subprocess.run(
                ["git", "add", str(rel_path)],
                cwd=self.git_root,
                capture_output=True,
                check=True,
                timeout=5
            )

            # Commit
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.git_root,
                capture_output=True,
                check=True,
                timeout=5
            )

        try:
            git_retry(_do_commit)
            logger.info(f"Committed {self.path}: {message}")
            return True
        except TimeoutError as e:
            logger.error(f"Git lock timeout for {self.path}: {e}")
            return False
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else (e.stderr or '')
            if "nothing to commit" in stderr.lower():
                logger.debug(f"No changes to commit in {self.path}")
                return True
            logger.error(f"Commit failed for {self.path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during commit: {e}", exc_info=True)
            return False

    def checkout(self, ref: str = "HEAD") -> bool:
        """Restore file from git revision (e.g., 'HEAD', 'HEAD~1')."""
        import subprocess

        if not self.git_root:
            logger.warning(f"No git repository found for {self.path}")
            return False

        try:
            result = subprocess.run(
                ["git", "show", f"{ref}:{self.path}"],
                cwd=self.git_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=True
            )

            self.path.write_text(result.stdout)
            logger.info(f"Restored {self.path} from {ref}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Checkout failed for {self.path}: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"Git show timed out for {self.path}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during checkout: {e}", exc_info=True)
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

    def __init__(self, doc_dir: Path, git_root: Optional[Path] = None):
        """Initialize document accessor."""
        self.dir = doc_dir
        self.git_root = git_root

    def __getattr__(self, name: str) -> DocumentSection:
        """Get file path by name (e.g., introduction -> introduction.tex)."""
        file_path = self.dir / "contents" / f"{name}.tex"
        return DocumentSection(file_path, git_root=self.git_root)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self.dir.name})"


class ManuscriptDocument(Document):
    """Manuscript document."""

    @property
    def abstract(self) -> DocumentSection:
        """Get abstract.tex."""
        return DocumentSection(self.dir / "contents" / "abstract.tex", git_root=self.git_root)

    @property
    def introduction(self) -> DocumentSection:
        """Get introduction.tex."""
        return DocumentSection(self.dir / "contents" / "introduction.tex", git_root=self.git_root)

    @property
    def methods(self) -> DocumentSection:
        """Get methods.tex."""
        return DocumentSection(self.dir / "contents" / "methods.tex", git_root=self.git_root)

    @property
    def results(self) -> DocumentSection:
        """Get results.tex."""
        return DocumentSection(self.dir / "contents" / "results.tex", git_root=self.git_root)

    @property
    def discussion(self) -> DocumentSection:
        """Get discussion.tex."""
        return DocumentSection(self.dir / "contents" / "discussion.tex", git_root=self.git_root)


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

        # Document accessors (pass git_root for efficiency)
        self.manuscript = ManuscriptDocument(self.project_dir / "01_manuscript", git_root=self.git_root)
        self.supplementary = SupplementaryDocument(self.project_dir / "02_supplementary", git_root=self.git_root)
        self.revision = RevisionDocument(self.project_dir / "03_revision", git_root=self.git_root)

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
        Initialize or detect git repository based on git_strategy with graceful degradation.

        Returns:
            Path to git repository root, or None if disabled

        Strategy details:
        - None: Git disabled, returns None
        - 'child': Creates isolated git repo in project directory
        - 'parent': Tries to use parent git, degrades to 'child' if not found
        """
        # Strategy: disabled
        if self.git_strategy is None:
            return None

        # Strategy: parent (with graceful degradation to child)
        if self.git_strategy == 'parent':
            parent_git = self._find_parent_git()

            if parent_git:
                logger.info(f"Using parent git repository: {parent_git}")
                return parent_git

            # Graceful degradation: no parent git found, use child strategy
            logger.warning(
                f"No parent git repository found for {self.project_dir}. "
                f"Degrading to 'child' strategy (isolated git repo)."
            )
            return self._create_child_git()

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
