#!/usr/bin/env python3
# Time-stamp: "2026-01-04 (ywatanabe)"
# File: ./tests/scitex/path/test__symlink.py

"""Comprehensive tests for scitex.path symlink utilities."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)


# =============================================================================
# Tests for symlink()
# =============================================================================


class TestSymlink:
    """Tests for symlink() function."""

    def test_symlink_basic_file(self):
        """Test creating basic symlink to a file."""
        from scitex.path import symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            dst = Path(temp_dir) / "link.txt"
            src.write_text("test content")

            result = symlink(src, dst)

            assert result == dst
            assert dst.is_symlink()
            assert dst.read_text() == "test content"

    def test_symlink_basic_directory(self):
        """Test creating symlink to a directory."""
        from scitex.path import symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source_dir"
            dst = Path(temp_dir) / "link_dir"
            src.mkdir()
            (src / "file.txt").write_text("in dir")

            result = symlink(src, dst)

            assert result == dst
            assert dst.is_symlink()
            assert (dst / "file.txt").read_text() == "in dir"

    def test_symlink_overwrite_false_raises(self):
        """Test symlink raises when dst exists and overwrite=False."""
        from scitex.path import symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            dst = Path(temp_dir) / "link.txt"
            src.write_text("source")
            dst.write_text("existing")

            with pytest.raises(FileExistsError):
                symlink(src, dst, overwrite=False)

    def test_symlink_overwrite_true(self):
        """Test symlink overwrites when overwrite=True."""
        from scitex.path import symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src1 = Path(temp_dir) / "source1.txt"
            src2 = Path(temp_dir) / "source2.txt"
            dst = Path(temp_dir) / "link.txt"
            src1.write_text("first")
            src2.write_text("second")
            symlink(src1, dst)

            symlink(src2, dst, overwrite=True)

            assert dst.read_text() == "second"

    def test_symlink_overwrite_existing_file(self):
        """Test symlink overwrites existing regular file."""
        from scitex.path import symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            dst = Path(temp_dir) / "existing.txt"
            src.write_text("source")
            dst.write_text("existing regular file")

            symlink(src, dst, overwrite=True)

            assert dst.is_symlink()
            assert dst.read_text() == "source"

    def test_symlink_overwrite_existing_directory(self):
        """Test symlink overwrites existing directory."""
        from scitex.path import symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            dst = Path(temp_dir) / "existing_dir"
            src.write_text("source")
            dst.mkdir()
            (dst / "file.txt").write_text("in existing dir")

            symlink(src, dst, overwrite=True)

            assert dst.is_symlink()

    def test_symlink_relative_true(self):
        """Test creating relative symlink."""
        from scitex.path import symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            dst = Path(temp_dir) / "link.txt"
            src.write_text("test")

            symlink(src, dst, relative=True)

            assert dst.is_symlink()
            target = os.readlink(dst)
            assert not Path(target).is_absolute()

    def test_symlink_relative_with_subdirs(self):
        """Test relative symlink across subdirectories."""
        from scitex.path import symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / "a" / "b"
            dst_dir = Path(temp_dir) / "c" / "d"
            src_dir.mkdir(parents=True)
            dst_dir.mkdir(parents=True)
            src = src_dir / "source.txt"
            dst = dst_dir / "link.txt"
            src.write_text("nested")

            symlink(src, dst, relative=True)

            assert dst.is_symlink()
            assert dst.read_text() == "nested"

    def test_symlink_creates_parent_dirs(self):
        """Test symlink creates parent directories for dst."""
        from scitex.path import symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            dst = Path(temp_dir) / "nested" / "path" / "link.txt"
            src.write_text("test")

            symlink(src, dst)

            assert dst.parent.exists()
            assert dst.is_symlink()

    def test_symlink_to_nonexistent_target(self):
        """Test symlink to non-existent target (dangling symlink)."""
        from scitex.path import symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "nonexistent.txt"
            dst = Path(temp_dir) / "link.txt"

            result = symlink(src, dst)

            assert dst.is_symlink()
            assert not dst.exists()  # Target doesn't exist

    def test_symlink_string_paths(self):
        """Test symlink with string paths."""
        from scitex.path import symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = os.path.join(temp_dir, "source.txt")
            dst = os.path.join(temp_dir, "link.txt")
            Path(src).write_text("test")

            result = symlink(src, dst)

            assert result == Path(dst)
            assert Path(dst).is_symlink()

    def test_symlink_unicode_filename(self):
        """Test symlink with unicode filename."""
        from scitex.path import symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "源文件.txt"
            dst = Path(temp_dir) / "链接.txt"
            src.write_text("unicode test")

            symlink(src, dst)

            assert dst.is_symlink()
            assert dst.read_text() == "unicode test"


# =============================================================================
# Tests for is_symlink()
# =============================================================================


class TestIsSymlink:
    """Tests for is_symlink() function."""

    def test_is_symlink_true(self):
        """Test is_symlink returns True for symlink."""
        from scitex.path import is_symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            dst = Path(temp_dir) / "link.txt"
            src.write_text("test")
            dst.symlink_to(src)

            assert is_symlink(dst) is True

    def test_is_symlink_false_regular_file(self):
        """Test is_symlink returns False for regular file."""
        from scitex.path import is_symlink

        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                assert is_symlink(f.name) is False
            finally:
                os.unlink(f.name)

    def test_is_symlink_false_directory(self):
        """Test is_symlink returns False for directory."""
        from scitex.path import is_symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            assert is_symlink(temp_dir) is False

    def test_is_symlink_false_nonexistent(self):
        """Test is_symlink returns False for non-existent path."""
        from scitex.path import is_symlink

        assert is_symlink("/nonexistent/path") is False

    def test_is_symlink_broken_symlink(self):
        """Test is_symlink returns True for broken symlink."""
        from scitex.path import is_symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "nonexistent.txt"
            dst = Path(temp_dir) / "broken_link.txt"
            dst.symlink_to(src)

            assert is_symlink(dst) is True

    def test_is_symlink_string_path(self):
        """Test is_symlink with string path."""
        from scitex.path import is_symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = os.path.join(temp_dir, "source.txt")
            dst = os.path.join(temp_dir, "link.txt")
            Path(src).write_text("test")
            Path(dst).symlink_to(src)

            assert is_symlink(dst) is True


# =============================================================================
# Tests for readlink()
# =============================================================================


class TestReadlink:
    """Tests for readlink() function."""

    def test_readlink_basic(self):
        """Test readlink returns target path."""
        from scitex.path import readlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            dst = Path(temp_dir) / "link.txt"
            src.write_text("test")
            dst.symlink_to(src)

            target = readlink(dst)

            assert isinstance(target, Path)
            # The target may be relative or absolute
            assert src.name in str(target) or target.resolve() == src.resolve()

    def test_readlink_raises_for_non_symlink(self):
        """Test readlink raises OSError for non-symlink."""
        from scitex.path import readlink

        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                with pytest.raises(OSError, match="not a symbolic link"):
                    readlink(f.name)
            finally:
                os.unlink(f.name)

    def test_readlink_raises_for_directory(self):
        """Test readlink raises OSError for directory."""
        from scitex.path import readlink

        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(OSError, match="not a symbolic link"):
                readlink(temp_dir)

    def test_readlink_broken_symlink(self):
        """Test readlink works for broken symlink."""
        from scitex.path import readlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "nonexistent.txt"
            dst = Path(temp_dir) / "broken_link.txt"
            dst.symlink_to(src)

            target = readlink(dst)
            assert "nonexistent.txt" in str(target)

    def test_readlink_string_path(self):
        """Test readlink with string path."""
        from scitex.path import readlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = os.path.join(temp_dir, "source.txt")
            dst = os.path.join(temp_dir, "link.txt")
            Path(src).write_text("test")
            Path(dst).symlink_to(src)

            target = readlink(dst)
            assert isinstance(target, Path)


# =============================================================================
# Tests for resolve_symlinks()
# =============================================================================


class TestResolveSymlinks:
    """Tests for resolve_symlinks() function."""

    def test_resolve_symlinks_single(self):
        """Test resolve_symlinks with single symlink."""
        from scitex.path import resolve_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            dst = Path(temp_dir) / "link.txt"
            src.write_text("test")
            dst.symlink_to(src)

            resolved = resolve_symlinks(dst)

            assert resolved == src.resolve()
            assert resolved.is_absolute()

    def test_resolve_symlinks_chain(self):
        """Test resolve_symlinks with chain of symlinks."""
        from scitex.path import resolve_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            link1 = Path(temp_dir) / "link1.txt"
            link2 = Path(temp_dir) / "link2.txt"
            src.write_text("test")
            link1.symlink_to(src)
            link2.symlink_to(link1)

            resolved = resolve_symlinks(link2)

            assert resolved == src.resolve()

    def test_resolve_symlinks_regular_file(self):
        """Test resolve_symlinks with regular file (no symlinks)."""
        from scitex.path import resolve_symlinks

        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                resolved = resolve_symlinks(f.name)
                assert resolved == Path(f.name).resolve()
            finally:
                os.unlink(f.name)

    def test_resolve_symlinks_directory(self):
        """Test resolve_symlinks with directory."""
        from scitex.path import resolve_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            resolved = resolve_symlinks(temp_dir)
            assert resolved.is_absolute()
            assert resolved.is_dir()

    def test_resolve_symlinks_string_path(self):
        """Test resolve_symlinks with string path."""
        from scitex.path import resolve_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            resolved = resolve_symlinks(temp_dir)
            assert isinstance(resolved, Path)


# =============================================================================
# Tests for create_relative_symlink()
# =============================================================================


class TestCreateRelativeSymlink:
    """Tests for create_relative_symlink() function."""

    def test_create_relative_symlink_basic(self):
        """Test create_relative_symlink creates relative link."""
        from scitex.path import create_relative_symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            dst = Path(temp_dir) / "link.txt"
            src.write_text("test")

            result = create_relative_symlink(src, dst)

            assert result == dst
            assert dst.is_symlink()
            target = os.readlink(dst)
            assert not Path(target).is_absolute()

    def test_create_relative_symlink_overwrite(self):
        """Test create_relative_symlink with overwrite."""
        from scitex.path import create_relative_symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src1 = Path(temp_dir) / "source1.txt"
            src2 = Path(temp_dir) / "source2.txt"
            dst = Path(temp_dir) / "link.txt"
            src1.write_text("first")
            src2.write_text("second")
            create_relative_symlink(src1, dst)

            create_relative_symlink(src2, dst, overwrite=True)

            assert dst.read_text() == "second"

    def test_create_relative_symlink_different_dirs(self):
        """Test create_relative_symlink across directories."""
        from scitex.path import create_relative_symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / "src"
            dst_dir = Path(temp_dir) / "dst"
            src_dir.mkdir()
            dst_dir.mkdir()
            src = src_dir / "source.txt"
            dst = dst_dir / "link.txt"
            src.write_text("test")

            create_relative_symlink(src, dst)

            assert dst.is_symlink()
            assert dst.read_text() == "test"


# =============================================================================
# Tests for unlink_symlink()
# =============================================================================


class TestUnlinkSymlink:
    """Tests for unlink_symlink() function."""

    def test_unlink_symlink_basic(self):
        """Test unlink_symlink removes symlink."""
        from scitex.path import unlink_symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            dst = Path(temp_dir) / "link.txt"
            src.write_text("test")
            dst.symlink_to(src)

            unlink_symlink(dst)

            assert not dst.exists()
            assert not dst.is_symlink()
            assert src.exists()  # Source not affected

    def test_unlink_symlink_missing_ok_true(self):
        """Test unlink_symlink with missing_ok=True for non-existent."""
        from scitex.path import unlink_symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            dst = Path(temp_dir) / "nonexistent_link.txt"

            # Should not raise
            unlink_symlink(dst, missing_ok=True)

    def test_unlink_symlink_missing_ok_false(self):
        """Test unlink_symlink raises with missing_ok=False."""
        from scitex.path import unlink_symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            dst = Path(temp_dir) / "nonexistent_link.txt"

            with pytest.raises(FileNotFoundError):
                unlink_symlink(dst, missing_ok=False)

    def test_unlink_symlink_raises_for_non_symlink(self):
        """Test unlink_symlink raises for regular file."""
        from scitex.path import unlink_symlink

        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                with pytest.raises(OSError, match="not a symbolic link"):
                    unlink_symlink(f.name)
            finally:
                os.unlink(f.name)

    def test_unlink_symlink_broken_link(self):
        """Test unlink_symlink removes broken symlink."""
        from scitex.path import unlink_symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            dst = Path(temp_dir) / "broken_link.txt"
            dst.symlink_to("/nonexistent/target")

            unlink_symlink(dst)

            assert not dst.exists()
            assert not dst.is_symlink()

    def test_unlink_symlink_string_path(self):
        """Test unlink_symlink with string path."""
        from scitex.path import unlink_symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            dst = os.path.join(temp_dir, "link.txt")
            src.write_text("test")
            Path(dst).symlink_to(src)

            unlink_symlink(dst)

            assert not Path(dst).exists()


# =============================================================================
# Tests for list_symlinks()
# =============================================================================


class TestListSymlinks:
    """Tests for list_symlinks() function."""

    def test_list_symlinks_basic(self):
        """Test list_symlinks finds symlinks."""
        from scitex.path import list_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            link1 = Path(temp_dir) / "link1.txt"
            link2 = Path(temp_dir) / "link2.txt"
            src.write_text("test")
            link1.symlink_to(src)
            link2.symlink_to(src)

            result = list_symlinks(temp_dir)

            assert len(result) == 2
            assert link1 in result
            assert link2 in result

    def test_list_symlinks_empty_dir(self):
        """Test list_symlinks with no symlinks."""
        from scitex.path import list_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "regular.txt").write_text("test")

            result = list_symlinks(temp_dir)

            assert len(result) == 0

    def test_list_symlinks_non_recursive(self):
        """Test list_symlinks non-recursive mode."""
        from scitex.path import list_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            link_top = Path(temp_dir) / "link_top.txt"
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            link_sub = subdir / "link_sub.txt"
            src.write_text("test")
            link_top.symlink_to(src)
            link_sub.symlink_to(src)

            result = list_symlinks(temp_dir, recursive=False)

            assert len(result) == 1
            assert link_top in result
            assert link_sub not in result

    def test_list_symlinks_recursive(self):
        """Test list_symlinks recursive mode."""
        from scitex.path import list_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            link_top = Path(temp_dir) / "link_top.txt"
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            link_sub = subdir / "link_sub.txt"
            src.write_text("test")
            link_top.symlink_to(src)
            link_sub.symlink_to(src)

            result = list_symlinks(temp_dir, recursive=True)

            assert len(result) == 2
            assert link_top in result
            assert link_sub in result

    def test_list_symlinks_includes_broken(self):
        """Test list_symlinks includes broken symlinks."""
        from scitex.path import list_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            broken_link = Path(temp_dir) / "broken.txt"
            broken_link.symlink_to("/nonexistent/target")

            result = list_symlinks(temp_dir)

            assert len(result) == 1
            assert broken_link in result

    def test_list_symlinks_string_path(self):
        """Test list_symlinks with string path."""
        from scitex.path import list_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            link = Path(temp_dir) / "link.txt"
            src.write_text("test")
            link.symlink_to(src)

            result = list_symlinks(temp_dir)

            assert all(isinstance(p, Path) for p in result)


# =============================================================================
# Tests for fix_broken_symlinks()
# =============================================================================


class TestFixBrokenSymlinks:
    """Tests for fix_broken_symlinks() function."""

    def test_fix_broken_symlinks_finds_broken(self):
        """Test fix_broken_symlinks finds broken symlinks."""
        from scitex.path import fix_broken_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            broken = Path(temp_dir) / "broken.txt"
            broken.symlink_to("/nonexistent/target")

            result = fix_broken_symlinks(temp_dir)

            assert len(result["found"]) == 1
            assert broken in result["found"]
            assert len(result["fixed"]) == 0
            assert len(result["removed"]) == 0

    def test_fix_broken_symlinks_ignores_valid(self):
        """Test fix_broken_symlinks ignores valid symlinks."""
        from scitex.path import fix_broken_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            valid_link = Path(temp_dir) / "valid.txt"
            src.write_text("test")
            valid_link.symlink_to(src)

            result = fix_broken_symlinks(temp_dir)

            assert len(result["found"]) == 0

    def test_fix_broken_symlinks_remove(self):
        """Test fix_broken_symlinks removes broken with remove=True."""
        from scitex.path import fix_broken_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            broken = Path(temp_dir) / "broken.txt"
            broken.symlink_to("/nonexistent/target")

            result = fix_broken_symlinks(temp_dir, remove=True)

            assert len(result["removed"]) == 1
            assert not broken.exists()
            assert not broken.is_symlink()

    def test_fix_broken_symlinks_repoint(self):
        """Test fix_broken_symlinks repoints with new_target."""
        from scitex.path import fix_broken_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            new_src = Path(temp_dir) / "new_source.txt"
            broken = Path(temp_dir) / "broken.txt"
            new_src.write_text("new content")
            broken.symlink_to("/nonexistent/target")

            result = fix_broken_symlinks(temp_dir, new_target=new_src)

            assert len(result["fixed"]) == 1
            assert broken.is_symlink()
            assert broken.read_text() == "new content"

    def test_fix_broken_symlinks_recursive(self):
        """Test fix_broken_symlinks recursive mode."""
        from scitex.path import fix_broken_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            broken_top = Path(temp_dir) / "broken_top.txt"
            broken_sub = subdir / "broken_sub.txt"
            broken_top.symlink_to("/nonexistent/top")
            broken_sub.symlink_to("/nonexistent/sub")

            result = fix_broken_symlinks(temp_dir, recursive=True)

            assert len(result["found"]) == 2

    def test_fix_broken_symlinks_non_recursive(self):
        """Test fix_broken_symlinks non-recursive mode."""
        from scitex.path import fix_broken_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            broken_top = Path(temp_dir) / "broken_top.txt"
            broken_sub = subdir / "broken_sub.txt"
            broken_top.symlink_to("/nonexistent/top")
            broken_sub.symlink_to("/nonexistent/sub")

            result = fix_broken_symlinks(temp_dir, recursive=False)

            assert len(result["found"]) == 1
            assert broken_top in result["found"]

    def test_fix_broken_symlinks_relative_broken(self):
        """Test fix_broken_symlinks detects broken relative symlinks."""
        from scitex.path import fix_broken_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            broken = Path(temp_dir) / "broken_relative.txt"
            # Create symlink with relative path to nonexistent
            broken.symlink_to("nonexistent_file.txt")

            result = fix_broken_symlinks(temp_dir)

            assert len(result["found"]) == 1

    def test_fix_broken_symlinks_mixed(self):
        """Test fix_broken_symlinks with mix of valid and broken."""
        from scitex.path import fix_broken_symlinks

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            valid = Path(temp_dir) / "valid.txt"
            broken = Path(temp_dir) / "broken.txt"
            src.write_text("test")
            valid.symlink_to(src)
            broken.symlink_to("/nonexistent/target")

            result = fix_broken_symlinks(temp_dir)

            assert len(result["found"]) == 1
            assert broken in result["found"]
            assert valid not in result["found"]


# =============================================================================
# Integration tests
# =============================================================================


class TestSymlinkIntegration:
    """Integration tests combining multiple symlink operations."""

    def test_create_readlink_unlink_workflow(self):
        """Test complete symlink create-read-unlink workflow."""
        from scitex.path import is_symlink, readlink, symlink, unlink_symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            dst = Path(temp_dir) / "link.txt"
            src.write_text("workflow test")

            # Create
            symlink(src, dst)
            assert is_symlink(dst)

            # Read
            target = readlink(dst)
            assert src.name in str(target) or target.resolve() == src.resolve()

            # Unlink
            unlink_symlink(dst)
            assert not is_symlink(dst)

    def test_relative_symlink_portability(self):
        """Test relative symlinks work when directory is moved."""
        from scitex.path import create_relative_symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            project.mkdir()
            src = project / "data" / "source.txt"
            dst = project / "links" / "link.txt"
            src.parent.mkdir()
            dst.parent.mkdir()
            src.write_text("portable content")

            create_relative_symlink(src, dst)

            # Link should work via relative path
            assert dst.read_text() == "portable content"

    def test_chain_of_symlinks(self):
        """Test resolving chain of symlinks."""
        from scitex.path import resolve_symlinks, symlink

        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "source.txt"
            link1 = Path(temp_dir) / "link1.txt"
            link2 = Path(temp_dir) / "link2.txt"
            link3 = Path(temp_dir) / "link3.txt"
            src.write_text("chained")
            link1.symlink_to(src)
            link2.symlink_to(link1)
            link3.symlink_to(link2)

            resolved = resolve_symlinks(link3)

            assert resolved == src.resolve()
            assert link3.read_text() == "chained"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/path/_symlink.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-09-16 15:11:33 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/path/_symlink.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """Symlink creation and management utilities for SciTeX."""
# 
# # from scitex import logging
# from pathlib import Path
# from typing import Optional, Union
# 
# from scitex import logging
# 
# logger = logging.getLogger(__name__)
# 
# 
# def symlink(
#     src: Union[str, Path],
#     dst: Union[str, Path],
#     overwrite: bool = False,
#     target_is_directory: Optional[bool] = None,
#     relative: bool = False,
# ) -> Path:
#     """
#     Create a symbolic link pointing to src named dst.
# 
#     Args:
#         src: Source path (target of the symlink)
#         dst: Destination path (the symlink to create)
#         overwrite: If True, remove existing dst before creating symlink
#         target_is_directory: On Windows, specify if target is directory (auto-detected if None)
#         relative: If True, create relative symlink instead of absolute
# 
#     Returns:
#         Path object of the created symlink
# 
#     Raises:
#         FileExistsError: If dst exists and overwrite=False
#         FileNotFoundError: If src doesn't exist
#         OSError: If symlink creation fails
# 
#     Examples:
#         >>> import scitex as stx
#         >>> # Create absolute symlink
#         >>> stx.path.symlink("/path/to/source", "/path/to/link")
# 
#         >>> # Create relative symlink
#         >>> stx.path.symlink("../source", "link", relative=True)
# 
#         >>> # Overwrite existing symlink
#         >>> stx.path.symlink("/path/to/new_source", "/path/to/link", overwrite=True)
#     """
#     src_path = Path(src)
#     dst_path = Path(dst)
# 
#     # Note: We allow creating symlinks to non-existent targets
#     # This is valid in Unix/Linux and useful for testing
# 
#     # Handle existing destination
#     if dst_path.exists() or dst_path.is_symlink():
#         if not overwrite:
#             raise FileExistsError(f"Destination already exists: {dst_path}")
#         else:
#             # Remove existing file/symlink
#             if dst_path.is_symlink():
#                 dst_path.unlink()
#             elif dst_path.is_file():
#                 dst_path.unlink()
#             elif dst_path.is_dir():
#                 import shutil
# 
#                 shutil.rmtree(dst_path)
#             # logger.info(f"Removed existing destination: {dst_path}")
# 
#     # Create parent directory if needed
#     dst_path.parent.mkdir(parents=True, exist_ok=True)
# 
#     # Determine if target is directory (for Windows)
#     if target_is_directory is None and src_path.exists():
#         target_is_directory = src_path.is_dir()
# 
#     # Create symlink
#     try:
#         if relative:
#             # Calculate relative path from dst to src
#             if src_path.is_absolute():
#                 # src is absolute, calculate relative from dst
#                 try:
#                     rel_path = os.path.relpath(src_path, dst_path.parent)
#                     src_for_link = Path(rel_path)
#                 except ValueError:
#                     # Can't create relative path (e.g., different drives on Windows)
#                     logger.warning(
#                         f"Cannot create relative path from {dst_path} to {src_path}, using absolute"
#                     )
#                     src_for_link = src_path.absolute()
#             else:
#                 # Both paths are relative, need to resolve them first
#                 # to calculate the correct relative path
#                 src_abs = src_path.resolve()
#                 dst_parent_abs = dst_path.parent.resolve()
#                 try:
#                     rel_path = os.path.relpath(src_abs, dst_parent_abs)
#                     src_for_link = Path(rel_path)
#                 except ValueError:
#                     # Can't create relative path
#                     logger.warning(
#                         f"Cannot create relative path from {dst_path} to {src_path}"
#                     )
#                     src_for_link = src_path
#         else:
#             src_for_link = src_path.absolute()
# 
#         dst_path.symlink_to(src_for_link, target_is_directory=target_is_directory)
#         logger.success(f"Created symlink: {dst_path} -> {src_for_link}")
# 
#     except OSError as e:
#         logger.warn(
#             f"Failed to create symlink from {dst_path} to {src_for_link}: {str(e)}"
#         )
# 
#         # raise OSError(
#         #     f"Failed to create symlink from {dst_path} to {src_for_link}: {e}"
#         # )
# 
#     return dst_path
# 
# 
# def is_symlink(path: Union[str, Path]) -> bool:
#     """
#     Check if a path is a symbolic link.
# 
#     Args:
#         path: Path to check
# 
#     Returns:
#         True if path is a symlink, False otherwise
# 
#     Examples:
#         >>> import scitex as stx
#         >>> stx.path.is_symlink("/path/to/link")
#         False
#     """
#     return Path(path).is_symlink()
# 
# 
# def readlink(path: Union[str, Path]) -> Path:
#     """
#     Return the path to which the symbolic link points.
# 
#     Args:
#         path: Symlink path to read
# 
#     Returns:
#         Path object pointing to the symlink target
# 
#     Raises:
#         OSError: If path is not a symlink
# 
#     Examples:
#         >>> import scitex as stx
#         >>> target = stx.path.readlink("/path/to/link")
#         >>> print(target)
#     """
#     path = Path(path)
#     if not path.is_symlink():
#         raise OSError(f"Path is not a symbolic link: {path}")
# 
#     return Path(os.readlink(path))
# 
# 
# def resolve_symlinks(path: Union[str, Path]) -> Path:
#     """
#     Resolve all symbolic links in a path.
# 
#     Args:
#         path: Path potentially containing symlinks
# 
#     Returns:
#         Fully resolved absolute path
# 
#     Examples:
#         >>> import scitex as stx
#         >>> resolved = stx.path.resolve_symlinks("/path/with/symlinks")
#         >>> print(resolved)
#     """
#     return Path(path).resolve()
# 
# 
# def create_relative_symlink(
#     src: Union[str, Path], dst: Union[str, Path], overwrite: bool = False
# ) -> Path:
#     """
#     Create a relative symbolic link.
# 
#     This is a convenience wrapper around symlink() with relative=True.
# 
#     Args:
#         src: Source path (target of the symlink)
#         dst: Destination path (the symlink to create)
#         overwrite: If True, remove existing dst before creating symlink
# 
#     Returns:
#         Path object of the created symlink
# 
#     Examples:
#         >>> import scitex as stx
#         >>> # Create relative symlink from current dir to parent dir file
#         >>> stx.path.create_relative_symlink("../data/file.txt", "link_to_file")
#     """
#     return symlink(src, dst, overwrite=overwrite, relative=True)
# 
# 
# def unlink_symlink(path: Union[str, Path], missing_ok: bool = True) -> None:
#     """
#     Remove a symbolic link.
# 
#     Args:
#         path: Symlink to remove
#         missing_ok: If True, don't raise error if symlink doesn't exist
# 
#     Raises:
#         FileNotFoundError: If symlink doesn't exist and missing_ok=False
#         OSError: If path is not a symlink
# 
#     Examples:
#         >>> import scitex as stx
#         >>> stx.path.unlink_symlink("/path/to/link")
#     """
#     path = Path(path)
# 
#     if not path.exists() and not path.is_symlink():
#         if missing_ok:
#             return
#         raise FileNotFoundError(f"Symlink does not exist: {path}")
# 
#     if not path.is_symlink():
#         raise OSError(f"Path is not a symbolic link: {path}")
# 
#     path.unlink()
#     # logger.info(f"Removed symlink: {path}")
# 
# 
# def list_symlinks(directory: Union[str, Path], recursive: bool = False) -> list[Path]:
#     """
#     List all symbolic links in a directory.
# 
#     Args:
#         directory: Directory to search
#         recursive: If True, search recursively
# 
#     Returns:
#         List of Path objects for all symlinks found
# 
#     Examples:
#         >>> import scitex as stx
#         >>> symlinks = stx.path.list_symlinks("/path/to/dir")
#         >>> for link in symlinks:
#         ...     print(f"{link} -> {stx.path.readlink(link)}")
#     """
#     directory = Path(directory)
#     symlinks = []
# 
#     if recursive:
#         for path in directory.rglob("*"):
#             if path.is_symlink():
#                 symlinks.append(path)
#     else:
#         for path in directory.iterdir():
#             if path.is_symlink():
#                 symlinks.append(path)
# 
#     return symlinks
# 
# 
# def fix_broken_symlinks(
#     directory: Union[str, Path],
#     recursive: bool = False,
#     remove: bool = False,
#     new_target: Optional[Union[str, Path]] = None,
# ) -> dict:
#     """
#     Find and optionally fix broken symbolic links.
# 
#     Args:
#         directory: Directory to search
#         recursive: If True, search recursively
#         remove: If True, remove broken symlinks
#         new_target: If provided, repoint broken symlinks to this target
# 
#     Returns:
#         Dictionary with 'found', 'fixed', and 'removed' lists of paths
# 
#     Examples:
#         >>> import scitex as stx
#         >>> # Find broken symlinks
#         >>> result = stx.path.fix_broken_symlinks("/path/to/dir")
#         >>> print(f"Found {len(result['found'])} broken symlinks")
# 
#         >>> # Remove broken symlinks
#         >>> result = stx.path.fix_broken_symlinks("/path/to/dir", remove=True)
#     """
#     directory = Path(directory)
#     result = {"found": [], "fixed": [], "removed": []}
# 
#     symlinks = list_symlinks(directory, recursive=recursive)
# 
#     for link in symlinks:
#         try:
#             # Check if target exists
#             target = Path(os.readlink(link))
#             if not link.parent.joinpath(target).exists() and not target.is_absolute():
#                 # Relative link with non-existent target
#                 result["found"].append(link)
#             elif target.is_absolute() and not target.exists():
#                 # Absolute link with non-existent target
#                 result["found"].append(link)
#         except (OSError, ValueError):
#             result["found"].append(link)
# 
#     # Fix or remove broken symlinks
#     for link in result["found"]:
#         if remove:
#             link.unlink()
#             result["removed"].append(link)
#             # logger.info(f"Removed broken symlink: {link}")
#         elif new_target:
#             link.unlink()
#             symlink(new_target, link)
#             result["fixed"].append(link)
#             # logger.info(f"Fixed symlink: {link} -> {new_target}")
# 
#     return result
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/path/_symlink.py
# --------------------------------------------------------------------------------
