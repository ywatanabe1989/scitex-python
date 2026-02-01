#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/tests/scitex/verify/test__hash.py

"""Tests for scitex.verify._hash module."""

import pytest

from scitex.verify import (
    combine_hashes,
    hash_directory,
    hash_file,
    hash_files,
    verify_hash,
)


class TestHashFile:
    """Tests for hash_file function."""

    def test_hash_file_basic(self, tmp_path):
        """Test basic file hashing."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        # Hash the file
        result = hash_file(test_file)

        # Should return a 32-character hex string
        assert isinstance(result, str)
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_file_deterministic(self, tmp_path):
        """Test that same content produces same hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Same content")

        hash1 = hash_file(test_file)
        hash2 = hash_file(test_file)

        assert hash1 == hash2

    def test_hash_file_different_content(self, tmp_path):
        """Test that different content produces different hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content A")
        file2.write_text("Content B")

        hash1 = hash_file(file1)
        hash2 = hash_file(file2)

        assert hash1 != hash2

    def test_hash_file_not_found(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            hash_file(tmp_path / "nonexistent.txt")

    def test_hash_file_binary(self, tmp_path):
        """Test hashing binary files."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd")

        result = hash_file(test_file)

        assert isinstance(result, str)
        assert len(result) == 32

    def test_hash_file_empty(self, tmp_path):
        """Test hashing empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        result = hash_file(test_file)

        assert isinstance(result, str)
        assert len(result) == 32

    def test_hash_file_path_types(self, tmp_path):
        """Test that both str and Path work."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        hash_from_path = hash_file(test_file)
        hash_from_str = hash_file(str(test_file))

        assert hash_from_path == hash_from_str


class TestHashDirectory:
    """Tests for hash_directory function."""

    def test_hash_directory_basic(self, tmp_path):
        """Test basic directory hashing."""
        # Create test files
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.txt").write_text("Content 2")

        result = hash_directory(tmp_path)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert "file1.txt" in result
        assert "file2.txt" in result

    def test_hash_directory_recursive(self, tmp_path):
        """Test recursive directory hashing."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "file1.txt").write_text("Root file")
        (subdir / "file2.txt").write_text("Subdir file")

        result = hash_directory(tmp_path, recursive=True)

        assert len(result) == 2
        assert "file1.txt" in result
        assert "subdir/file2.txt" in result or "subdir\\file2.txt" in result

    def test_hash_directory_non_recursive(self, tmp_path):
        """Test non-recursive directory hashing."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "file1.txt").write_text("Root file")
        (subdir / "file2.txt").write_text("Subdir file")

        result = hash_directory(tmp_path, recursive=False)

        assert len(result) == 1
        assert "file1.txt" in result

    def test_hash_directory_pattern(self, tmp_path):
        """Test directory hashing with pattern filter."""
        (tmp_path / "data.csv").write_text("csv data")
        (tmp_path / "config.json").write_text("{}")

        result = hash_directory(tmp_path, pattern="*.csv")

        assert len(result) == 1
        assert "data.csv" in result

    def test_hash_directory_not_a_directory(self, tmp_path):
        """Test that file path raises NotADirectoryError."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(NotADirectoryError):
            hash_directory(test_file)

    def test_hash_directory_empty(self, tmp_path):
        """Test hashing empty directory."""
        result = hash_directory(tmp_path)
        assert result == {}


class TestHashFiles:
    """Tests for hash_files function."""

    def test_hash_files_basic(self, tmp_path):
        """Test hashing multiple files."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        result = hash_files([file1, file2])

        assert isinstance(result, dict)
        assert len(result) == 2
        assert str(file1) in result
        assert str(file2) in result

    def test_hash_files_missing_file(self, tmp_path):
        """Test that missing files are skipped."""
        file1 = tmp_path / "exists.txt"
        file2 = tmp_path / "missing.txt"
        file1.write_text("Content")

        result = hash_files([file1, file2])

        assert len(result) == 1
        assert str(file1) in result

    def test_hash_files_empty_list(self):
        """Test hashing empty list."""
        result = hash_files([])
        assert result == {}


class TestCombineHashes:
    """Tests for combine_hashes function."""

    def test_combine_hashes_basic(self):
        """Test basic hash combining."""
        hashes = {"file1": "abc123", "file2": "def456"}

        result = combine_hashes(hashes)

        assert isinstance(result, str)
        assert len(result) == 32

    def test_combine_hashes_deterministic(self):
        """Test that same hashes produce same combined hash."""
        hashes = {"a": "hash1", "b": "hash2"}

        result1 = combine_hashes(hashes)
        result2 = combine_hashes(hashes)

        assert result1 == result2

    def test_combine_hashes_order_independent(self):
        """Test that key order doesn't matter (sorted internally)."""
        hashes1 = {"a": "hash1", "b": "hash2"}
        hashes2 = {"b": "hash2", "a": "hash1"}

        result1 = combine_hashes(hashes1)
        result2 = combine_hashes(hashes2)

        assert result1 == result2

    def test_combine_hashes_different_content(self):
        """Test that different hashes produce different combined hash."""
        hashes1 = {"a": "hash1", "b": "hash2"}
        hashes2 = {"a": "hash1", "b": "hash3"}

        result1 = combine_hashes(hashes1)
        result2 = combine_hashes(hashes2)

        assert result1 != result2

    def test_combine_hashes_empty(self):
        """Test combining empty dict."""
        result = combine_hashes({})
        assert isinstance(result, str)
        assert len(result) == 32


class TestVerifyHash:
    """Tests for verify_hash function."""

    def test_verify_hash_match(self, tmp_path):
        """Test verification when hash matches."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        expected = hash_file(test_file)
        result = verify_hash(test_file, expected)

        assert result is True

    def test_verify_hash_mismatch(self, tmp_path):
        """Test verification when hash doesn't match."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        result = verify_hash(test_file, "wronghash12345678901234567890ab")

        assert result is False

    def test_verify_hash_missing_file(self, tmp_path):
        """Test verification of missing file returns False."""
        result = verify_hash(tmp_path / "missing.txt", "somehash")
        assert result is False

    def test_verify_hash_truncated(self, tmp_path):
        """Test verification with truncated hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        full_hash = hash_file(test_file)
        short_hash = full_hash[:16]

        result = verify_hash(test_file, short_hash)
        assert result is True


# EOF
