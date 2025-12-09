#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/results/_SectionReadResponse.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/dataclasses/results/_SectionReadResponse.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
SectionReadResponse - dataclass for section read operation results.
Ensures type safety and prevents silent failures.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class SectionReadResponse:
    """Result of reading a section."""

    success: bool
    """Whether the read operation succeeded"""

    content: str
    """Section content (LaTeX code)"""

    section_name: str
    """Section name (e.g., 'abstract', 'introduction')"""

    section_id: str
    """Full hierarchical section ID (e.g., 'manuscript/abstract')"""

    doc_type: str
    """Document type: 'manuscript', 'supplementary', 'revision', or 'shared'"""

    file_path: Optional[Path] = None
    """Absolute path to the section .tex file"""

    error: Optional[str] = None
    """Error message if read failed"""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "content": self.content,
            "section_name": self.section_name,
            "section_id": self.section_id,
            "doc_type": self.doc_type,
            "file_path": str(self.file_path) if self.file_path else None,
            "error": self.error,
        }

    def __str__(self):
        """Human-readable summary."""
        if self.success:
            return f"Read {self.section_id}: {len(self.content)} chars"
        else:
            return f"Failed to read {self.section_id}: {self.error}"

    @classmethod
    def create_success(
        cls,
        content: str,
        section_name: str,
        section_id: str,
        doc_type: str,
        file_path: Optional[Path] = None,
    ) -> SectionReadResponse:
        """Create a successful response."""
        return cls(
            success=True,
            content=content,
            section_name=section_name,
            section_id=section_id,
            doc_type=doc_type,
            file_path=file_path,
            error=None,
        )

    @classmethod
    def create_failure(cls, section_id: str, error_message: str) -> SectionReadResponse:
        """Create a failed response."""
        parts = section_id.split("/")
        doc_type = parts[0] if len(parts) > 1 else "manuscript"
        section_name = parts[1] if len(parts) > 1 else section_id

        return cls(
            success=False,
            content="",
            section_name=section_name,
            section_id=section_id,
            doc_type=doc_type,
            file_path=None,
            error=error_message,
        )

    def validate(self) -> None:
        """Validate response data - raises ValueError if invalid."""
        if (
            self.success
            and not self.content
            and self.section_name not in ["compiled_pdf", "compiled_tex"]
        ):
            raise ValueError(
                f"SectionReadResponse marked as success but content is empty for {self.section_id}"
            )

        if not self.success and not self.error:
            raise ValueError(
                f"SectionReadResponse marked as failed but no error message for {self.section_id}"
            )

        if not self.section_name:
            raise ValueError("section_name cannot be empty")

        if not self.section_id:
            raise ValueError("section_id cannot be empty")

        if self.doc_type not in ["manuscript", "supplementary", "revision", "shared"]:
            raise ValueError(f"Invalid doc_type: {self.doc_type}")


__all__ = ["SectionReadResponse"]

# EOF
