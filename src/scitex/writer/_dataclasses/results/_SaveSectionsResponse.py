#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/results/_SaveSectionsResponse.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/dataclasses/results/_SaveSectionsResponse.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
SaveSectionsResponse - dataclass for section save operation results.
Ensures type safety and prevents silent failures.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class SaveSectionsResponse:
    """Result of saving multiple sections."""

    success: bool
    """Whether the save operation succeeded"""

    sections_saved: int
    """Number of sections successfully saved"""

    sections_skipped: int = 0
    """Number of sections skipped or failed"""

    message: str = ""
    """Human-readable status message"""

    errors: List[str] = field(default_factory=list)
    """List of error messages (if any)"""

    error_details: Dict[str, str] = field(default_factory=dict)
    """Detailed errors per section: {section_id: error_message}"""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "sections_saved": self.sections_saved,
            "sections_skipped": self.sections_skipped,
            "message": self.message,
            "errors": self.errors,
            "error_details": self.error_details,
        }

    def __str__(self):
        """Human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Save {status}: {self.sections_saved} saved, {self.sections_skipped} skipped",
        ]
        if self.message:
            lines.append(f"Message: {self.message}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for error in self.errors[:3]:  # Show first 3 errors
                lines.append(f"  - {error}")
        return "\n".join(lines)

    @classmethod
    def create_success(
        cls, sections_saved: int, message: str = ""
    ) -> SaveSectionsResponse:
        """Create a successful response."""
        return cls(
            success=True,
            sections_saved=sections_saved,
            sections_skipped=0,
            message=message or f"Saved {sections_saved} sections",
        )

    @classmethod
    def create_failure(
        cls, error_message: str, errors: List[str] = None, sections_saved: int = 0
    ) -> SaveSectionsResponse:
        """Create a failed response."""
        return cls(
            success=False,
            sections_saved=sections_saved,
            sections_skipped=len(errors) if errors else 0,
            message=error_message,
            errors=errors or [error_message],
        )

    def validate(self) -> None:
        """Validate response data - raises ValueError if invalid."""
        if self.success and self.sections_saved == 0:
            raise ValueError(
                "SaveSectionsResponse marked as success but no sections were saved"
            )

        if not self.success and not self.errors:
            raise ValueError(
                "SaveSectionsResponse marked as failed but no errors provided"
            )

        if self.sections_saved < 0:
            raise ValueError(
                f"Invalid sections_saved: {self.sections_saved} (must be >= 0)"
            )

        if self.sections_skipped < 0:
            raise ValueError(
                f"Invalid sections_skipped: {self.sections_skipped} (must be >= 0)"
            )


__all__ = ["SaveSectionsResponse"]

# EOF
