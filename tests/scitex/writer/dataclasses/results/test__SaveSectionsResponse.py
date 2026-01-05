#!/usr/bin/env python3
"""Tests for scitex.writer.dataclasses.results._SaveSectionsResponse."""

import pytest

from scitex.writer.dataclasses.results._SaveSectionsResponse import SaveSectionsResponse


class TestSaveSectionsResponseCreation:
    """Tests for SaveSectionsResponse instantiation."""

    def test_required_fields(self):
        """Verify required fields are set correctly."""
        response = SaveSectionsResponse(success=True, sections_saved=5)
        assert response.success is True
        assert response.sections_saved == 5

    def test_optional_fields_defaults(self):
        """Verify optional fields have proper defaults."""
        response = SaveSectionsResponse(success=True, sections_saved=1)
        assert response.sections_skipped == 0
        assert response.message == ""
        assert response.errors == []
        assert response.error_details == {}

    def test_all_fields_can_be_set(self):
        """Verify all fields can be explicitly set."""
        response = SaveSectionsResponse(
            success=False,
            sections_saved=2,
            sections_skipped=3,
            message="Partial save",
            errors=["Error 1"],
            error_details={"abstract": "File not found"},
        )
        assert response.success is False
        assert response.sections_saved == 2
        assert response.sections_skipped == 3
        assert response.message == "Partial save"
        assert response.errors == ["Error 1"]
        assert response.error_details == {"abstract": "File not found"}


class TestSaveSectionsResponseFactoryMethods:
    """Tests for SaveSectionsResponse factory methods."""

    def test_create_success_basic(self):
        """Verify create_success factory method."""
        response = SaveSectionsResponse.create_success(5)
        assert response.success is True
        assert response.sections_saved == 5
        assert response.sections_skipped == 0
        assert "5 sections" in response.message

    def test_create_success_with_message(self):
        """Verify create_success with custom message."""
        response = SaveSectionsResponse.create_success(3, "Custom message")
        assert response.message == "Custom message"

    def test_create_failure_basic(self):
        """Verify create_failure factory method."""
        response = SaveSectionsResponse.create_failure("Save failed")
        assert response.success is False
        assert response.message == "Save failed"
        assert "Save failed" in response.errors

    def test_create_failure_with_errors_list(self):
        """Verify create_failure with error list."""
        errors = ["Error 1", "Error 2", "Error 3"]
        response = SaveSectionsResponse.create_failure("Multiple errors", errors=errors)
        assert response.errors == errors
        assert response.sections_skipped == 3


class TestSaveSectionsResponseToDict:
    """Tests for SaveSectionsResponse to_dict method."""

    def test_to_dict_contains_all_fields(self):
        """Verify to_dict includes all fields."""
        response = SaveSectionsResponse(
            success=True,
            sections_saved=5,
            sections_skipped=1,
            message="Done",
            errors=["Error"],
            error_details={"intro": "Failed"},
        )
        result = response.to_dict()

        assert result["success"] is True
        assert result["sections_saved"] == 5
        assert result["sections_skipped"] == 1
        assert result["message"] == "Done"
        assert result["errors"] == ["Error"]
        assert result["error_details"] == {"intro": "Failed"}


class TestSaveSectionsResponseStr:
    """Tests for SaveSectionsResponse __str__ method."""

    def test_str_success(self):
        """Verify string representation for success."""
        response = SaveSectionsResponse.create_success(5)
        str_result = str(response)
        assert "SUCCESS" in str_result
        assert "5 saved" in str_result

    def test_str_failure(self):
        """Verify string representation for failure."""
        response = SaveSectionsResponse.create_failure("Failed", errors=["E1", "E2"])
        str_result = str(response)
        assert "FAILED" in str_result
        assert "2 skipped" in str_result


class TestSaveSectionsResponseValidation:
    """Tests for SaveSectionsResponse validate method."""

    def test_validate_success_with_no_sections_raises(self):
        """Verify validate raises for success with 0 sections."""
        response = SaveSectionsResponse(success=True, sections_saved=0)
        with pytest.raises(ValueError, match="success but no sections"):
            response.validate()

    def test_validate_failure_with_no_errors_raises(self):
        """Verify validate raises for failure with no errors."""
        response = SaveSectionsResponse(success=False, sections_saved=0, errors=[])
        with pytest.raises(ValueError, match="failed but no errors"):
            response.validate()

    def test_validate_negative_sections_saved_raises(self):
        """Verify validate raises for negative sections_saved."""
        response = SaveSectionsResponse(success=True, sections_saved=-1)
        with pytest.raises(ValueError, match="Invalid sections_saved"):
            response.validate()

    def test_validate_negative_sections_skipped_raises(self):
        """Verify validate raises for negative sections_skipped."""
        response = SaveSectionsResponse(
            success=True, sections_saved=1, sections_skipped=-1
        )
        with pytest.raises(ValueError, match="Invalid sections_skipped"):
            response.validate()

    def test_validate_valid_success_passes(self):
        """Verify validate passes for valid success response."""
        response = SaveSectionsResponse.create_success(5)
        response.validate()

    def test_validate_valid_failure_passes(self):
        """Verify validate passes for valid failure response."""
        response = SaveSectionsResponse.create_failure("Error occurred")
        response.validate()


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
