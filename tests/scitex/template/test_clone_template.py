#!/usr/bin/env python3
# Timestamp: 2026-02-08
# File: tests/scitex/template/test_clone_template.py

"""Tests for the unified clone_template dispatcher."""

from unittest.mock import MagicMock, patch

import pytest

from scitex.template._project._clone_template import (
    ALIASES,
    TEMPLATES,
    clone_template,
    get_all_template_ids,
    get_template_ids,
)


class TestCloneTemplateDispatch:
    """Test that clone_template dispatches to correct functions."""

    @pytest.mark.parametrize("template_id", list(TEMPLATES.keys()))
    def test_canonical_ids_dispatch(self, template_id):
        """Each canonical template ID dispatches to its function."""
        mock_func = MagicMock(return_value=True)
        with patch.dict(TEMPLATES, {template_id: mock_func}):
            result = clone_template(
                template_id=template_id,
                project_dir="/tmp/test-project",
            )
            assert result is True
            mock_func.assert_called_once_with(
                project_dir="/tmp/test-project",
                git_strategy="child",
                branch=None,
                tag=None,
            )

    @pytest.mark.parametrize(
        "alias,canonical",
        list(ALIASES.items()),
    )
    def test_aliases_resolve(self, alias, canonical):
        """Aliases resolve to canonical IDs."""
        mock_func = MagicMock(return_value=True)
        with patch.dict(TEMPLATES, {canonical: mock_func}):
            result = clone_template(
                template_id=alias,
                project_dir="/tmp/test-alias",
            )
            assert result is True
            mock_func.assert_called_once()

    def test_unknown_template_raises(self):
        """Unknown template ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown template"):
            clone_template(
                template_id="nonexistent",
                project_dir="/tmp/test",
            )

    def test_kwargs_forwarded(self):
        """git_strategy, branch, tag are forwarded."""
        mock_func = MagicMock(return_value=True)
        with patch.dict(TEMPLATES, {"research": mock_func}):
            clone_template(
                template_id="research",
                project_dir="/tmp/test",
                git_strategy="origin",
                branch="develop",
                tag=None,
            )
            mock_func.assert_called_once_with(
                project_dir="/tmp/test",
                git_strategy="origin",
                branch="develop",
                tag=None,
            )

    def test_git_strategy_none(self):
        """git_strategy=None is forwarded correctly."""
        mock_func = MagicMock(return_value=True)
        with patch.dict(TEMPLATES, {"research": mock_func}):
            clone_template(
                template_id="research",
                project_dir="/tmp/test",
                git_strategy=None,
            )
            mock_func.assert_called_once_with(
                project_dir="/tmp/test",
                git_strategy=None,
                branch=None,
                tag=None,
            )

    def test_return_false_propagated(self):
        """False return from clone function is propagated."""
        mock_func = MagicMock(return_value=False)
        with patch.dict(TEMPLATES, {"research": mock_func}):
            result = clone_template(
                template_id="research",
                project_dir="/tmp/test",
            )
            assert result is False


class TestTemplateIdHelpers:
    """Test helper functions for template IDs."""

    def test_get_template_ids(self):
        """get_template_ids returns canonical IDs only."""
        ids = get_template_ids()
        assert "research" in ids
        assert "research_minimal" in ids
        assert "pip_project" in ids
        assert "singularity" in ids
        assert "paper_directory" in ids
        assert "minimal" not in ids

    def test_get_all_template_ids(self):
        """get_all_template_ids includes aliases."""
        ids = get_all_template_ids()
        assert "research" in ids
        assert "minimal" in ids
        assert "pip-project" in ids
        assert "paper" in ids


class TestImportFromPackage:
    """Test that clone_template is importable from scitex.template."""

    def test_import_from_template(self):
        """clone_template is importable from scitex.template."""
        from scitex.template import clone_template as ct

        assert callable(ct)

    def test_in_all(self):
        """clone_template is in __all__."""
        import scitex.template

        assert "clone_template" in scitex.template.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF
