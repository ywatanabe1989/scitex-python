#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-11 16:00:00
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/msword/test_profiles.py

"""Tests for scitex.msword.profiles module."""

import pytest


class TestListProfiles:
    """Tests for list_profiles function."""

    def test_list_profiles_returns_list(self):
        """list_profiles should return a list."""
        from scitex.msword import list_profiles

        profiles = list_profiles()
        assert isinstance(profiles, list)

    def test_list_profiles_contains_generic(self):
        """list_profiles should include 'generic' profile."""
        from scitex.msword import list_profiles

        profiles = list_profiles()
        assert "generic" in profiles

    def test_list_profiles_contains_expected_journals(self):
        """list_profiles should include major journal profiles."""
        from scitex.msword import list_profiles

        profiles = list_profiles()
        expected = ["mdpi-ijerph", "resna-2025", "iop-double-anonymous", "ieee"]
        for profile_name in expected:
            assert profile_name in profiles, f"Missing profile: {profile_name}"

    def test_list_profiles_sorted(self):
        """list_profiles should return sorted list."""
        from scitex.msword import list_profiles

        profiles = list_profiles()
        assert profiles == sorted(profiles)


class TestGetProfile:
    """Tests for get_profile function."""

    def test_get_profile_generic(self):
        """get_profile('generic') should return a valid profile."""
        from scitex.msword import get_profile

        profile = get_profile("generic")
        assert profile.name == "generic"
        assert profile.heading_styles is not None

    def test_get_profile_none_returns_generic(self):
        """get_profile(None) should return generic profile."""
        from scitex.msword import get_profile

        profile = get_profile(None)
        assert profile.name == "generic"

    def test_get_profile_unknown_raises_keyerror(self):
        """get_profile with unknown name should raise KeyError."""
        from scitex.msword import get_profile

        with pytest.raises(KeyError) as exc_info:
            get_profile("unknown-profile-xyz")

        assert "unknown-profile-xyz" in str(exc_info.value)

    def test_get_profile_mdpi(self):
        """get_profile('mdpi-ijerph') should return MDPI profile."""
        from scitex.msword import get_profile

        profile = get_profile("mdpi-ijerph")
        assert profile.name == "mdpi-ijerph"
        assert profile.columns == 1

    def test_get_profile_resna(self):
        """get_profile('resna-2025') should return RESNA profile."""
        from scitex.msword import get_profile

        profile = get_profile("resna-2025")
        assert profile.name == "resna-2025"
        assert profile.columns == 2

    def test_get_profile_iop_double_anonymous(self):
        """get_profile('iop-double-anonymous') should return IOP profile."""
        from scitex.msword import get_profile

        profile = get_profile("iop-double-anonymous")
        assert profile.name == "iop-double-anonymous"
        assert profile.double_anonymous is True

    def test_get_profile_alias_mdpi(self):
        """get_profile('mdpi') should work as alias."""
        from scitex.msword import get_profile

        profile = get_profile("mdpi")
        assert "mdpi" in profile.name.lower()


class TestBaseWordProfile:
    """Tests for BaseWordProfile dataclass."""

    def test_base_word_profile_defaults(self):
        """BaseWordProfile should have sensible defaults."""
        from scitex.msword import BaseWordProfile

        profile = BaseWordProfile(name="test", description="Test profile")

        assert profile.name == "test"
        assert profile.caption_style == "Caption"
        assert profile.normal_style == "Normal"
        assert profile.columns == 1
        assert profile.double_anonymous is False

    def test_base_word_profile_heading_styles(self):
        """BaseWordProfile should accept custom heading styles."""
        from scitex.msword import BaseWordProfile

        profile = BaseWordProfile(
            name="custom",
            description="Custom profile",
            heading_styles={1: "Title", 2: "Subtitle", 3: "H3"},
        )

        assert profile.heading_styles[1] == "Title"
        assert profile.heading_styles[2] == "Subtitle"

    def test_base_word_profile_reference_section_titles(self):
        """BaseWordProfile should have reference section titles."""
        from scitex.msword import BaseWordProfile

        profile = BaseWordProfile(name="test", description="Test")
        assert "References" in profile.reference_section_titles

    def test_base_word_profile_figure_caption_prefixes(self):
        """BaseWordProfile should have figure caption prefixes."""
        from scitex.msword import BaseWordProfile

        profile = BaseWordProfile(name="test", description="Test")
        assert "Figure" in profile.figure_caption_prefixes
        assert "Fig." in profile.figure_caption_prefixes


class TestRegisterProfile:
    """Tests for register_profile function."""

    def test_register_profile(self):
        """register_profile should add a custom profile."""
        from scitex.msword import BaseWordProfile, register_profile, list_profiles

        custom = BaseWordProfile(
            name="test-custom-journal",
            description="Test custom journal",
            heading_styles={1: "Section", 2: "Subsection"},
        )

        register_profile(custom)
        profiles = list_profiles()
        assert "test-custom-journal" in profiles

    def test_register_profile_retrievable(self):
        """Registered profile should be retrievable via get_profile."""
        from scitex.msword import (
            BaseWordProfile,
            register_profile,
            get_profile,
        )

        custom = BaseWordProfile(
            name="test-retrievable-profile",
            description="Test retrievable profile",
            columns=2,
        )

        register_profile(custom)
        retrieved = get_profile("test-retrievable-profile")
        assert retrieved.name == "test-retrievable-profile"
        assert retrieved.columns == 2

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/msword/profiles.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: 2025-12-11 15:15:00
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/msword/profiles.py
# 
# """
# Profiles for mapping MS Word styles to SciTeX writer structures.
# 
# Each profile corresponds to a journal / conference template, such as:
# - "generic"
# - "mdpi-ijerph"
# - "resna-2025"
# - "iop-double-anonymous"
# 
# The profiles define:
# - Which Word style names correspond to section headings
# - How to detect captions for figures and tables
# - How to handle references, lists, equations, etc.
# - Layout settings (columns, margins, fonts)
# """
# 
# from __future__ import annotations
# 
# from dataclasses import dataclass, field
# from typing import Dict, List, Optional, Callable
# 
# 
# @dataclass
# class BaseWordProfile:
#     """
#     Base configuration for mapping between DOCX and SciTeX writer documents.
# 
#     Attributes
#     ----------
#     name : str
#         Profile identifier (e.g., "mdpi-ijerph").
#     description : str
#         Human-readable description.
#     heading_styles : dict[int, str]
#         Mapping from section depth (1, 2, 3...) to Word style names
#         (e.g., {1: "Heading 1", 2: "Heading 2"}).
#     caption_style : str
#         Word style name used for figure/table captions.
#     normal_style : str
#         Default paragraph style.
#     reference_section_titles : list[str]
#         Titles that indicate the start of the reference section.
#     figure_caption_prefixes : list[str]
#         Prefixes that identify figure captions (e.g., ["Figure", "Fig."]).
#     table_caption_prefixes : list[str]
#         Prefixes that identify table captions (e.g., ["Table"]).
#     list_styles : dict[str, str]
#         Mapping for list styles (bullet, numbered).
#     equation_style : str | None
#         Style name for equations, if any.
#     columns : int
#         Number of columns in the layout (1 or 2).
#     double_anonymous : bool
#         Whether this profile requires double-anonymous formatting.
#     """
# 
#     name: str
#     description: str
#     heading_styles: Dict[int, str] = field(default_factory=dict)
#     caption_style: str = "Caption"
#     normal_style: str = "Normal"
#     reference_section_titles: List[str] = field(
#         default_factory=lambda: ["References", "REFERENCES"]
#     )
#     figure_caption_prefixes: List[str] = field(
#         default_factory=lambda: ["Figure", "Fig.", "Fig"]
#     )
#     table_caption_prefixes: List[str] = field(
#         default_factory=lambda: ["Table", "Tab.", "Tab"]
#     )
#     list_styles: Dict[str, str] = field(
#         default_factory=lambda: {
#             "bullet": "List Bullet",
#             "numbered": "List Number",
#         }
#     )
#     equation_style: Optional[str] = None
#     columns: int = 1
#     double_anonymous: bool = False
# 
#     # Post-processing hooks
#     post_import_hooks: List[Callable] = field(default_factory=list)
#     pre_export_hooks: List[Callable] = field(default_factory=list)
# 
# 
# # --- Concrete profiles ------------------------------------------------------
# 
# 
# def _generic_profile() -> BaseWordProfile:
#     """
#     Generic Word template profile.
# 
#     This profile is intentionally conservative and assumes that:
#     - "Heading 1/2/3" are used for section headings.
#     - "Caption" is used for figure/table captions.
#     - "Normal" is the default body text.
# 
#     This should work reasonably well for many simple manuscripts.
#     """
#     return BaseWordProfile(
#         name="generic",
#         description="Generic Word mapping with standard Heading styles.",
#         heading_styles={
#             1: "Heading 1",
#             2: "Heading 2",
#             3: "Heading 3",
#             4: "Heading 4",
#         },
#         caption_style="Caption",
#         normal_style="Normal",
#         reference_section_titles=["References", "REFERENCES", "Bibliography"],
#     )
# 
# 
# def _mdpi_ijerph_profile() -> BaseWordProfile:
#     """
#     MDPI IJERPH template profile.
# 
#     Based on the MDPI Word template structure:
#     - Section headings use built-in heading styles.
#     - References section is titled "References".
#     - Single column layout.
#     - Specific section order: Introduction, Materials and Methods,
#       Results, Discussion, Conclusions.
#     """
#     return BaseWordProfile(
#         name="mdpi-ijerph",
#         description="MDPI IJERPH (Int. J. Environ. Res. Public Health) Word template.",
#         heading_styles={
#             1: "Heading 1",
#             2: "Heading 2",
#             3: "Heading 3",
#         },
#         caption_style="Caption",
#         normal_style="Normal",
#         reference_section_titles=["References"],
#         columns=1,
#     )
# 
# 
# def _resna_2025_profile() -> BaseWordProfile:
#     """
#     RESNA 2025 scientific paper template profile.
# 
#     The RESNA template:
#     - Uses all-caps section headings (INTRODUCTION, METHODS, etc.)
#     - Strict 4-page layout
#     - Two-column format
#     """
#     return BaseWordProfile(
#         name="resna-2025",
#         description="RESNA 2025 Scientific Paper Word template.",
#         heading_styles={
#             1: "Heading 1",  # INTRODUCTION, METHODS, etc.
#             2: "Heading 2",  # First-level sub-heading
#         },
#         caption_style="Caption",
#         normal_style="Normal",
#         reference_section_titles=["References", "REFERENCES"],
#         columns=2,
#     )
# 
# 
# def _iop_double_anonymous_profile() -> BaseWordProfile:
#     """
#     IOP double-anonymous Word template profile.
# 
#     The IOP template uses custom styles:
#     - IOPH1, IOPH2, IOPH3 for headings
#     - IOPTitle for title
#     - IOPAbsText for abstract
#     - IOPAff for affiliations
#     - Requires removal of author-identifying information
#     """
#     return BaseWordProfile(
#         name="iop-double-anonymous",
#         description="IOP double-anonymous Word template.",
#         heading_styles={
#             1: "IOPH1",
#             2: "IOPH2",
#             3: "IOPH3",
#         },
#         caption_style="Caption",
#         normal_style="Normal",
#         reference_section_titles=["References"],
#         double_anonymous=True,
#     )
# 
# 
# def _ieee_profile() -> BaseWordProfile:
#     """
#     IEEE conference/journal template profile.
# 
#     The IEEE template:
#     - Two-column format
#     - Roman numeral section numbering
#     - Specific citation style
#     """
#     return BaseWordProfile(
#         name="ieee",
#         description="IEEE conference/journal Word template.",
#         heading_styles={
#             1: "Heading 1",
#             2: "Heading 2",
#             3: "Heading 3",
#         },
#         caption_style="Caption",
#         normal_style="Normal",
#         reference_section_titles=["References", "REFERENCES"],
#         columns=2,
#     )
# 
# 
# def _springer_profile() -> BaseWordProfile:
#     """
#     Springer Nature journal template profile.
#     """
#     return BaseWordProfile(
#         name="springer",
#         description="Springer Nature journal Word template.",
#         heading_styles={
#             1: "Heading 1",
#             2: "Heading 2",
#             3: "Heading 3",
#         },
#         caption_style="Caption",
#         normal_style="Normal",
#         reference_section_titles=["References"],
#         columns=1,
#     )
# 
# 
# def _elsevier_profile() -> BaseWordProfile:
#     """
#     Elsevier journal template profile.
#     """
#     return BaseWordProfile(
#         name="elsevier",
#         description="Elsevier journal Word template.",
#         heading_styles={
#             1: "Heading 1",
#             2: "Heading 2",
#             3: "Heading 3",
#         },
#         caption_style="Caption",
#         normal_style="Normal",
#         reference_section_titles=["References"],
#         columns=1,
#     )
# 
# 
# # Registry of known profiles
# _PROFILES: Dict[str, BaseWordProfile] = {
#     "generic": _generic_profile(),
#     "mdpi-ijerph": _mdpi_ijerph_profile(),
#     "mdpi": _mdpi_ijerph_profile(),  # Alias
#     "resna-2025": _resna_2025_profile(),
#     "resna": _resna_2025_profile(),  # Alias
#     "iop-double-anonymous": _iop_double_anonymous_profile(),
#     "iop": _iop_double_anonymous_profile(),  # Alias
#     "ieee": _ieee_profile(),
#     "springer": _springer_profile(),
#     "elsevier": _elsevier_profile(),
# }
# 
# 
# def list_profiles() -> list[str]:
#     """
#     List available MS Word profiles.
# 
#     Returns
#     -------
#     list[str]
#         List of profile names (e.g., ["generic", "mdpi-ijerph", ...]).
# 
#     Examples
#     --------
#     >>> from scitex.msword import list_profiles
#     >>> profiles = list_profiles()
#     >>> "generic" in profiles
#     True
#     """
#     return sorted(_PROFILES.keys())
# 
# 
# def get_profile(name: str | None) -> BaseWordProfile:
#     """
#     Get a Word profile by name.
# 
#     Parameters
#     ----------
#     name : str | None
#         Profile name. If None, "generic" is used.
# 
#     Returns
#     -------
#     BaseWordProfile
#         The requested profile.
# 
#     Raises
#     ------
#     KeyError
#         If the profile name is unknown.
# 
#     Examples
#     --------
#     >>> from scitex.msword import get_profile
#     >>> profile = get_profile("mdpi-ijerph")
#     >>> profile.columns
#     1
#     """
#     if name is None:
#         return _PROFILES["generic"]
#     try:
#         return _PROFILES[name]
#     except KeyError as exc:
#         available = ", ".join(list_profiles())
#         raise KeyError(
#             f"Unknown MS Word profile: {name!r}. " f"Available profiles: {available}"
#         ) from exc
# 
# 
# def register_profile(profile: BaseWordProfile) -> None:
#     """
#     Register a custom Word profile.
# 
#     Parameters
#     ----------
#     profile : BaseWordProfile
#         The profile to register.
# 
#     Examples
#     --------
#     >>> from scitex.msword import BaseWordProfile, register_profile
#     >>> custom = BaseWordProfile(
#     ...     name="my-journal",
#     ...     description="My custom journal template",
#     ...     heading_styles={1: "Title", 2: "Subtitle"},
#     ... )
#     >>> register_profile(custom)
#     >>> "my-journal" in list_profiles()
#     True
#     """
#     _PROFILES[profile.name] = profile
# 
# 
# __all__ = [
#     "BaseWordProfile",
#     "list_profiles",
#     "get_profile",
#     "register_profile",
# ]

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/msword/profiles.py
# --------------------------------------------------------------------------------
