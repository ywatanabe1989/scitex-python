#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/_figz_modules/_caption.py

"""Caption generation mixin for Figz bundles."""

from typing import Any, Dict, List, Optional


class FigzCaptionMixin:
    """Mixin providing caption generation capabilities for Figz bundles."""

    def set_panel_info(
        self,
        element_id: str,
        panel_letter: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Set panel letter and/or description for an element.

        Parameters
        ----------
        element_id : str
            ID of the element to update
        panel_letter : str, optional
            Panel letter (e.g., "A", "B", "C")
        description : str, optional
            Description for caption generation

        Returns
        -------
        bool
            True if element was found and updated
        """
        elem = self.get_element(element_id)
        if elem is None:
            return False

        if panel_letter is not None:
            elem["panel_letter"] = panel_letter
        if description is not None:
            elem["description"] = description

        self._modified = True
        return True

    def get_panel_info(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Get panel info for an element.

        Returns dict with panel_letter and description if set.
        """
        elem = self.get_element(element_id)
        if elem is None:
            return None

        return {
            "panel_letter": elem.get("panel_letter", ""),
            "description": elem.get("description", ""),
        }

    def auto_assign_panel_letters(self, style: str = "uppercase") -> None:
        """Auto-assign panel letters to plot elements in order.

        Parameters
        ----------
        style : str
            Letter style: "uppercase" (A,B,C), "lowercase" (a,b,c),
            "roman" (i,ii,iii), "Roman" (I,II,III)
        """
        plot_elements = [e for e in self.elements if e.get("type") == "plot"]

        for idx, elem in enumerate(plot_elements):
            if style == "uppercase":
                letter = chr(ord("A") + idx)
            elif style == "lowercase":
                letter = chr(ord("a") + idx)
            elif style == "roman":
                letter = _to_roman(idx + 1).lower()
            elif style == "Roman":
                letter = _to_roman(idx + 1)
            else:
                letter = chr(ord("A") + idx)

            elem["panel_letter"] = letter

        self._modified = True

    def get_caption(self) -> str:
        """Generate plain text caption from figure title and panel descriptions.

        Format: "Figure 1: Main Title. (A) Description A. (B) Description B."

        Returns
        -------
        str
            Generated caption text
        """
        from scitex.schema import (
            generate_caption,
        )

        title, caption, panel_labels = self._load_caption_settings()
        panels = self._build_panel_info_list()

        return generate_caption(title, caption, panels, panel_labels)

    def get_caption_latex(self) -> str:
        """Generate LaTeX-formatted caption.

        Format: "\\textbf{Figure 1: Main Title.} \\textbf{(A)} Description A."

        Returns
        -------
        str
            LaTeX-formatted caption
        """
        from scitex.schema import (
            generate_caption_latex,
        )

        title, caption, panel_labels = self._load_caption_settings()
        panels = self._build_panel_info_list()

        return generate_caption_latex(title, caption, panels, panel_labels)

    def get_caption_markdown(self) -> str:
        """Generate Markdown-formatted caption.

        Format: "**Figure 1: Main Title.** **(A)** Description A."

        Returns
        -------
        str
            Markdown-formatted caption
        """
        from scitex.schema import (
            generate_caption_markdown,
        )

        title, caption, panel_labels = self._load_caption_settings()
        panels = self._build_panel_info_list()

        return generate_caption_markdown(title, caption, panels, panel_labels)

    def _load_caption_settings(self):
        """Load figure title, caption, and panel labels from theme.json."""
        from scitex.schema import Caption, FigureTitle, PanelLabels

        theme = self._load_theme()

        title_data = theme.get("figure_title", {})
        title = FigureTitle.from_dict(title_data) if title_data else FigureTitle()

        caption_data = theme.get("caption", {})
        caption = Caption.from_dict(caption_data) if caption_data else Caption()

        labels_data = theme.get("panel_labels", {})
        labels = PanelLabels.from_dict(labels_data) if labels_data else PanelLabels()

        return title, caption, labels

    def _load_theme(self) -> Dict[str, Any]:
        """Load theme.json data."""
        import json

        from scitex.io.bundle import ZipBundle

        if self._is_dir:
            theme_path = self.path / "theme.json"
            if theme_path.exists():
                with open(theme_path, encoding="utf-8") as f:
                    return json.load(f)
        else:
            try:
                with ZipBundle(self.path, mode="r") as zb:
                    return zb.read_json("theme.json")
            except FileNotFoundError:
                pass

        return {}

    def _build_panel_info_list(self) -> List:
        """Build list of PanelInfo from elements."""
        from scitex.schema import PanelInfo

        panels = []
        plot_elements = [e for e in self.elements if e.get("type") == "plot"]

        for idx, elem in enumerate(plot_elements):
            panel = PanelInfo(
                panel_id=elem.get("id", ""),
                letter=elem.get("panel_letter", ""),
                description=elem.get("description", ""),
                order=idx,
            )
            panels.append(panel)

        return panels

    def set_figure_title(
        self,
        text: str,
        prefix: str = "Figure",
        number: Optional[int] = None,
    ) -> None:
        """Set figure title in theme.json.

        Parameters
        ----------
        text : str
            Main title text
        prefix : str
            Prefix (e.g., "Figure", "Fig.")
        number : int, optional
            Figure number
        """
        theme = self._load_theme()
        theme["figure_title"] = {
            "text": text,
            "prefix": prefix,
            "font_size_pt": 10.0,
            "font_weight": "bold",
            "position": "top",
            "visible": True,
        }
        if number is not None:
            theme["figure_title"]["number"] = number

        self._save_theme(theme)

    def set_caption(self, text: str = "", auto_generate: bool = True) -> None:
        """Set caption in theme.json.

        Parameters
        ----------
        text : str
            Manual caption text (used if auto_generate is False)
        auto_generate : bool
            Whether to auto-generate from panel descriptions
        """
        theme = self._load_theme()
        theme["caption"] = {
            "text": text,
            "auto_generate": auto_generate,
            "font_size_pt": 8.0,
            "position": "bottom",
            "visible": True,
        }
        self._save_theme(theme)

    def add_visual_caption(
        self,
        fontsize: float = 9.0,
        margin_mm: float = 3.0,
        max_width_mm: Optional[float] = None,
        wrap: bool = True,
    ) -> str:
        """Add visual caption below the figure as a text element.

        This renders the caption text from get_caption() directly on the figure,
        similar to figure legends in scientific papers. The caption wraps within
        the figure width - the figure width is NOT expanded.

        Parameters
        ----------
        fontsize : float
            Font size in points (default: 9.0)
        margin_mm : float
            Margin between figure content and caption (default: 3.0)
        max_width_mm : float, optional
            Maximum width for text wrapping. Defaults to figure width - 10mm.
        wrap : bool
            Whether to wrap long text (default: True)

        Returns
        -------
        str
            The caption text that was added

        Example
        -------
        >>> fig.set_figure_title("Results", prefix="Figure", number=1)
        >>> fig.set_panel_info("plot_A", panel_letter="A", description="Raw data")
        >>> fig.add_visual_caption()  # Renders "Figure 1. Results. (A) Raw data."
        """
        import textwrap

        # Get caption text
        caption_text = self.get_caption()
        if not caption_text:
            return ""

        # Calculate dimensions - keep figure width fixed
        current_size = self.size_mm
        fig_width = current_size.get("width", 170)
        fig_height = current_size.get("height", 120)

        if max_width_mm is None:
            max_width_mm = fig_width - 10  # 5mm margin on each side

        # Calculate chars per line based on font size
        # Conservative estimate: at 9pt, ~0.5 chars per mm (accounts for variable width)
        chars_per_mm = 0.5 * (9.0 / fontsize)
        chars_per_line = max(40, int(max_width_mm * chars_per_mm))  # min 40 chars

        # Wrap text with newlines for proper rendering
        if wrap and len(caption_text) > chars_per_line:
            wrapped_lines = textwrap.wrap(caption_text, width=chars_per_line)
            wrapped_text = "\n".join(wrapped_lines)
            num_lines = len(wrapped_lines)
        else:
            wrapped_text = caption_text
            num_lines = 1

        # Calculate caption height (line height ≈ 1.4x font size in mm)
        line_height_mm = fontsize * 0.35 * 1.4
        caption_height_mm = num_lines * line_height_mm

        # Expand only figure HEIGHT to accommodate caption (width stays same)
        new_height = fig_height + margin_mm + caption_height_mm + margin_mm
        self._spec["size_mm"] = {"width": fig_width, "height": new_height}

        # Position caption below existing content
        caption_y = fig_height + margin_mm

        # Remove existing caption element if present
        self.remove_element("_visual_caption")

        # Add caption as text element with pre-wrapped text
        self.add_element(
            "_visual_caption",
            "text",
            wrapped_text,
            position={"x_mm": 5, "y_mm": caption_y},
            fontsize=fontsize,
            ha="left",
            va="top",
        )

        self._modified = True
        return caption_text

    def _save_theme(self, theme: Dict[str, Any]) -> None:
        """Save theme.json data."""
        import json

        from scitex.io.bundle import ZipBundle

        if self._is_dir:
            with open(self.path / "theme.json", "w", encoding="utf-8") as f:
                json.dump(theme, f, indent=2)
        else:
            with ZipBundle(self.path, mode="a") as zb:
                zb.write_json("theme.json", theme)


def _to_roman(num: int) -> str:
    """Convert integer to Roman numeral."""
    values = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    result = ""
    for value, numeral in values:
        while num >= value:
            result += numeral
            num -= value
    return result


__all__ = ["FigzCaptionMixin"]


# EOF
