#!/usr/bin/env python3
"""Scitex utils module."""

from ._calc_bacc_from_conf_mat import calc_bacc_from_conf_mat
from ._calc_nice_ticks import calc_nice_ticks
from ._close import close
from ._colorbar import add_shared_colorbar, colorbar
from ._configure_mpl import configure_mpl
from ._figure_mm import apply_style_mm, create_figure_ax_mm, THEME_COLORS, _apply_theme_colors
from ._histogram_utils import HistogramBinManager, histogram_bin_manager
from ._im2grid import im2grid
from ._is_valid_axis import assert_valid_axis, is_valid_axis
from ._mk_colorbar import mk_colorbar
from ._mk_patches import mk_patches
from ._scientific_captions import (
    ScientificCaption,
    _escape_latex,
    _format_caption_for_md,
    _format_caption_for_tex,
    _format_caption_for_txt,
    add_figure_caption,
    add_panel_captions,
    caption_manager,
    create_figure_list,
    cross_ref,
    enhance_scitex_save_with_captions,
    export_captions,
    quick_caption,
    save_with_caption,
)
from ._scitex_config import SciTeXConfig, configure_scitex_ecosystem, get_scitex_config
from ._dimension_viewer import compare_modes, view_dimensions
from ._figure_from_axes_mm import (
    create_axes_with_size_mm,
    get_dimension_info,
    print_dimension_info,
)
from ._units import inch_to_mm, mm_to_inch, mm_to_pt, pt_to_mm
from .metadata import (
    assert_csv_json_consistency,
    collect_figure_metadata,
    collect_recipe_metadata,
    verify_csv_json_consistency,
)
from ._csv_column_naming import (
    get_csv_column_name,
    get_csv_column_prefix,
    parse_csv_column_name,
    sanitize_trace_id,
)
from ._hitmap import (
    extract_path_data,
    generate_hitmap_id_colors,
    get_all_artists,
    query_hitmap_neighborhood,
    save_hitmap_png,
)

__all__ = [
    "HistogramBinManager",
    "SciTeXConfig",
    "assert_csv_json_consistency",
    "ScientificCaption",
    "_escape_latex",
    "_format_caption_for_md",
    "_format_caption_for_tex",
    "_format_caption_for_txt",
    "add_figure_caption",
    "add_panel_captions",
    "add_shared_colorbar",
    "apply_style_mm",
    "assert_valid_axis",
    "calc_bacc_from_conf_mat",
    "calc_nice_ticks",
    "caption_manager",
    "close",
    "collect_figure_metadata",
    "collect_recipe_metadata",
    "colorbar",
    "compare_modes",
    "configure_mpl",
    "configure_scitex_ecosystem",
    "create_axes_with_size_mm",
    "create_figure_ax_mm",
    "create_figure_list",
    "cross_ref",
    "enhance_scitex_save_with_captions",
    "export_captions",
    "extract_path_data",
    "generate_hitmap_id_colors",
    "get_all_artists",
    "get_csv_column_name",
    "get_csv_column_prefix",
    "get_dimension_info",
    "get_scitex_config",
    "histogram_bin_manager",
    "im2grid",
    "inch_to_mm",
    "is_valid_axis",
    "mk_colorbar",
    "mk_patches",
    "mm_to_inch",
    "mm_to_pt",
    "parse_csv_column_name",
    "print_dimension_info",
    "pt_to_mm",
    "query_hitmap_neighborhood",
    "quick_caption",
    "sanitize_trace_id",
    "save_hitmap_png",
    "save_with_caption",
    "verify_csv_json_consistency",
    "view_dimensions",
]
