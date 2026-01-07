# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_backend/_render.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # File: ./src/scitex/vis/backend/render.py
# """Render figure models to matplotlib figures using scitex.plt."""
# 
# from typing import Any, Dict
# 
# import numpy as np
# 
# import scitex.logging as logging
# 
# from .._models import AnnotationModel, AxesModel, FigureModel, GuideModel, PlotModel
# from ._parser import parse_figure_json
# 
# logger = logging.getLogger(__name__)
# 
# 
# def render_figure(fig_model: FigureModel):
#     """
#     Render FigureModel to matplotlib figure using scitex.plt.
# 
#     Parameters
#     ----------
#     fig_model : FigureModel
#         Figure model to render
# 
#     Returns
#     -------
#     tuple
#         (fig, axes) where fig is FigWrapper and axes is list of AxisWrapper
# 
#     Examples
#     --------
#     >>> fig_model = FigureModel(width_mm=180, height_mm=120)
#     >>> fig, axes = render_figure(fig_model)
#     """
#     import scitex as stx
# 
#     # Validate before rendering
#     fig_model.validate()
# 
#     # Convert mm to inches for matplotlib
#     MM_TO_INCH = 1 / 25.4
#     figsize_inch = (
#         fig_model.width_mm * MM_TO_INCH,
#         fig_model.height_mm * MM_TO_INCH,
#     )
# 
#     # Create figure with mm dimensions
#     fig, axes = stx.plt.subplots(
#         nrows=fig_model.nrows,
#         ncols=fig_model.ncols,
#         figsize=figsize_inch,
#         dpi=fig_model.dpi,
#         facecolor=fig_model.facecolor,
#         edgecolor=fig_model.edgecolor,
#     )
# 
#     # Ensure axes is always a list
#     if not isinstance(axes, (list, np.ndarray)):
#         axes = [axes]
#     elif isinstance(axes, np.ndarray):
#         axes = axes.flatten().tolist()
# 
#     # TODO: Apply mm-based spacing (left_mm, right_mm, etc.)
#     # Currently, matplotlib subplots_adjust uses figure coordinates (0-1), not mm
#     # Future: Pass margin_left_mm, margin_right_mm etc. to stx.plt.subplots()
#     # to leverage scitex.plt's mm-control system
# 
#     # Add suptitle if specified
#     if fig_model.suptitle:
#         suptitle_kwargs = {"t": fig_model.suptitle}
#         if fig_model.suptitle_fontsize is not None:
#             suptitle_kwargs["fontsize"] = fig_model.suptitle_fontsize
#         if fig_model.suptitle_fontweight is not None:
#             suptitle_kwargs["fontweight"] = fig_model.suptitle_fontweight
#         if fig_model.suptitle_y is not None:
#             suptitle_kwargs["y"] = fig_model.suptitle_y
#         fig.suptitle(**suptitle_kwargs)
# 
#     # Render each axes
#     for i, axes_config in enumerate(fig_model.axes):
#         if i < len(axes):
#             axes_model = AxesModel.from_dict(axes_config)
#             render_axes(axes[i], axes_model)
# 
#     return fig, axes
# 
# 
# def render_axes(ax, axes_model: AxesModel):
#     """
#     Render AxesModel onto a matplotlib axis.
# 
#     Parameters
#     ----------
#     ax : AxisWrapper or matplotlib Axes
#         Target axis
#     axes_model : AxesModel
#         Axes model to render
#     """
#     # Validate
#     axes_model.validate()
# 
#     # Render plots first (background layer)
#     for plot_config in axes_model.plots:
#         plot_model = PlotModel.from_dict(plot_config)
#         render_plot(ax, plot_model)
# 
#     # Render guides (middle layer - reference lines/spans over plots)
#     # Note: Use zorder in guide config to control layering if needed
#     for guide_config in axes_model.guides:
#         guide_model = GuideModel.from_dict(guide_config)
#         render_guide(ax, guide_model)
# 
#     # Render annotations last (foreground layer)
#     for annotation_config in axes_model.annotations:
#         annotation_model = AnnotationModel.from_dict(annotation_config)
#         render_annotation(ax, annotation_model)
# 
#     # Apply axis properties
#     if axes_model.xlabel:
#         ax.set_xlabel(axes_model.xlabel)
# 
#     if axes_model.ylabel:
#         ax.set_ylabel(axes_model.ylabel)
# 
#     if axes_model.title:
#         ax.set_title(axes_model.title)
# 
#     # Set axis limits
#     if axes_model.xlim:
#         ax.set_xlim(axes_model.xlim)
# 
#     if axes_model.ylim:
#         ax.set_ylim(axes_model.ylim)
# 
#     # Set axis scales
#     ax.set_xscale(axes_model.xscale)
#     ax.set_yscale(axes_model.yscale)
# 
#     # Ticks
#     if axes_model.xticks is not None:
#         ax.set_xticks(axes_model.xticks)
# 
#     if axes_model.yticks is not None:
#         ax.set_yticks(axes_model.yticks)
# 
#     if axes_model.xticklabels is not None:
#         ax.set_xticklabels(axes_model.xticklabels)
# 
#     if axes_model.yticklabels is not None:
#         ax.set_yticklabels(axes_model.yticklabels)
# 
#     # Apply style properties
#     style = axes_model.style
# 
#     # Grid
#     if style.grid:
#         ax.grid(
#             alpha=style.grid_alpha,
#             linestyle=style.grid_linestyle,
#             linewidth=style.grid_linewidth,
#         )
# 
#     # Spines
#     for spine, visible in style.spines_visible.items():
#         if hasattr(ax, "spines") and spine in ax.spines:
#             ax.spines[spine].set_visible(visible)
# 
#     # Tick label size
#     if style.tick_fontsize is not None:
#         ax.tick_params(labelsize=style.tick_fontsize)
# 
#     # Legend
#     if style.legend:
#         legend_kwargs = {
#             "loc": style.legend_loc,
#             "frameon": style.legend_frameon,
#         }
#         if style.legend_fontsize is not None:
#             legend_kwargs["fontsize"] = style.legend_fontsize
#         ax.legend(**legend_kwargs)
# 
#     # Aspect
#     if style.aspect:
#         ax.set_aspect(style.aspect)
# 
#     # Background color
#     if style.facecolor:
#         ax.set_facecolor(style.facecolor)
# 
# 
# def render_plot(ax, plot_model: PlotModel):
#     """
#     Render PlotModel onto an axis.
# 
#     Parameters
#     ----------
#     ax : AxisWrapper or matplotlib Axes
#         Target axis
#     plot_model : PlotModel
#         Plot model to render
#     """
#     # Validate
#     plot_model.validate()
# 
#     data = plot_model.data
# 
#     # Build kwargs from style (clean separation of concerns)
#     # Only include relevant kwargs for each plot type
#     style_dict = plot_model.style.to_dict()
# 
#     # Common kwargs for all plot types
#     common_keys = {"color", "alpha", "zorder"}
# 
#     # Define relevant keys for each plot type
#     plot_type_keys = {
#         "line": {"linewidth", "linestyle", "marker", "markersize"},
#         "scatter": {"marker", "markersize"},
#         "errorbar": {"linewidth", "linestyle", "marker", "xerr", "yerr", "capsize"},
#         "bar": {"width"},
#         "barh": {"width"},  # Will be renamed to height
#         "hist": {"bins", "density", "cumulative"},
#         "fill_between": {"linewidth", "fill_color", "fill_alpha"},
#         "heatmap": {"cmap", "vmin", "vmax", "interpolation"},
#         "imshow": {"cmap", "vmin", "vmax", "interpolation"},
#         "contour": {"cmap", "vmin", "vmax"},
#         "contourf": {"cmap", "vmin", "vmax"},
#     }
# 
#     # Get relevant keys for this plot type
#     relevant_keys = common_keys | plot_type_keys.get(plot_model.plot_type, set())
# 
#     # Filter kwargs to only include relevant ones
#     kwargs = {k: v for k, v in style_dict.items() if k in relevant_keys}
# 
#     # Add label if present
#     if plot_model.label:
#         kwargs["label"] = plot_model.label
# 
#     # Merge extra kwargs
#     kwargs.update(plot_model.extra_kwargs)
# 
#     # Render based on plot type
#     if plot_model.plot_type == "line":
#         ax.plot(data["x"], data["y"], **kwargs)
# 
#     elif plot_model.plot_type == "scatter":
#         # Rename markersize -> s for scatter
#         if "markersize" in kwargs:
#             kwargs["s"] = kwargs.pop("markersize")
#         ax.scatter(data["x"], data["y"], **kwargs)
# 
#     elif plot_model.plot_type == "errorbar":
#         ax.errorbar(data["x"], data["y"], **kwargs)
# 
#     elif plot_model.plot_type == "bar":
#         height = data.get("height", data.get("y"))
#         ax.bar(data["x"], height, **kwargs)
# 
#     elif plot_model.plot_type == "barh":
#         # Rename width -> height for horizontal bars
#         if "width" in kwargs:
#             kwargs["height"] = kwargs.pop("width")
#         width = data.get("width", data.get("x"))
#         ax.barh(data["y"], width, **kwargs)
# 
#     elif plot_model.plot_type == "hist":
#         ax.hist(data["x"], **kwargs)
# 
#     elif plot_model.plot_type == "fill_between":
#         # Use fill_alpha and fill_color if present
#         if "fill_alpha" in kwargs:
#             kwargs["alpha"] = kwargs.pop("fill_alpha")
#         if "fill_color" in kwargs:
#             kwargs["color"] = kwargs.pop("fill_color")
#         ax.fill_between(data["x"], data["y1"], data["y2"], **kwargs)
# 
#     elif plot_model.plot_type in ["heatmap", "imshow"]:
#         img_data = data.get("z", data.get("img"))
#         ax.imshow(img_data, **kwargs)
# 
#     elif plot_model.plot_type == "contour":
#         ax.contour(data["x"], data["y"], data["z"], **kwargs)
# 
#     elif plot_model.plot_type == "contourf":
#         ax.contourf(data["x"], data["y"], data["z"], **kwargs)
# 
#     else:
#         raise ValueError(f"Unsupported plot type: {plot_model.plot_type}")
# 
# 
# def render_guide(ax, guide_model: GuideModel):
#     """
#     Render GuideModel onto an axis.
# 
#     Parameters
#     ----------
#     ax : AxisWrapper or matplotlib Axes
#         Target axis
#     guide_model : GuideModel
#         Guide model to render
#     """
#     # Validate
#     guide_model.validate()
# 
#     # Build kwargs from style
#     kwargs = guide_model.style.to_dict()
# 
#     # Add label if present
#     if guide_model.label:
#         kwargs["label"] = guide_model.label
# 
#     # Render based on guide type
#     if guide_model.guide_type == "axhline":
#         ax.axhline(guide_model.y, **kwargs)
# 
#     elif guide_model.guide_type == "axvline":
#         ax.axvline(guide_model.x, **kwargs)
# 
#     elif guide_model.guide_type == "axhspan":
#         ax.axhspan(guide_model.ymin, guide_model.ymax, **kwargs)
# 
#     elif guide_model.guide_type == "axvspan":
#         ax.axvspan(guide_model.xmin, guide_model.xmax, **kwargs)
# 
# 
# def render_annotation(ax, annotation_model: AnnotationModel):
#     """
#     Render AnnotationModel onto an axis.
# 
#     Parameters
#     ----------
#     ax : AxisWrapper or matplotlib Axes
#         Target axis
#     annotation_model : AnnotationModel
#         Annotation model to render
#     """
#     # Validate
#     annotation_model.validate()
# 
#     # Build kwargs from style
#     kwargs = annotation_model.style.to_dict()
# 
#     # Render based on annotation type
#     if annotation_model.annotation_type == "text":
#         ax.text(
#             annotation_model.x,
#             annotation_model.y,
#             annotation_model.text,
#             **kwargs,
#         )
# 
#     elif annotation_model.annotation_type == "annotate":
#         ax.annotate(
#             annotation_model.text,
#             xy=(annotation_model.x, annotation_model.y),
#             xytext=annotation_model.xytext,
#             **kwargs,
#         )
# 
# 
# def build_figure_from_json(fig_json: Dict[str, Any]):
#     """
#     Build matplotlib figure from figure JSON.
# 
#     This is the main entry point for rendering.
# 
#     Parameters
#     ----------
#     fig_json : Dict[str, Any]
#         Figure JSON specification
# 
#     Returns
#     -------
#     tuple
#         (fig, axes) where fig is FigWrapper and axes is list of AxisWrapper
# 
#     Examples
#     --------
#     >>> fig_json = {...}
#     >>> fig, axes = build_figure_from_json(fig_json)
#     >>> import scitex as stx
#     >>> stx.io.save(fig, "output.png")
#     """
#     fig_model = parse_figure_json(fig_json)
#     return render_figure(fig_model)
# 
# 
# def render_traces(ax, trace, data, theme=None):
#     """
#     Render a TraceEncoding onto an axis.
# 
#     Parameters
#     ----------
#     ax : matplotlib Axes
#         Target axis
#     trace : TraceEncoding
#         Trace encoding specification
#     data : pd.DataFrame or None
#         Data to plot
#     theme : Theme, optional
#         Theme for styling
# 
#     Raises
#     ------
#     ValueError
#         If data is None or required columns are missing
#     """
#     if data is None:
#         logger.error(f"No data provided for trace '{trace.trace_id}'")
#         raise ValueError(f"No data provided for trace '{trace.trace_id}'")
# 
#     # Get style from theme
#     color = None
#     linewidth = 1.0
#     alpha = 1.0
# 
#     if theme:
#         # Check for trace-specific theme
#         if hasattr(theme, "traces") and theme.traces:
#             for t in theme.traces:
#                 if t.trace_id == trace.trace_id:
#                     color = t.color
#                     if t.line_width_pt:
#                         linewidth = t.line_width_pt
#                     if t.alpha:
#                         alpha = t.alpha
#                     break
# 
#         # Fall back to default colors
#         if not color and hasattr(theme, "colors"):
#             palette = theme.colors.palette
#             if palette:
#                 color = palette[0]
# 
#         if hasattr(theme, "lines"):
#             linewidth = theme.lines.width_pt
# 
#     # Default color if none set
#     if not color:
#         color = "#1f77b4"
# 
#     # Get column names from encoding
#     x_col = trace.x.column if trace.x else None
#     y_col = trace.y.column if trace.y else None
# 
#     if not x_col or not y_col:
#         logger.error(f"Trace '{trace.trace_id}' missing x or y encoding")
#         raise ValueError(f"Trace '{trace.trace_id}' missing x or y encoding")
# 
#     if x_col not in data.columns:
#         logger.error(f"Column '{x_col}' not found in data. Available: {list(data.columns)}")
#         raise ValueError(f"Column '{x_col}' not found in data. Available: {list(data.columns)}")
#     if y_col not in data.columns:
#         logger.error(f"Column '{y_col}' not found in data. Available: {list(data.columns)}")
#         raise ValueError(f"Column '{y_col}' not found in data. Available: {list(data.columns)}")
# 
#     # Plot the data
#     ax.plot(
#         data[x_col],
#         data[y_col],
#         color=color,
#         linewidth=linewidth,
#         alpha=alpha,
#         label=trace.trace_id,
#     )
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_backend/_render.py
# --------------------------------------------------------------------------------
