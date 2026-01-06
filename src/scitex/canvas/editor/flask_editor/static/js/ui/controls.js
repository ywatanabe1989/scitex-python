/**
 * Controls Panel
 * Handles all form controls and data collection
 */

// ============================================================================
// Collect Overrides from Form
// ============================================================================
function collectOverrides() {
    const o = {};

    // Labels - Title
    const title = document.getElementById('title')?.value;
    if (title) o.title = title;
    o.show_title = document.getElementById('show_title')?.checked;
    o.title_fontsize = parseInt(document.getElementById('title_fontsize')?.value) || 8;

    // Labels - Caption
    const caption = document.getElementById('caption')?.value;
    if (caption) o.caption = caption;
    o.show_caption = document.getElementById('show_caption')?.checked;
    o.caption_fontsize = parseInt(document.getElementById('caption_fontsize')?.value) || 7;

    // Labels - Axis
    const xlabel = document.getElementById('xlabel')?.value;
    const ylabel = document.getElementById('ylabel')?.value;
    if (xlabel) o.xlabel = xlabel;
    if (ylabel) o.ylabel = ylabel;

    // Axis limits
    const xmin = document.getElementById('xmin')?.value;
    const xmax = document.getElementById('xmax')?.value;
    if (xmin !== '' && xmax !== '') o.xlim = [parseFloat(xmin), parseFloat(xmax)];

    const ymin = document.getElementById('ymin')?.value;
    const ymax = document.getElementById('ymax')?.value;
    if (ymin !== '' && ymax !== '') o.ylim = [parseFloat(ymin), parseFloat(ymax)];

    // Traces
    o.traces = traces;

    // Legend
    o.legend_visible = document.getElementById('legend_visible')?.checked;
    o.legend_loc = document.getElementById('legend_loc')?.value;
    o.legend_frameon = document.getElementById('legend_frameon')?.checked;
    o.legend_fontsize = parseInt(document.getElementById('legend_fontsize')?.value) || 6;
    o.legend_ncols = parseInt(document.getElementById('legend_ncols')?.value) || 1;
    o.legend_x = parseFloat(document.getElementById('legend_x')?.value) || 0.95;
    o.legend_y = parseFloat(document.getElementById('legend_y')?.value) || 0.95;

    // Axis and Ticks - X Axis (Bottom)
    o.x_n_ticks = parseInt(document.getElementById('x_n_ticks')?.value) || 4;
    o.hide_x_ticks = document.getElementById('hide_x_ticks')?.checked;
    o.x_tick_fontsize = parseInt(document.getElementById('x_tick_fontsize')?.value) || 7;
    o.x_tick_direction = document.getElementById('x_tick_direction')?.value;
    o.x_tick_length = parseFloat(document.getElementById('x_tick_length')?.value) || 0.8;
    o.x_tick_width = parseFloat(document.getElementById('x_tick_width')?.value) || 0.2;

    // X Axis (Top)
    o.show_x_top = document.getElementById('show_x_top')?.checked;
    o.x_top_mirror = document.getElementById('x_top_mirror')?.checked;

    // Y Axis (Left)
    o.y_n_ticks = parseInt(document.getElementById('y_n_ticks')?.value) || 4;
    o.hide_y_ticks = document.getElementById('hide_y_ticks')?.checked;
    o.y_tick_fontsize = parseInt(document.getElementById('y_tick_fontsize')?.value) || 7;
    o.y_tick_direction = document.getElementById('y_tick_direction')?.value;
    o.y_tick_length = parseFloat(document.getElementById('y_tick_length')?.value) || 0.8;
    o.y_tick_width = parseFloat(document.getElementById('y_tick_width')?.value) || 0.2;

    // Y Axis (Right)
    o.show_y_right = document.getElementById('show_y_right')?.checked;
    o.y_right_mirror = document.getElementById('y_right_mirror')?.checked;

    // Spines
    o.hide_bottom_spine = document.getElementById('hide_bottom_spine')?.checked;
    o.hide_left_spine = document.getElementById('hide_left_spine')?.checked;

    // Z Axis (3D)
    o.hide_z_ticks = document.getElementById('hide_z_ticks')?.checked;
    o.z_n_ticks = parseInt(document.getElementById('z_n_ticks')?.value) || 4;
    o.z_tick_fontsize = parseInt(document.getElementById('z_tick_fontsize')?.value) || 7;
    o.z_tick_direction = document.getElementById('z_tick_direction')?.value;

    // Style
    o.grid = document.getElementById('grid')?.checked;
    o.hide_top_spine = document.getElementById('hide_top_spine')?.checked;
    o.hide_right_spine = document.getElementById('hide_right_spine')?.checked;
    o.axis_width = parseFloat(document.getElementById('axis_width')?.value) || 0.2;
    o.axis_fontsize = parseInt(document.getElementById('axis_fontsize')?.value) || 7;
    o.facecolor = document.getElementById('facecolor')?.value;
    o.transparent = document.getElementById('transparent')?.value === 'true';

    // Dimensions (always in inches for matplotlib)
    o.fig_size = getFigSizeInches();
    o.dpi = parseInt(document.getElementById('dpi')?.value) || 300;

    // Annotations
    o.annotations = overrides.annotations || [];

    // Element-specific overrides (per-element styles)
    if (overrides.element_overrides) {
        o.element_overrides = overrides.element_overrides;
    }

    return o;
}

// ============================================================================
// Update Controls from Overrides
// ============================================================================
function updateControlsFromOverrides() {
    // This function updates the form fields from the overrides object
    // (Would be populated from the server data)
}

// ============================================================================
// Axis Tab Switching
// ============================================================================
function switchAxisTab(axis) {
    // Update tab buttons
    document.querySelectorAll('.axis-tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.axis === axis);
    });

    // Update panels
    document.querySelectorAll('.axis-panel').forEach(panel => {
        panel.style.display = panel.dataset.axis === axis ? 'block' : 'none';
    });
}

// ============================================================================
// Custom Legend Position Toggle
// ============================================================================
function toggleCustomLegendPosition() {
    const customPos = document.getElementById('legend-custom-position');
    const legendLoc = document.getElementById('legend_loc');
    if (customPos && legendLoc) {
        customPos.style.display = legendLoc.value === 'custom' ? 'block' : 'none';
    }
}

// ============================================================================
// Background Type Toggle
// ============================================================================
function setBackgroundType(type) {
    backgroundType = type;

    // Update hidden inputs for collectOverrides
    const transparentInput = document.getElementById('transparent');
    const facecolorInput = document.getElementById('facecolor');

    if (type === 'white') {
        if (transparentInput) transparentInput.value = 'false';
        if (facecolorInput) facecolorInput.value = 'white';
    } else if (type === 'black') {
        if (transparentInput) transparentInput.value = 'false';
        if (facecolorInput) facecolorInput.value = 'black';
    } else {
        // transparent
        if (transparentInput) transparentInput.value = 'true';
        if (facecolorInput) facecolorInput.value = 'none';
    }

    // Update button states
    document.querySelectorAll('.background-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.type === type);
    });

    // Trigger update only after initialization
    if (!initializingBackground) {
        scheduleUpdate();
    }
}

// ============================================================================
// Reset Overrides
// ============================================================================
function resetOverrides() {
    if (confirm('Reset all settings to defaults?')) {
        overrides = {};
        updatePreview();
    }
}
