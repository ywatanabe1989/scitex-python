/**
 * Main Entry Point
 * Initializes all modules and sets up the Flask figure editor
 *
 * Module Structure:
 * - core/: State management, utilities, API communication
 * - canvas/: Canvas view, panel dragging, resizing, selection
 * - editor/: Preview management, element detection, overlay rendering
 * - alignment/: Panel alignment (basic and axis-based)
 * - shortcuts/: Keyboard shortcuts and context menus
 * - ui/: Form controls, downloads, help, theme
 */

// ============================================================================
// Initialization
// ============================================================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('Flask Figure Editor initializing...');

    // Initialize theme
    initializeTheme();

    // Initialize form controls
    initializeControls();

    // Load initial preview
    loadInitialPreview();

    // Setup resize observer for overlay
    setupResizeObserver();

    // Setup field-to-element sync
    setupFieldToElementSync();

    console.log('Flask Figure Editor initialized');
});

// ============================================================================
// Form Controls Initialization
// ============================================================================
function initializeControls() {
    // Labels - Title
    const titleInput = document.getElementById('title');
    if (titleInput) titleInput.addEventListener('input', scheduleUpdate);

    // Labels - Caption
    const captionInput = document.getElementById('caption');
    if (captionInput) captionInput.addEventListener('input', scheduleUpdate);

    // Labels - Axis
    const xlabelInput = document.getElementById('xlabel');
    const ylabelInput = document.getElementById('ylabel');
    if (xlabelInput) xlabelInput.addEventListener('input', scheduleUpdate);
    if (ylabelInput) ylabelInput.addEventListener('input', scheduleUpdate);

    // Axis limits
    ['xmin', 'xmax', 'ymin', 'ymax'].forEach(id => {
        const input = document.getElementById(id);
        if (input) input.addEventListener('input', scheduleUpdate);
    });

    // Traces
    updateTracesList();

    // Legend
    const legendVisibleInput = document.getElementById('legend_visible');
    if (legendVisibleInput) legendVisibleInput.addEventListener('change', scheduleUpdate);

    const legendLocInput = document.getElementById('legend_loc');
    if (legendLocInput) {
        legendLocInput.addEventListener('change', function() {
            toggleCustomLegendPosition();
            scheduleUpdate();
        });
    }

    // Axis and Ticks - X Axis (Bottom)
    ['x_n_ticks', 'hide_x_ticks', 'x_tick_fontsize', 'x_tick_direction', 'x_tick_length', 'x_tick_width'].forEach(id => {
        const input = document.getElementById(id);
        if (input) input.addEventListener('change', scheduleUpdate);
    });

    // X Axis (Top)
    ['show_x_top', 'x_top_mirror'].forEach(id => {
        const input = document.getElementById(id);
        if (input) input.addEventListener('change', scheduleUpdate);
    });

    // Y Axis (Left)
    ['y_n_ticks', 'hide_y_ticks', 'y_tick_fontsize', 'y_tick_direction', 'y_tick_length', 'y_tick_width'].forEach(id => {
        const input = document.getElementById(id);
        if (input) input.addEventListener('change', scheduleUpdate);
    });

    // Y Axis (Right)
    ['show_y_right', 'y_right_mirror'].forEach(id => {
        const input = document.getElementById(id);
        if (input) input.addEventListener('change', scheduleUpdate);
    });

    // Spines
    ['hide_bottom_spine', 'hide_left_spine'].forEach(id => {
        const input = document.getElementById(id);
        if (input) input.addEventListener('change', scheduleUpdate);
    });

    // Z Axis (3D)
    ['hide_z_ticks', 'z_n_ticks', 'z_tick_fontsize', 'z_tick_direction'].forEach(id => {
        const input = document.getElementById(id);
        if (input) input.addEventListener('change', scheduleUpdate);
    });

    // Style
    ['grid', 'hide_top_spine', 'hide_right_spine', 'axis_width', 'axis_fontsize'].forEach(id => {
        const input = document.getElementById(id);
        if (input) input.addEventListener('change', scheduleUpdate);
    });

    // Initialize background type from overrides
    if (overrides.transparent) {
        setBackgroundType('transparent');
    } else if (overrides.facecolor === 'black') {
        setBackgroundType('black');
    } else {
        setBackgroundType('white');
    }

    // Dimensions (convert from inches in metadata to mm by default)
    if (overrides.fig_size) {
        // fig_size is in inches in the JSON - convert to mm for default display
        const widthInput = document.getElementById('fig_width');
        const heightInput = document.getElementById('fig_height');
        if (widthInput && heightInput) {
            widthInput.value = (overrides.fig_size[0] * INCH_TO_MM).toFixed(1);
            heightInput.value = (overrides.fig_size[1] * INCH_TO_MM).toFixed(1);
        }
    }

    // Default unit is mm, which is already set in HTML and JS state

    // Setup color sync for selected element property inputs
    setupColorSync('trace-color-picker', 'trace-color-text');
    setupColorSync('scatter-color-picker', 'scatter-color-text');
    setupColorSync('fill-color-picker', 'fill-color-text');

    // Mark initialization complete - now background changes will trigger updates
    initializingBackground = false;

    console.log('Controls initialized');
}

// ============================================================================
// Resize Observer Setup
// ============================================================================
function setupResizeObserver() {
    // Add resize handler to update overlay when window/image size changes
    window.addEventListener('resize', updateOverlay);

    // Use ResizeObserver to detect when the preview container changes size
    const previewContainer = document.getElementById('preview-container');
    if (previewContainer && typeof ResizeObserver !== 'undefined') {
        const resizeObserver = new ResizeObserver(() => {
            updateOverlay();
        });
        resizeObserver.observe(previewContainer);
    }
}

// ============================================================================
// Field-to-Element Synchronization
// ============================================================================
function setupFieldToElementSync() {
    // Map field IDs to element names
    const fieldToElement = {
        // Title, Labels & Caption section
        'title': 'title',
        'caption': 'caption',
        'xlabel': 'xlabel',
        'ylabel': 'ylabel',

        // Axis & Ticks section
        'xmin': 'xaxis',
        'xmax': 'xaxis',
        'ymin': 'yaxis',
        'ymax': 'yaxis',
        'x_n_ticks': 'xaxis',
        'y_n_ticks': 'yaxis',
        'hide_x_ticks': 'xaxis',
        'hide_y_ticks': 'yaxis',

        // Legend section
        'legend_visible': 'legend',
        'legend_loc': 'legend',
        'legend_frameon': 'legend',
        'legend_fontsize': 'legend',
        'legend_ncols': 'legend',
        'legend_x': 'legend',
        'legend_y': 'legend',
    };

    // Add focus listeners to all mapped fields
    for (const [fieldId, elementName] of Object.entries(fieldToElement)) {
        const field = document.getElementById(fieldId);
        if (field) {
            field.addEventListener('focus', function() {
                // Find the element in bboxes - for multi-panel, check ax_00 first
                let foundElement = elementName;
                if (elementBboxes[`ax_00_${elementName}`]) {
                    foundElement = `ax_00_${elementName}`;
                } else if (elementBboxes[elementName]) {
                    foundElement = elementName;
                } else {
                    // Try to find with axis prefix (e.g., ax_00_title)
                    for (const key of Object.keys(elementBboxes)) {
                        if (key.endsWith(`_${elementName}`)) {
                            foundElement = key;
                            break;
                        }
                    }
                }

                if (elementBboxes[foundElement]) {
                    hoveredElement = foundElement;
                    updateOverlay();
                }
            });

            // Also handle mouseenter for hover feedback
            field.addEventListener('mouseenter', function() {
                const helpText = this.getAttribute('title') || this.getAttribute('placeholder');
                if (helpText) {
                    setStatus(helpText, false);
                }
            });

            field.addEventListener('mouseleave', function() {
                setStatus('Ready', false);
            });
        }
    }
}

// ============================================================================
// Traces List Update
// ============================================================================
function updateTracesList() {
    const container = document.getElementById('traces-list');
    if (!container) return;

    container.innerHTML = '';
    if (!traces || traces.length === 0) return;

    traces.forEach((trace, idx) => {
        const item = document.createElement('div');
        item.className = 'trace-item';
        item.innerHTML = `
            <label>${trace.label || `Trace ${idx}`}</label>
            <input type="color" value="${trace.color || '#000000'}" onchange="updateTraceColor(${idx}, this.value)">
            <select onchange="updateTraceStyle(${idx}, this.value)">
                <option value="-" ${trace.linestyle === '-' ? 'selected' : ''}>Solid</option>
                <option value="--" ${trace.linestyle === '--' ? 'selected' : ''}>Dashed</option>
                <option value="-." ${trace.linestyle === '-.' ? 'selected' : ''}>Dash-dot</option>
                <option value=":" ${trace.linestyle === ':' ? 'selected' : ''}>Dotted</option>
            </select>
        `;
        container.appendChild(item);
    });
}

function updateTraceColor(idx, color) {
    if (traces[idx]) {
        traces[idx].color = color;
        scheduleUpdate();
    }
}

function updateTraceStyle(idx, style) {
    if (traces[idx]) {
        traces[idx].linestyle = style;
        scheduleUpdate();
    }
}

// ============================================================================
// Hover System Initialization (for SVG mode)
// ============================================================================
function initHoverSystemForElement(el) {
    // Initialize hover detection for inline SVG elements
    if (!el) return;

    el.addEventListener('mousemove', (e) => {
        // Find element at cursor position
        // This would require SVG-specific hit detection
        // For now, just update overlay
        updateOverlay();
    });

    el.addEventListener('click', (e) => {
        // Select element on click
        // This would require SVG-specific hit detection
    });
}

// ============================================================================
// Canvas Item Interaction (Multi-Panel)
// ============================================================================
function initCanvasItemInteraction(item, panelIdx, panelName) {
    const container = item.querySelector('.panel-card-container');
    if (!container) return;

    const img = container.querySelector('img');
    const overlay = container.querySelector('svg');
    if (!img || !overlay) return;

    // Wait for image to load to get dimensions
    img.addEventListener('load', () => {
        overlay.setAttribute('width', img.offsetWidth);
        overlay.setAttribute('height', img.offsetHeight);
        overlay.style.width = img.offsetWidth + 'px';
        overlay.style.height = img.offsetHeight + 'px';
    });

    // Mousemove for hover detection (accounting for object-fit:contain letterboxing)
    container.addEventListener('mousemove', (e) => {
        const panelCache = panelBboxesCache[panelName];
        if (!panelCache) return;

        const rect = img.getBoundingClientRect();
        const dims = getObjectFitContainDimensions(img);

        // Mouse position relative to container
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Adjust for letterbox offset to get position relative to actual rendered image
        const imgRelX = x - dims.offsetX;
        const imgRelY = y - dims.offsetY;

        // Check if click is within rendered image bounds
        if (imgRelX < 0 || imgRelY < 0 || imgRelX > dims.displayWidth || imgRelY > dims.displayHeight) {
            // Outside rendered image area (in letterbox region)
            if (panelHoveredElement !== null) {
                panelHoveredElement = null;
                updatePanelOverlay(overlay, panelCache.bboxes, panelCache.imgSize, rect.width, rect.height, null, null, img);
            }
            return;
        }

        // Scale to original image coordinates
        const scaleX = panelCache.imgSize.width / dims.displayWidth;
        const scaleY = panelCache.imgSize.height / dims.displayHeight;
        const imgX = imgRelX * scaleX;
        const imgY = imgRelY * scaleY;

        const element = findElementInPanelAt(imgX, imgY, panelCache.bboxes);
        if (element !== panelHoveredElement || activePanelCard !== item) {
            panelHoveredElement = element;
            activePanelCard = item;
            updatePanelOverlay(overlay, panelCache.bboxes, panelCache.imgSize, rect.width, rect.height, element, selectedElement, img);
        }
    });

    // Mouseleave to clear hover
    container.addEventListener('mouseleave', () => {
        panelHoveredElement = null;
        activePanelCard = null;
        const panelCache = panelBboxesCache[panelName];
        if (panelCache) {
            updatePanelOverlay(overlay, panelCache.bboxes, panelCache.imgSize, img.offsetWidth, img.offsetHeight, null, selectedElement, img);
        }
    });

    // Mousedown to start element drag (ONLY for legends and panel letters)
    container.addEventListener('mousedown', (e) => {
        if (panelHoveredElement && isDraggableElement(panelHoveredElement, panelBboxesCache[panelName]?.bboxes)) {
            // Only allow dragging of legends and panel letters (scientific rigor)
            startElementDrag(e, panelHoveredElement, panelName, img, panelBboxesCache[panelName].bboxes);
        }
    });

    // Click to select element (accounting for object-fit:contain letterboxing)
    container.addEventListener('click', (e) => {
        // Recalculate element at click position (in case hover didn't detect it)
        const panelCache = panelBboxesCache[panelName];
        if (!panelCache) return;

        const rect = img.getBoundingClientRect();
        const dims = getObjectFitContainDimensions(img);

        // Mouse position relative to container
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Adjust for letterbox offset
        const imgRelX = x - dims.offsetX;
        const imgRelY = y - dims.offsetY;

        // Check if click is within rendered image bounds
        if (imgRelX >= 0 && imgRelY >= 0 && imgRelX <= dims.displayWidth && imgRelY <= dims.displayHeight) {
            // Scale to original image coordinates
            const scaleX = panelCache.imgSize.width / dims.displayWidth;
            const scaleY = panelCache.imgSize.height / dims.displayHeight;
            const imgX = imgRelX * scaleX;
            const imgY = imgRelY * scaleY;

            const element = findElementInPanelAt(imgX, imgY, panelCache.bboxes);
            if (element) {
                loadPanelForEditing(panelName);
            }
        }
    });

    // Drag support for repositioning
    initPanelDrag(item, panelName);
}

// ============================================================================
// Panel Card Interaction (for panel grid view - if used)
// ============================================================================
function initPanelCardInteraction(card, panelIdx, panelName) {
    // Similar to initCanvasItemInteraction but for panel grid cards
    // Simplified version - delegates to initCanvasItemInteraction
    initCanvasItemInteraction(card, panelIdx, panelName);
}

console.log('main.js loaded');
