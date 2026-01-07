/**
 * Preview Management
 * Handles preview loading, panel navigation, and grid view
 */

// ============================================================================
// Initial Preview Loading
// ============================================================================
async function loadInitialPreview() {
    setStatus('Loading preview...', false);
    try {
        const resp = await fetch('/preview');
        const data = await resp.json();

        console.log('=== PREVIEW DATA RECEIVED ===');
        console.log('format:', data.format);
        console.log('img_size:', data.img_size);
        console.log('bboxes keys:', Object.keys(data.bboxes || {}));

        const previewContainer = document.getElementById('preview-container');
        const img = document.getElementById('preview');

        if (data.format === 'svg' && data.svg) {
            // Handle SVG: replace img with inline SVG
            const svgWrapper = document.createElement('div');
            svgWrapper.id = 'preview-svg-wrapper';
            svgWrapper.innerHTML = data.svg;
            svgWrapper.style.width = '100%';
            svgWrapper.style.maxHeight = '70vh';

            // Find the SVG element and set styles
            const svgEl = svgWrapper.querySelector('svg');
            if (svgEl) {
                svgEl.style.width = '100%';
                svgEl.style.height = 'auto';
                svgEl.style.maxHeight = '70vh';
                svgEl.id = 'preview';  // Keep same ID for event handlers
            }

            img.style.display = 'none';
            const existingWrapper = document.getElementById('preview-svg-wrapper');
            if (existingWrapper) existingWrapper.remove();
            previewContainer.appendChild(svgWrapper);
        } else if (data.image) {
            // Handle PNG: show as base64 image
            img.src = 'data:image/png;base64,' + data.image;
            img.style.display = 'block';
            const existingWrapper = document.getElementById('preview-svg-wrapper');
            if (existingWrapper) existingWrapper.remove();
        }

        if (data.bboxes) {
            elementBboxes = data.bboxes;
            originalBboxes = JSON.parse(JSON.stringify(data.bboxes));  // Deep copy
            if (data.bboxes._meta) {
                schemaMeta = data.bboxes._meta;
            }
        }
        if (data.img_size) {
            imgSize = data.img_size;
            originalImgSize = {...data.img_size};  // Copy
        }

        isShowingOriginalPreview = true;
        updateOverlay();
        setStatus('Preview loaded', false);

        // Initialize hover system for the SVG if needed
        if (data.format === 'svg') {
            const svgWrapper = document.getElementById('preview-svg-wrapper');
            if (svgWrapper) {
                initHoverSystemForElement(svgWrapper.querySelector('svg'));
            }
        }

        // Draw debug bboxes if debug mode is on
        if (debugMode) {
            updateOverlay();
        }

        // Handle multi-panel figure bundles
        if (data.panel_info && data.panel_info.panels) {
            panelData = data.panel_info;
            currentPanelIndex = data.panel_info.current_index || 0;
            console.log('Multi-panel figure detected:', panelData.panels.length, 'panels');
            loadPanelGrid();
        }

        // Start auto-update AFTER initial preview is loaded
        setAutoUpdateInterval();
    } catch (e) {
        setStatus('Error loading preview: ' + e.message, true);
        console.error('Preview load error:', e);
        // Start auto-update even on error so the editor works
        setAutoUpdateInterval();
    }
}

// ============================================================================
// Multi-Panel Grid Loading
// ============================================================================
async function loadPanelGrid() {
    if (!panelData || panelData.panels.length <= 1) {
        // Not a multi-panel bundle or only one panel
        document.getElementById('panel-grid-section').style.display = 'none';
        document.getElementById('preview-header').style.display = 'none';
        return;
    }

    console.log('Loading panel canvas for', panelData.panels.length, 'panels');

    // Hide single-panel preview completely for multi-panel bundles (unified canvas only)
    document.getElementById('preview-header').style.display = 'none';
    const previewWrapper = document.querySelector('.preview-wrapper');
    if (previewWrapper) {
        previewWrapper.style.display = 'none';
    }

    // Fetch all panel images with bboxes
    try {
        const resp = await fetch('/panels');
        const data = await resp.json();

        if (data.error) {
            console.error('Panel canvas error:', data.error);
            return;
        }

        const canvasEl = document.getElementById('panel-canvas');
        canvasEl.innerHTML = '';

        // Use figure layout to position panels as unified canvas (matching export)
        const hasLayout = data.layout && Object.keys(data.layout).length > 0;

        // Calculate scale factor: convert mm to pixels
        // Find total figure dimensions from layout
        let maxX = 0, maxY = 0;
        if (hasLayout) {
            Object.values(data.layout).forEach(l => {
                const right = (l.position?.x_mm || 0) + (l.size?.width_mm || 80);
                const bottom = (l.position?.y_mm || 0) + (l.size?.height_mm || 50);
                maxX = Math.max(maxX, right);
                maxY = Math.max(maxY, bottom);
            });
        }

        // Scale to fit canvas (max width ~700px for good display)
        const canvasMaxWidth = 700;
        const scale = hasLayout && maxX > 0 ? canvasMaxWidth / maxX : 3;  // ~3px per mm fallback
        canvasScale = scale;  // Store globally for drag conversions

        // Reset layout tracking
        panelLayoutMm = {};
        layoutModified = false;

        data.panels.forEach((panel, idx) => {
            // Store bboxes and imgSize in cache for interactive hover/click
            if (panel.bboxes && panel.img_size) {
                panelBboxesCache[panel.name] = {
                    bboxes: panel.bboxes,
                    imgSize: panel.img_size
                };
            }

            // Use figure layout for positioning (unified canvas like export)
            let pos, posMm;
            if (panel.layout && panel.layout.position && panel.layout.size) {
                const x_mm = panel.layout.position.x_mm || 0;
                const y_mm = panel.layout.position.y_mm || 0;
                const width_mm = panel.layout.size.width_mm || 80;
                const height_mm = panel.layout.size.height_mm || 50;
                pos = {
                    x: x_mm * scale,
                    y: y_mm * scale,
                    width: width_mm * scale,
                    height: height_mm * scale,
                };
                posMm = { x_mm, y_mm, width_mm, height_mm };
            } else {
                // Fallback grid layout if no figure layout
                const cols = Math.ceil(Math.sqrt(data.panels.length));
                const baseWidth = 220, baseHeight = 180, padding = 15;
                const col = idx % cols;
                const row = Math.floor(idx / cols);
                pos = {
                    x: padding + col * (baseWidth + padding),
                    y: padding + row * (baseHeight + padding),
                    width: baseWidth,
                    height: baseHeight,
                };
                // Convert to mm for fallback
                posMm = {
                    x_mm: pos.x / scale,
                    y_mm: pos.y / scale,
                    width_mm: pos.width / scale,
                    height_mm: pos.height / scale,
                };
            }
            panelPositions[panel.name] = pos;
            panelLayoutMm[panel.name] = posMm;

            const item = document.createElement('div');
            item.className = 'panel-canvas-item' + (idx === currentPanelIndex ? ' active' : '');
            item.dataset.panelIndex = idx;
            item.dataset.panelName = panel.name;
            item.style.left = pos.x + 'px';
            item.style.top = pos.y + 'px';
            item.style.width = pos.width + 'px';
            item.style.height = pos.height + 'px';

            if (panel.image) {
                item.innerHTML = `
                    <span class="panel-canvas-label">${panel.name}</span>
                    <span class="panel-position-indicator" id="pos-${panel.name}"></span>
                    <div class="panel-drag-handle" title="Drag to move panel">⋮⋮</div>
                    <div class="panel-card-container">
                        <img src="data:image/png;base64,${panel.image}" alt="Panel ${panel.name}">
                        <svg class="panel-card-overlay" id="panel-overlay-${idx}"></svg>
                    </div>
                `;
            } else {
                item.innerHTML = `
                    <span class="panel-canvas-label">${panel.name}</span>
                    <span class="panel-position-indicator" id="pos-${panel.name}"></span>
                    <div class="panel-drag-handle" title="Drag to move panel">⋮⋮</div>
                    <div style="padding: 20px; color: var(--text-muted);">No preview</div>
                `;
            }

            // Add interactive event handlers (hover, click for element selection)
            // Note: initCanvasItemInteraction already calls initPanelDrag internally
            initCanvasItemInteraction(item, idx, panel.name);

            canvasEl.appendChild(item);
        });

        // Update canvas size to fit all panels (unified canvas)
        const canvasHeight = Math.max(...Object.values(panelPositions).map(p => p.y + p.height)) + 10;
        const canvasWidth = Math.max(...Object.values(panelPositions).map(p => p.x + p.width)) + 10;
        canvasEl.style.minHeight = Math.max(400, canvasHeight) + 'px';
        canvasEl.style.minWidth = canvasWidth + 'px';

        // Update panel indicator
        updatePanelIndicator();

        // Show unified canvas for multi-panel figures
        showingPanelGrid = true;
        document.getElementById('panel-grid-section').style.display = 'block';
    } catch (e) {
        console.error('Error loading panels:', e);
    }
}

// ============================================================================
// Panel Navigation
// ============================================================================
function togglePanelGrid() {
    showingPanelGrid = !showingPanelGrid;
    if (showingPanelGrid) {
        loadPanelGrid();
    } else {
        document.getElementById('panel-grid-section').style.display = 'none';
    }
}

function prevPanel() {
    if (currentPanelIndex > 0) {
        loadPanelForEditing(panelData.panels[currentPanelIndex - 1].name);
    }
}

function nextPanel() {
    if (currentPanelIndex < panelData.panels.length - 1) {
        loadPanelForEditing(panelData.panels[currentPanelIndex + 1].name);
    }
}

function updatePanelIndicator() {
    if (!panelData) return;

    const indicator = document.getElementById('panel-indicator');
    const prevBtn = document.getElementById('btn-prev-panel');
    const nextBtn = document.getElementById('btn-next-panel');

    // Update indicator text (if elements exist)
    if (indicator) {
        indicator.textContent = `${currentPanelIndex + 1} / ${panelData.panels.length}`;
    }

    // Update button states
    if (prevBtn) prevBtn.disabled = currentPanelIndex === 0;
    if (nextBtn) nextBtn.disabled = currentPanelIndex === panelData.panels.length - 1;
}
