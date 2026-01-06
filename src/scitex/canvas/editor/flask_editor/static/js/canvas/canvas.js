/**
 * Canvas View Management
 * Handles the unified canvas view for multi-panel figures
 */

// ============================================================================
// Canvas Mode Control
// ============================================================================
function setCanvasMode(mode) {
    canvasMode = mode;
    document.getElementById('canvas-grid').classList.toggle('canvas-mode', mode === 'canvas');
    document.getElementById('canvas-grid').classList.toggle('grid-mode', mode === 'grid');
}

// ============================================================================
// Canvas Rendering
// ============================================================================
function renderCanvasView() {
    const container = document.getElementById('canvas-grid');
    container.innerHTML = '';

    // Fetch panels if not cached
    if (!panelData || !panelData.panels) {
        return;
    }

    if (canvasMode === 'canvas') {
        // Calculate canvas size based on number of panels
        const panels = panelData.panels;

        panels.forEach((panel, idx) => {
            const item = document.createElement('div');
            item.className = 'panel-canvas-item';
            item.dataset.panelName = panel.name;

            // Initialize position if not set
            if (!panelPositions[panel.name]) {
                panelPositions[panel.name] = {
                    x: idx * 150,
                    y: idx * 150,
                    width: panel.width_px || 400,
                    height: panel.height_px || 300
                };
            }

            const pos = panelPositions[panel.name];
            item.style.left = pos.x + 'px';
            item.style.top = pos.y + 'px';
            item.style.width = pos.width + 'px';
            item.style.height = pos.height + 'px';

            item.innerHTML = `
                <div class="panel-drag-handle">☰</div>
                <div class="panel-label">${panel.name}</div>
                <img src="data:image/png;base64,${panel.image_base64}" style="width: 100%; height: 100%; object-fit: contain;">
                <canvas class="panel-overlay"></canvas>
            `;

            container.appendChild(item);

            // Double-click to edit
            item.addEventListener('dblclick', () => {
                loadPanelForEditing(panel.name);
            });

            // Drag start
            initPanelDrag(item, panel.name);
        });

        // Update canvas height to fit all panels
        updateCanvasSize();
    } else {
        // Grid mode - use CSS grid layout (simpler)
        loadPanelGrid();
    }
}

// ============================================================================
// Interactive Element Detection Helper
// ============================================================================
function isInteractiveElement(target) {
    // SVG paths with hover-path class are interactive elements
    if (target.classList && target.classList.contains('hover-path')) {
        return true;
    }
    // Check parent elements for hover-path (click might be on child)
    let parent = target.parentElement;
    while (parent) {
        if (parent.tagName === 'path' || parent.classList.contains('hover-path')) {
            // Path elements in SVG overlay are interactive
            return true;
        }
        parent = parent.parentElement;
    }
    return false;
}

// ============================================================================
// Canvas Size Management
// ============================================================================
function updateCanvasSize() {
    // Find the maximum extent of all panels
    let maxX = 0;
    let maxY = 0;

    Object.values(panelPositions).forEach(pos => {
        maxX = Math.max(maxX, pos.x + pos.width);
        maxY = Math.max(maxY, pos.y + pos.height);
    });

    // Add some padding
    const container = document.getElementById('canvas-grid');
    if (container && canvasMode === 'canvas') {
        container.style.minHeight = (maxY + 100) + 'px';
        container.style.minWidth = (maxX + 100) + 'px';
    }
}

// ============================================================================
// Canvas Zoom Functions
// ============================================================================
function zoomCanvas(factor) {
    canvasZoom = Math.max(0.1, Math.min(5.0, canvasZoom * factor));
    const container = document.getElementById('canvas-grid');
    if (container) {
        container.style.transform = `scale(${canvasZoom})`;
        container.style.transformOrigin = 'top left';
    }
}

function fitCanvasToWindow() {
    const container = document.getElementById('canvas-grid');
    if (!container) return;

    const containerWidth = container.scrollWidth;
    const windowWidth = window.innerWidth - 400; // Account for side panels
    canvasZoom = Math.min(1.0, windowWidth / containerWidth);
    container.style.transform = `scale(${canvasZoom})`;
    container.style.transformOrigin = 'top left';
}

function resizeCanvas(factor) {
    const container = document.getElementById('canvas-grid');
    if (!container) return;

    // Scale all panel positions and sizes
    Object.keys(panelPositions).forEach(name => {
        const pos = panelPositions[name];
        pos.x *= factor;
        pos.y *= factor;
        pos.width *= factor;
        pos.height *= factor;
    });

    renderCanvasView();
}

// ============================================================================
// Panel Layout Update from DOM
// ============================================================================
function updatePanelLayoutFromDOM() {
    document.querySelectorAll('.panel-canvas-item').forEach(item => {
        const name = item.dataset.panelName;
        const rect = item.getBoundingClientRect();
        panelPositions[name] = {
            x: parseFloat(item.style.left),
            y: parseFloat(item.style.top),
            width: rect.width,
            height: rect.height
        };
    });
}
