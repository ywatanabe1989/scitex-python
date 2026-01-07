/**
 * API Communication
 * Handles all server requests for preview, update, save, etc.
 */

// ============================================================================
// Preview and Update API
// ============================================================================
function updatePreview() {
    const data = collectOverrides();

    // Skip auto-update if showing original preview (user hasn't explicitly requested update)
    if (isShowingOriginalPreview) {
        return;
    }

    setStatus('Updating preview...');

    // Preserve current selection to restore after update
    const prevSelectedElement = selectedElement;

    fetch('/update', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    })
    .then(r => r.json())
    .then(result => {
        // Remove SVG wrapper if exists, show img element for re-rendered preview
        const svgWrapper = document.getElementById('preview-svg-wrapper');
        if (svgWrapper) {
            svgWrapper.style.display = 'none';
        }
        document.getElementById('preview').style.display = 'block';

        // Mark that we're no longer showing original preview
        isShowingOriginalPreview = false;

        if (result.image) {
            document.getElementById('preview').src = 'data:image/png;base64,' + result.image;

            // Store schema v0.3 metadata if available
            if (result.schema_meta) {
                schemaMeta = result.schema_meta;
            }
        }
        elementBboxes = result.bboxes || {};
        imgSize = result.img_size || {width: 0, height: 0};
        setStatus('Preview updated');

        // Restore selection if the element still exists in the new bboxes
        if (prevSelectedElement && elementBboxes[prevSelectedElement]) {
            selectedElement = prevSelectedElement;
            updateOverlay();
        } else {
            selectedElement = null;
            updateOverlay();
        }
    })
    .catch(err => {
        setStatus('Error updating preview: ' + err, true);
    });
}

// ============================================================================
// Save API
// ============================================================================
function saveToBundle() {
    const data = collectOverrides();
    setStatus('Saving...');
    return fetch('/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    })
    .then(r => r.json())
    .then(result => {
        setStatus(result.status || 'Saved successfully');
    })
    .catch(err => {
        setStatus('Error saving: ' + err, true);
        throw err;
    });
}

// ============================================================================
// Export API
// ============================================================================
function exportToFormat(format) {
    const data = collectOverrides();
    setStatus('Exporting to ' + format + '...');

    // First export to bundle
    fetch('/export', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({...data, format: format})
    })
    .then(r => r.json())
    .then(result => {
        setStatus(result.status || 'Exported successfully');
        // Then trigger download
        downloadFile(result.path, format);
    })
    .catch(err => {
        setStatus('Error exporting: ' + err, true);
    });
}

function downloadFile(path, format) {
    const filename = path.split('/').pop();
    fetch('/download/' + filename)
        .then(r => r.blob())
        .then(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        });
}

// ============================================================================
// Panel API (Multi-panel figures)
// ============================================================================

// Helper to normalize panel names for comparison (strip .plot extensions)
function normalizePanelName(name) {
    if (typeof name !== 'string') return '';
    return name.replace('.plot', '').replace('.plot', '');
}

function loadPanelForEditing(panelName) {
    setStatus('Loading panel ' + panelName + '...');

    // Find panel index from panel name
    // Handle both formats: array of strings ["A", "B"] or array of objects [{name: "A"}, ...]
    // Also handle extension mismatches (e.g., "A" vs "A.plot")
    let panelIndex = -1;
    const normalizedSearch = normalizePanelName(panelName);

    if (panelData && panelData.panels) {
        panelIndex = panelData.panels.findIndex(p => {
            // Check if it's a string or an object with name property
            if (typeof p === 'string') {
                return normalizePanelName(p) === normalizedSearch;
            } else if (p && p.name) {
                return normalizePanelName(p.name) === normalizedSearch;
            }
            return false;
        });
    }

    if (panelIndex === -1) {
        setStatus('Panel not found: ' + panelName, true);
        return Promise.reject(new Error('Panel not found: ' + panelName));
    }

    return fetch('/switch_panel/' + panelIndex)
    .then(r => r.json())
    .then(result => {
        if (result.error) {
            throw new Error(result.error);
        }

        // Update panel state
        currentPanelIndex = panelIndex;

        // Update preview image
        if (result.image) {
            document.getElementById('preview').src = 'data:image/png;base64,' + result.image;
        }

        // Update bboxes and overlays
        elementBboxes = result.bboxes || {};
        imgSize = result.img_size || {width: 0, height: 0};

        // Sync bboxes cache
        panelBboxesCache[panelName] = {bboxes: elementBboxes, imgSize: imgSize};

        // Update overrides
        if (result.overrides) {
            overrides = result.overrides;
        }

        // Select the element that was clicked
        if (result.selected_element) {
            selectedElement = result.selected_element;
        }

        // Scroll to section and show properties
        scrollToSection(selectedElement);

        // Keep unified canvas view only - don't show single-panel preview
        showingPanelGrid = true;
        // Update panel path display in right panel header
        const panelPath = document.getElementById('current-panel-path');
        if (panelPath) {
            panelPath.textContent = panelName || 'Single Panel';
        }

        setStatus('Panel loaded');
        return result;
    })
    .catch(err => {
        setStatus('Error loading panel: ' + err, true);
        throw err;
    });
}

// ============================================================================
// Layout Save API
// ============================================================================
function saveLayout() {
    const data = {
        layout: panelLayoutMm,
        overrides: overrides
    };

    setStatus('Saving layout...');

    return fetch('/save_layout', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    })
    .then(r => r.json())
    .then(result => {
        setStatus('Layout saved');
        layoutModified = false;
    })
    .catch(err => {
        setStatus('Error saving layout: ' + err, true);
        throw err;
    });
}

// ============================================================================
// Element Position Update API
// ============================================================================
function updateElementPosition(panelName, elementName, position) {
    const data = {
        panel_name: panelName,
        element_name: elementName,
        position: position,
        overrides: overrides
    };

    return fetch('/save_element_position', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    })
    .then(r => r.json())
    .then(result => {
        setStatus('Element position updated');
        // Reload panel to show updated position
        return loadPanelForEditing(panelName);
    })
    .catch(err => {
        setStatus('Error updating element position: ' + err, true);
        throw err;
    });
}

// ============================================================================
// Auto-Update Scheduling
// ============================================================================
function scheduleUpdate() {
    clearTimeout(updateTimer);
    updateTimer = setTimeout(updatePreview, DEBOUNCE_DELAY);
}

function setAutoUpdateInterval() {
    if (autoUpdateIntervalId) {
        clearInterval(autoUpdateIntervalId);
        autoUpdateIntervalId = null;
    }

    const intervalMs = parseInt(document.getElementById('auto_update_interval').value);
    if (intervalMs > 0) {
        autoUpdateIntervalId = setInterval(() => {
            updatePreview();
        }, intervalMs);
    }
}
