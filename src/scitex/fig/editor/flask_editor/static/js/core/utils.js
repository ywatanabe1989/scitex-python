/**
 * Utility Functions
 * Common helper functions used across the editor
 */

// ============================================================================
// Theme Utilities
// ============================================================================
function isDarkMode() {
    return document.body.classList.contains('dark-theme');
}

// ============================================================================
// Image Dimension Calculations
// ============================================================================
function getObjectFitContainDimensions(img) {
    const containerWidth = img.clientWidth;
    const containerHeight = img.clientHeight;
    const imgWidth = img.naturalWidth;
    const imgHeight = img.naturalHeight;

    // Handle edge cases
    if (!containerWidth || !containerHeight || !imgWidth || !imgHeight) {
        return {
            displayWidth: containerWidth,
            displayHeight: containerHeight,
            offsetX: 0,
            offsetY: 0
        };
    }

    // Calculate scale factor for object-fit: contain
    const containerRatio = containerWidth / containerHeight;
    const imgRatio = imgWidth / imgHeight;

    let displayWidth, displayHeight, offsetX, offsetY;

    if (imgRatio > containerRatio) {
        // Image is wider than container - fit to width, letterbox top/bottom
        displayWidth = containerWidth;
        displayHeight = containerWidth / imgRatio;
        offsetX = 0;
        offsetY = (containerHeight - displayHeight) / 2;
    } else {
        // Image is taller than container - fit to height, letterbox left/right
        displayHeight = containerHeight;
        displayWidth = containerHeight * imgRatio;
        offsetX = (containerWidth - displayWidth) / 2;
        offsetY = 0;
    }

    return {displayWidth, displayHeight, offsetX, offsetY};
}

// ============================================================================
// Status Display
// ============================================================================
function setStatus(msg, isError = false) {
    document.getElementById('status').textContent = msg;
    document.getElementById('status').style.color = isError ? 'red' : '';

    // Show/hide spinner for loading states
    if (msg.toLowerCase().includes('updating') || msg.toLowerCase().includes('loading')) {
        // Show global overlay (visible for both single and multi-panel views)
        const globalOverlay = document.getElementById('loading-overlay');
        if (globalOverlay) globalOverlay.style.display = 'flex';

        // Also show local overlay if visible
        const localOverlay = document.getElementById('overlay-loading');
        if (localOverlay) localOverlay.style.display = 'flex';
    } else {
        // Hide both overlays
        const globalOverlay = document.getElementById('loading-overlay');
        if (globalOverlay) globalOverlay.style.display = 'none';
        const localOverlay = document.getElementById('overlay-loading');
        if (localOverlay) localOverlay.style.display = 'none';
    }
}

// ============================================================================
// Loading State Management
// ============================================================================
function showLoading(message = 'Loading...') {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = 'flex';
        const text = overlay.querySelector('.loading-text');
        if (text) text.textContent = message;
    }
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// ============================================================================
// Section Management
// ============================================================================
function toggleSection(header) {
    const content = header.nextElementSibling;
    content.style.display = content.style.display === 'none' ? 'block' : 'none';
}

function expandSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        const header = section.querySelector('h3');
        const content = section.querySelector('.section-content');
        if (content && content.style.display === 'none') {
            content.style.display = 'block';
            if (header) {
                header.classList.add('expanded');
            }
        }

        // Scroll the section into view
        section.scrollIntoView({behavior: 'smooth', block: 'start'});
    }
}

// ============================================================================
// Scroll to Section by Element Name
// ============================================================================
function scrollToSection(elementName) {
    if (!elementName) return;

    // Map element names to section IDs
    let sectionId = null;
    const lowerName = elementName.toLowerCase();

    if (lowerName.includes('legend')) {
        sectionId = 'section-legend';
    } else if (lowerName.includes('title')) {
        sectionId = 'section-labels';
    } else if (lowerName.includes('xlabel') || lowerName.includes('ylabel') || lowerName.includes('label')) {
        sectionId = 'section-labels';
    } else if (lowerName.includes('xaxis') || lowerName.includes('yaxis') || lowerName.includes('axis') || lowerName.includes('tick') || lowerName.includes('spine')) {
        sectionId = 'section-ticks';
    } else if (lowerName.includes('trace') || lowerName.includes('line') || lowerName.includes('scatter') || lowerName.includes('bar')) {
        sectionId = 'section-traces';
    } else if (lowerName.includes('panel') || lowerName.includes('ax_')) {
        sectionId = 'section-selected';
    }

    if (sectionId) {
        expandSection(sectionId);
    }
}

// ============================================================================
// Color Synchronization
// ============================================================================
function setupColorSync(colorId, textId) {
    const colorInput = document.getElementById(colorId);
    const textInput = document.getElementById(textId);
    if (colorInput && textInput) {
        colorInput.addEventListener('input', function() {
            textInput.value = this.value;
        });
        textInput.addEventListener('input', function() {
            colorInput.value = this.value;
        });
    }
}

// ============================================================================
// Dimension Unit Conversion
// ============================================================================
function setDimensionUnit(unit) {
    if (dimensionUnit === unit) return;

    const oldUnit = dimensionUnit;
    dimensionUnit = unit;

    // Get current values
    const widthInput = document.getElementById('fig_width');
    const heightInput = document.getElementById('fig_height');

    // Convert values
    if (unit === 'mm') {
        // inch to mm
        if (widthInput.value) {
            widthInput.value = (parseFloat(widthInput.value) * INCH_TO_MM).toFixed(1);
        }
        if (heightInput.value) {
            heightInput.value = (parseFloat(heightInput.value) * INCH_TO_MM).toFixed(1);
        }
    } else {
        // mm to inch
        if (widthInput.value) {
            widthInput.value = (parseFloat(widthInput.value) * MM_TO_INCH).toFixed(2);
        }
        if (heightInput.value) {
            heightInput.value = (parseFloat(heightInput.value) * MM_TO_INCH).toFixed(2);
        }
    }

    // Update values and labels
    document.getElementById('unit_width').textContent = unit;
    document.getElementById('unit_height').textContent = unit;

    // Update button states
    document.getElementById('btn_mm').classList.toggle('active', unit === 'mm');
    document.getElementById('btn_inch').classList.toggle('active', unit === 'inch');
}

// ============================================================================
// Figure Size Conversion
// ============================================================================
function getFigSizeInches() {
    const width = parseFloat(document.getElementById('fig_width').value);
    const height = parseFloat(document.getElementById('fig_height').value);

    if (dimensionUnit === 'mm') {
        return {
            width: width * MM_TO_INCH,
            height: height * MM_TO_INCH
        };
    }
    return {width, height};
}

// ============================================================================
// Significance Class Helper
// ============================================================================
function getSigClass(stars) {
    if (stars >= 3) return 'sig-high';
    if (stars >= 2) return 'sig-med';
    if (stars >= 1) return 'sig-low';
    return '';
}

// ============================================================================
// Debounce Helper
// ============================================================================
function debounce(func, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}
