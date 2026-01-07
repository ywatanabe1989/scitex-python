/**
 * Bounding Box Detection
 * Element hit detection and proximity calculations
 */

// ============================================================================
// Hit Detection - Main Entry Point
// ============================================================================
function findElementAt(x, y) {
    // Multi-panel aware hit detection with specificity hierarchy:
    // 1. Data elements with legacy points - proximity detection (correct saved-image coords)
    // 2. Small elements (labels, ticks, legends, bars, fills)
    // 3. Panel bboxes - lowest priority (fallback)

    const PROXIMITY_THRESHOLD = 15;
    const SCATTER_THRESHOLD = 20;  // Larger threshold for scatter points

    // First: Check for data elements using legacy points (in saved-image coordinates)
    let closestDataElement = null;
    let minDistance = Infinity;

    for (const [name, bbox] of Object.entries(elementBboxes)) {
        if (name === '_meta') continue;  // Skip metadata entry

        // Prioritize legacy points array (already in correct saved-image coordinates)
        if (bbox.points && bbox.points.length > 0) {
            // Check if cursor is within general bbox area first
            if (x >= bbox.x0 - SCATTER_THRESHOLD && x <= bbox.x1 + SCATTER_THRESHOLD &&
                y >= bbox.y0 - SCATTER_THRESHOLD && y <= bbox.y1 + SCATTER_THRESHOLD) {

                const elementType = bbox.element_type || 'line';
                let dist;

                if (elementType === 'scatter') {
                    // For scatter, find distance to nearest point
                    dist = distanceToNearestPoint(x, y, bbox.points);
                } else {
                    // For lines, find distance to line segments
                    dist = distanceToLine(x, y, bbox.points);
                }

                if (dist < minDistance) {
                    minDistance = dist;
                    closestDataElement = name;
                }
            }
        }
    }

    // Use appropriate threshold based on element type
    if (closestDataElement) {
        const bbox = elementBboxes[closestDataElement];
        const threshold = (bbox.element_type === 'scatter') ? SCATTER_THRESHOLD : PROXIMITY_THRESHOLD;
        if (minDistance <= threshold) {
            return closestDataElement;
        }
    }

    // Second: Collect all bbox matches, excluding panels and data elements with points
    const elementMatches = [];
    const panelMatches = [];

    for (const [name, bbox] of Object.entries(elementBboxes)) {
        if (x >= bbox.x0 && x <= bbox.x1 && y >= bbox.y0 && y <= bbox.y1) {
            const area = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0);
            const isPanel = bbox.is_panel || name.endsWith('_panel');
            const hasPoints = bbox.points && bbox.points.length > 0;

            if (hasPoints) {
                // Already handled above with proximity
                continue;
            } else if (isPanel) {
                panelMatches.push({name, area, bbox});
            } else {
                elementMatches.push({name, area, bbox});
            }
        }
    }

    // Return smallest non-panel element if any
    if (elementMatches.length > 0) {
        elementMatches.sort((a, b) => a.area - b.area);
        return elementMatches[0].name;
    }

    // Fallback to panel selection (useful for multi-panel figures)
    if (panelMatches.length > 0) {
        panelMatches.sort((a, b) => a.area - b.area);
        return panelMatches[0].name;
    }

    return null;
}

// ============================================================================
// Find All Overlapping Elements (for cycle selection)
// ============================================================================
function findAllElementsAt(x, y) {
    // Find all elements at cursor position (for cycle selection)
    // Returns array sorted by specificity (most specific first)
    const PROXIMITY_THRESHOLD = 15;
    const SCATTER_THRESHOLD = 20;

    const results = [];

    for (const [name, bbox] of Object.entries(elementBboxes)) {
        let match = false;
        let distance = Infinity;
        let priority = 0;  // Lower = more specific

        const hasPoints = bbox.points && bbox.points.length > 0;
        const elementType = bbox.element_type || '';
        const isPanel = bbox.is_panel || name.endsWith('_panel');

        // Check data elements with points (lines, scatter)
        if (hasPoints) {
            if (x >= bbox.x0 - SCATTER_THRESHOLD && x <= bbox.x1 + SCATTER_THRESHOLD &&
                y >= bbox.y0 - SCATTER_THRESHOLD && y <= bbox.y1 + SCATTER_THRESHOLD) {

                if (elementType === 'scatter') {
                    distance = distanceToNearestPoint(x, y, bbox.points);
                    if (distance <= SCATTER_THRESHOLD) {
                        match = true;
                        priority = 1;  // Scatter points = high priority
                    }
                } else {
                    distance = distanceToLine(x, y, bbox.points);
                    if (distance <= PROXIMITY_THRESHOLD) {
                        match = true;
                        priority = 2;  // Lines = high priority
                    }
                }
            }
        }

        // Check bbox containment
        if (x >= bbox.x0 && x <= bbox.x1 && y >= bbox.y0 && y <= bbox.y1) {
            const area = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0);

            if (!match) {
                match = true;
                distance = 0;
            }

            if (isPanel) {
                priority = 100;  // Panels = lowest priority
            } else if (!hasPoints) {
                // Small elements like labels, ticks - use area for priority
                priority = 10 + Math.min(area / 10000, 50);
            }
        }

        if (match) {
            results.push({ name, distance, priority, bbox });
        }
    }

    // Sort by priority (lower first), then by distance
    results.sort((a, b) => {
        if (a.priority !== b.priority) return a.priority - b.priority;
        return a.distance - b.distance;
    });

    return results.map(r => r.name);
}

// ============================================================================
// Panel Element Detection (for multi-panel canvas)
// ============================================================================
function findElementInPanelAt(x, y, bboxes) {
    const PROXIMITY_THRESHOLD = 15;
    const SCATTER_THRESHOLD = 20;

    let closestDataElement = null;
    let minDistance = Infinity;

    // Check data elements with points
    for (const [name, bbox] of Object.entries(bboxes)) {
        if (name === '_meta') continue;

        if (bbox.points && bbox.points.length > 0) {
            if (x >= bbox.x0 - SCATTER_THRESHOLD && x <= bbox.x1 + SCATTER_THRESHOLD &&
                y >= bbox.y0 - SCATTER_THRESHOLD && y <= bbox.y1 + SCATTER_THRESHOLD) {

                const elementType = bbox.element_type || 'line';
                let dist;

                if (elementType === 'scatter') {
                    dist = distanceToNearestPoint(x, y, bbox.points);
                } else {
                    dist = distanceToLine(x, y, bbox.points);
                }

                if (dist < minDistance) {
                    minDistance = dist;
                    closestDataElement = name;
                }
            }
        }
    }

    if (closestDataElement) {
        const bbox = bboxes[closestDataElement];
        const threshold = (bbox.element_type === 'scatter') ? SCATTER_THRESHOLD : PROXIMITY_THRESHOLD;
        if (minDistance <= threshold) {
            return closestDataElement;
        }
    }

    // Check bbox containment for other elements
    const elementMatches = [];
    for (const [name, bbox] of Object.entries(bboxes)) {
        if (name === '_meta') continue;

        if (x >= bbox.x0 && x <= bbox.x1 && y >= bbox.y0 && y <= bbox.y1) {
            const area = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0);
            const isPanel = bbox.is_panel || name.endsWith('_panel');

            if (!isPanel) {
                elementMatches.push({name, area, bbox});
            }
        }
    }

    if (elementMatches.length > 0) {
        elementMatches.sort((a, b) => a.area - b.area);
        return elementMatches[0].name;
    }

    return null;
}

// ============================================================================
// Distance Calculations
// ============================================================================
function distanceToNearestPoint(px, py, points) {
    // Find distance to nearest point in scatter
    if (!Array.isArray(points) || points.length === 0) return Infinity;
    let minDist = Infinity;
    for (const pt of points) {
        if (!Array.isArray(pt) || pt.length < 2) continue;
        const [x, y] = pt;
        const dist = Math.sqrt((px - x) ** 2 + (py - y) ** 2);
        if (dist < minDist) minDist = dist;
    }
    return minDist;
}

function distanceToLine(px, py, points) {
    if (!Array.isArray(points) || points.length < 2) return Infinity;
    let minDist = Infinity;
    for (let i = 0; i < points.length - 1; i++) {
        const pt1 = points[i];
        const pt2 = points[i + 1];
        if (!Array.isArray(pt1) || pt1.length < 2) continue;
        if (!Array.isArray(pt2) || pt2.length < 2) continue;
        const [x1, y1] = pt1;
        const [x2, y2] = pt2;
        const dist = distanceToSegment(px, py, x1, y1, x2, y2);
        if (dist < minDist) minDist = dist;
    }
    return minDist;
}

function distanceToSegment(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const lenSq = dx * dx + dy * dy;

    if (lenSq === 0) {
        return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2);
    }

    let t = ((px - x1) * dx + (py - y1) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));

    const projX = x1 + t * dx;
    const projY = y1 + t * dy;

    return Math.sqrt((px - projX) ** 2 + (py - projY) ** 2);
}

// ============================================================================
// Polygon Test
// ============================================================================
function pointInPolygon(px, py, polygon) {
    if (!Array.isArray(polygon) || polygon.length < 3) return false;

    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const ptI = polygon[i];
        const ptJ = polygon[j];
        if (!Array.isArray(ptI) || ptI.length < 2) continue;
        if (!Array.isArray(ptJ) || ptJ.length < 2) continue;
        const [xi, yi] = ptI;
        const [xj, yj] = ptJ;

        if (((yi > py) !== (yj > py)) &&
            (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

// ============================================================================
// Geometry Extraction
// ============================================================================
function getGeometryPoints(bbox) {
    // Extract points for overlay drawing
    // Returns array of [x, y] points or null

    // For scatter: use points array directly
    if (bbox.element_type === 'scatter' && bbox.points) {
        return bbox.points;
    }

    // For lines: use path_simplified
    if (bbox.element_type === 'line' && bbox.path_simplified) {
        return bbox.path_simplified;
    }

    // For fills/polygons: use polygon
    if (bbox.polygon) {
        return bbox.polygon;
    }

    return null;
}

// ============================================================================
// Axes Coordinate Transformation (for future use with geometry_px)
// ============================================================================
function axesLocalToImage(axLocalX, axLocalY, axesBbox) {
    // axesBbox has: x, y, width, height in figure pixel coordinates
    // The local editor uses tight layout which shifts coordinates
    // For now we use the existing image coordinates from bboxes
    return {x: axLocalX, y: axLocalY};
}
