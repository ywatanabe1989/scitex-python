/**
 * Global State Variables
 * Central state management for the Flask figure editor
 *
 * Note: `overrides` is set in the HTML page via inline script before loading modules.
 * This allows Flask to inject the initial data using Jinja2 templating.
 *
 * IMPORTANT: Variables use `var` (not `let`) to ensure they're accessible
 * as globals across all script files loaded via separate <script> tags.
 */

// ============================================================================
// Override Data and Element State
// ============================================================================
// `overrides` is defined in HTML before this script loads
// Fallback to empty object if not defined
if (typeof overrides === 'undefined') {
    var overrides = {};
}
var traces = overrides.traces || [];
var elementBboxes = {};
var imgSize = {width: 0, height: 0};
var hoveredElement = null;
var selectedElement = null;
var debugMode = false;  // Debug mode to show all hit areas
var isShowingOriginalPreview = false;  // True when showing existing SVG/PNG from bundle
var originalBboxes = null;  // Store original bboxes from /preview
var originalImgSize = null;  // Store original img size from /preview

// Schema metadata storage
var schemaMeta = null;

// ============================================================================
// Multi-Panel State
// ============================================================================
var panelData = null;  // Panel info from /preview
var currentPanelIndex = 0;
var showingPanelGrid = false;
var panelBboxesCache = {};  // Cache bboxes per panel {panelName: {bboxes, imgSize}}
var activePanelCard = null;  // Currently active panel card for hover/click
var panelHoveredElement = null;  // Hovered element in panel grid
var panelDebugMode = false;  // Show hit regions in panel grid

// ============================================================================
// Cycle Selection State
// ============================================================================
var elementsAtCursor = [];  // All elements at current cursor position
var currentCycleIndex = 0;  // Current index in cycle

// ============================================================================
// Dimension Settings
// ============================================================================
var dimensionUnit = 'mm';
var MM_TO_INCH = 1 / 25.4;
var INCH_TO_MM = 25.4;

// ============================================================================
// Hitmap State
// ============================================================================
var hitmapCanvas = null;
var hitmapCtx = null;
var hitmapColorMap = {};  // Maps RGB string -> element info
var hitmapLoaded = false;
var hitmapImgSize = {width: 0, height: 0};

// ============================================================================
// Background Settings
// ============================================================================
var backgroundType = 'transparent';
var initializingBackground = true;  // Flag to prevent updates during init

// ============================================================================
// Canvas Mode and Layout State
// ============================================================================
var canvasMode = 'grid';  // 'grid' or 'canvas'
var panelPositions = {};  // Store panel positions {name: {x, y, width, height}} in pixels
var panelLayoutMm = {};   // Store panel positions in mm for saving {name: {x_mm, y_mm, width_mm, height_mm}}
var canvasScale = 3;      // Scale factor: pixels per mm (updated in loadPanelGrid)
var draggedPanel = null;
var dragOffset = {x: 0, y: 0};
var layoutModified = false;  // Track if layout has been modified

// ============================================================================
// Element Drag State
// ============================================================================
var elementDragState = null;  // {element, panelName, startPos, elementType, axId}

// Snap positions for legend alignment
var SNAP_POSITIONS = {
    'upper right': [1.0, 1.0],
    'upper left': [0.0, 1.0],
    'lower left': [0.0, 0.0],
    'lower right': [1.0, 0.0],
    'upper center': [0.5, 1.0],
    'lower center': [0.5, 0.0],
    'center left': [0.0, 0.5],
    'center right': [1.0, 0.5],
    'center': [0.5, 0.5]
};

// ============================================================================
// Resize State
// ============================================================================
var resizingPanel = null;

// ============================================================================
// Update Scheduling
// ============================================================================
var updateTimer = null;
var DEBOUNCE_DELAY = 500;

// ============================================================================
// Shortcut Mode State
// ============================================================================
var shortcutMode = null;  // For multi-key shortcuts like Alt+A → L

// ============================================================================
// Canvas Zoom and View State
// ============================================================================
var canvasZoom = 1.0;

// ============================================================================
// Grid Visibility
// ============================================================================
var gridVisible = true;

// ============================================================================
// Undo/Redo State
// ============================================================================
var undoStack = [];
var redoStack = [];

// ============================================================================
// Context Menu State
// ============================================================================
var contextMenu = null;

// ============================================================================
// Auto-Update State
// ============================================================================
var autoUpdateIntervalId = null;

console.log('state.js loaded - global state variables initialized');
