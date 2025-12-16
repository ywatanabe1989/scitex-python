/**
 * Theme Management
 * Dark/light theme toggle
 */

// ============================================================================
// Theme Toggle
// ============================================================================
function toggleTheme() {
    document.body.classList.toggle('dark-theme');
    const isDark = document.body.classList.contains('dark-theme');

    // Save theme preference
    localStorage.setItem('theme', isDark ? 'dark' : 'light');

    // Re-render single panel preview with dark/light mode colors (if visible)
    if (!showingPanelGrid) {
        updatePreview();
    }
}

// ============================================================================
// Initialize Theme
// ============================================================================
function initializeTheme() {
    // Load saved theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', initializeTheme);
