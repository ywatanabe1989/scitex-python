/**
 * Download Menu
 * Handles file download functionality
 */

// ============================================================================
// Toggle Download Menu
// ============================================================================
function toggleDownloadMenu() {
    const menu = document.getElementById('download-menu');
    if (menu.style.display === 'block') {
        menu.style.display = 'none';
        document.removeEventListener('click', closeDownloadMenuOnClickOutside);
    } else {
        menu.style.display = 'block';
        // Close when clicking outside
        setTimeout(() => {
            document.addEventListener('click', closeDownloadMenuOnClickOutside);
        }, 0);
    }
}

function closeDownloadMenuOnClickOutside(e) {
    const menu = document.getElementById('download-menu');
    const btn = document.getElementById('download-btn');
    if (menu && btn && !menu.contains(e.target) && !btn.contains(e.target)) {
        menu.style.display = 'none';
        document.removeEventListener('click', closeDownloadMenuOnClickOutside);
    }
}

// ============================================================================
// Download Handlers
// ============================================================================
function downloadAs(format) {
    saveToBundle().then(() => {
        // First export to bundle
        const data = collectOverrides();
        data.format = format;

        fetch('/export', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        })
        .then(r => r.json())
        .then(result => {
            // Then trigger download
            const filename = result.path.split('/').pop();
            window.location.href = '/download/' + filename;
            toggleDownloadMenu();
        })
        .catch(err => {
            setStatus('Error exporting: ' + err, true);
        });
    });
}
