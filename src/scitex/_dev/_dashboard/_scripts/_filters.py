#!/usr/bin/env python3
# Timestamp: 2026-02-05
# File: scitex/_dev/_dashboard/_scripts/_filters.py

"""Filter rendering functions for dashboard JavaScript."""


def get_filters_js() -> str:
    """Return JavaScript for filter rendering."""
    return """
function renderFilters() {
    if (!cachedData) return;

    const packageFilters = document.getElementById('packageFilters');
    const packages = Object.keys(cachedData.packages || {});
    packageFilters.innerHTML = packages.map(pkg =>
        `<label><input type="checkbox" value="${pkg}" checked onchange="renderData()"> ${pkg}</label>`
    ).join('');

    const hostFilters = document.getElementById('hostFilters');
    const hosts = Object.keys(cachedData.hosts || {}).filter(h => !h.startsWith('_'));
    if (hosts.length > 0) {
        hostFilters.innerHTML = hosts.map(host =>
            `<label><input type="checkbox" value="${host}" checked onchange="renderData()"> ${host}</label>`
        ).join('');
    } else {
        hostFilters.innerHTML = '<span style="color: var(--text-secondary)">Loading...</span>';
    }

    const remoteFilters = document.getElementById('remoteFilters');
    const remotes = Object.keys(cachedData.remotes || {}).filter(r => !r.startsWith('_'));
    if (remotes.length > 0) {
        remoteFilters.innerHTML = remotes.map(remote =>
            `<label><input type="checkbox" value="${remote}" checked onchange="renderData()"> ${remote}</label>`
        ).join('');
    } else {
        remoteFilters.innerHTML = '<span style="color: var(--text-secondary)">Loading...</span>';
    }

    const rtdFilters = document.getElementById('rtdFilters');
    const rtdVersions = Object.keys(cachedData.rtd || {});
    if (rtdVersions.length > 0) {
        rtdFilters.innerHTML = rtdVersions.map(v =>
            `<label><input type="checkbox" value="${v}" checked onchange="renderData()"> ${v}</label>`
        ).join('');
    } else {
        rtdFilters.innerHTML = '<span style="color: var(--text-secondary)">Loading...</span>';
    }

    document.querySelectorAll('#statusFilters input').forEach(input => {
        input.onchange = renderData;
    });
}

function getSelectedFilters() {
    const getChecked = (containerId) =>
        [...document.querySelectorAll(`#${containerId} input:checked`)].map(el => el.value);
    return {
        packages: getChecked('packageFilters'),
        statuses: getChecked('statusFilters'),
        hosts: getChecked('hostFilters'),
        remotes: getChecked('remoteFilters')
    };
}

function toggleFilters() {
    const filters = document.querySelector('.filters');
    filters.classList.toggle('collapsed');
}
"""


# EOF
