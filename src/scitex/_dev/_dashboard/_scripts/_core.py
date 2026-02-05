#!/usr/bin/env python3
# Timestamp: 2026-02-05
# File: scitex/_dev/_dashboard/_scripts/_core.py

"""Core fetch and cache functions for dashboard JavaScript."""


def get_core_js() -> str:
    """Return core JavaScript for data fetching and caching."""
    return """
let cachedData = { packages: {}, hosts: {}, remotes: {}, rtd: {} };

async function fetchVersions() {
    showLoading(true);
    cachedData = { packages: {}, hosts: {}, remotes: {}, rtd: {} };
    renderFilters();
    renderData();

    // Load packages first (fast)
    fetchPackages();
    // Load hosts, remotes, and RTD in parallel (slower)
    fetchHosts();
    fetchRemotes();
    fetchRtd();
}

async function fetchPackages() {
    setSectionLoading('package', true);
    try {
        const response = await fetch('/api/packages');
        cachedData.packages = await response.json();
        renderFilters();
        renderData();
        updateTimestamp();
        setSectionUpdated('package');
    } catch (err) {
        console.error('Failed to fetch packages:', err);
    } finally {
        showLoading(false);
        setSectionLoading('package', false);
    }
}

async function fetchHosts() {
    setSectionLoading('host', true);
    try {
        const response = await fetch('/api/hosts');
        cachedData.hosts = await response.json();
        renderFilters();
        renderData();
        setSectionUpdated('host');
    } catch (err) {
        console.error('Failed to fetch hosts:', err);
        cachedData.hosts = { error: err.message };
    } finally {
        setSectionLoading('host', false);
    }
}

async function fetchRemotes() {
    setSectionLoading('remote', true);
    try {
        const response = await fetch('/api/remotes');
        cachedData.remotes = await response.json();
        renderFilters();
        renderData();
        setSectionUpdated('remote');
    } catch (err) {
        console.error('Failed to fetch remotes:', err);
        cachedData.remotes = { error: err.message };
    } finally {
        setSectionLoading('remote', false);
    }
}

async function fetchRtd() {
    setSectionLoading('rtd', true);
    try {
        const response = await fetch('/api/rtd');
        cachedData.rtd = await response.json();
        renderFilters();
        renderData();
        setSectionUpdated('rtd');
    } catch (err) {
        console.error('Failed to fetch RTD status:', err);
        cachedData.rtd = { error: err.message };
    } finally {
        setSectionLoading('rtd', false);
    }
}

function setSectionLoading(section, loading) {
    const el = document.getElementById(section + 'Filters');
    if (el) {
        if (loading) {
            el.classList.add('loading-section');
        } else {
            el.classList.remove('loading-section');
        }
    }
}

function setSectionUpdated(section) {
    const el = document.getElementById(section + 'Filters');
    if (el) {
        el.classList.add('just-updated');
        setTimeout(() => el.classList.remove('just-updated'), 1000);
    }
}

function updateTimestamp() {
    document.getElementById('lastUpdated').textContent =
        'Last updated: ' + new Date().toLocaleTimeString();
}

function showLoading(show) {
    document.getElementById('loading').classList.toggle('active', show);
    document.getElementById('overlay').classList.toggle('active', show);
}

// Auto-refresh settings
let autoRefreshInterval = null;
let autoRefreshSeconds = 0;

function toggleAutoRefresh(seconds) {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
        autoRefreshSeconds = 0;
        document.getElementById('autoRefreshBtn').textContent = 'Auto: Off';
        return;
    }
    autoRefreshSeconds = seconds;
    document.getElementById('autoRefreshBtn').textContent = `Auto: ${seconds}s`;
    autoRefreshInterval = setInterval(() => {
        fetchPackages();
        fetchHosts();
        fetchRemotes();
    }, seconds * 1000);
}

function cycleAutoRefresh() {
    const options = [0, 30, 60, 120];
    const current = options.indexOf(autoRefreshSeconds);
    const next = options[(current + 1) % options.length];
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
    }
    if (next > 0) {
        toggleAutoRefresh(next);
    } else {
        autoRefreshSeconds = 0;
        document.getElementById('autoRefreshBtn').textContent = 'Auto: Off';
    }
}

async function refreshData() { await fetchVersions(); }
"""


# EOF
