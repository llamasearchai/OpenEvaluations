/**
 * OpenEvals - Main JavaScript
 * ===========================
 * 
 * Core JavaScript functionality for the OpenEvals web interface.
 * Handles UI interactions, API calls, and real-time updates.
 * 
 * Author: Nik Jois <nikjois@llamasearch.ai>
 */

class OpenEvalsApp {
    constructor() {
        this.apiBase = '/api';
        this.currentRunId = null;
        this.activePolling = new Map();
        this.charts = new Map();
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.initializeComponents();
        this.setupSocketListeners();
        
        // Initialize page-specific functionality
        this.initializePage();
    }
    
    bindEvents() {
        // Global event listeners
        document.addEventListener('DOMContentLoaded', () => {
            this.initializeComponents();
        });
        
        // Form submissions
        const evaluationForm = document.getElementById('evaluation-form');
        if (evaluationForm) {
            evaluationForm.addEventListener('submit', (e) => this.handleEvaluationSubmit(e));
        }
        
        // Button clicks
        document.addEventListener('click', (e) => {
            if (e.target.matches('.btn-start-evaluation')) {
                this.handleStartEvaluation(e);
            }
            
            if (e.target.matches('.btn-view-results')) {
                this.handleViewResults(e);
            }
            
            if (e.target.matches('.btn-download-results')) {
                this.handleDownloadResults(e);
            }
            
            if (e.target.matches('.modal-close, .modal-overlay')) {
                this.closeModal();
            }
            
            if (e.target.matches('.dropdown-toggle')) {
                this.toggleDropdown(e.target.closest('.dropdown'));
            }
        });
        
        // Escape key to close modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
        
        // Auto-refresh toggles
        const refreshToggles = document.querySelectorAll('[data-auto-refresh]');
        refreshToggles.forEach(toggle => {
            toggle.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.startAutoRefresh(e.target.dataset.autoRefresh);
                } else {
                    this.stopAutoRefresh(e.target.dataset.autoRefresh);
                }
            });
        });
    }
    
    initializeComponents() {
        // Initialize tooltips
        this.initializeTooltips();
        
        // Initialize progress bars
        this.updateProgressBars();
        
        // Initialize charts
        this.initializeCharts();
        
        // Initialize dropdowns
        this.initializeDropdowns();
        
        // Load initial data
        this.loadDashboardData();
    }
    
    initializePage() {
        const path = window.location.pathname;
        
        if (path.includes('/run/')) {
            const runId = path.split('/run/')[1];
            this.currentRunId = runId;
            this.initializeRunDetailPage(runId);
        } else if (path === '/dashboard') {
            this.initializeDashboardPage();
        } else if (path === '/new-evaluation') {
            this.initializeNewEvaluationPage();
        } else {
            this.initializeHomePage();
        }
    }
    
    setupSocketListeners() {
        if (typeof socket !== 'undefined') {
            socket.on('evaluation_progress', (data) => {
                this.handleEvaluationProgress(data);
            });
            
            socket.on('evaluation_complete', (data) => {
                this.handleEvaluationComplete(data);
            });
            
            socket.on('evaluation_error', (data) => {
                this.handleEvaluationError(data);
            });
        }
    }
    
    // API Methods
    async apiCall(endpoint, options = {}) {
        const url = `${this.apiBase}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };
        
        try {
            showLoading();
            const response = await fetch(url, config);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || `HTTP ${response.status}`);
            }
            
            return data;
        } catch (error) {
            console.error('API call failed:', error);
            showToast(error.message, 'error');
            throw error;
        } finally {
            hideLoading();
        }
    }
    
    async startEvaluation(config) {
        try {
            const result = await this.apiCall('/start-evaluation', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            showToast('Evaluation started successfully!', 'success');
            
            // Join the run room for real-time updates
            if (typeof socket !== 'undefined' && result.run_id) {
                socket.emit('join_run', { run_id: result.run_id });
            }
            
            return result;
        } catch (error) {
            showToast('Failed to start evaluation', 'error');
            throw error;
        }
    }
    
    async getRunStatus(runId) {
        return await this.apiCall(`/run/${runId}`);
    }
    
    async getAllRuns() {
        return await this.apiCall('/runs');
    }
    
    // Event Handlers
    async handleEvaluationSubmit(event) {
        event.preventDefault();
        
        const form = event.target;
        const formData = new FormData(form);
        const config = Object.fromEntries(formData.entries());
        
        try {
            const result = await this.startEvaluation(config);
            
            // Redirect to run detail page
            window.location.href = `/run/${result.run_id}`;
        } catch (error) {
            // Error already handled in startEvaluation
        }
    }
    
    async handleStartEvaluation(event) {
        const button = event.target;
        const runId = button.dataset.runId;
        
        if (runId) {
            try {
                await this.apiCall(`/runs/${runId}/restart`, { method: 'POST' });
                showToast('Evaluation restarted', 'success');
                this.refreshRunData(runId);
            } catch (error) {
                // Error already handled in apiCall
            }
        }
    }
    
    handleViewResults(event) {
        const button = event.target;
        const runId = button.dataset.runId;
        
        if (runId) {
            window.location.href = `/run/${runId}`;
        }
    }
    
    async handleDownloadResults(event) {
        const button = event.target;
        const runId = button.dataset.runId;
        
        if (runId) {
            try {
                const url = `${this.apiBase}/download-results/${runId}`;
                const link = document.createElement('a');
                link.href = url;
                link.download = `evaluation_results_${runId}.json`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                showToast('Download started', 'success');
            } catch (error) {
                showToast('Download failed', 'error');
            }
        }
    }
    
    handleEvaluationProgress(data) {
        const { run_id, status, progress, current_task, error } = data;
        
        // Update progress bars
        const progressBars = document.querySelectorAll(`[data-run-id="${run_id}"] .progress-bar`);
        progressBars.forEach(bar => {
            bar.style.width = `${progress}%`;
        });
        
        // Update progress text
        const progressTexts = document.querySelectorAll(`[data-run-id="${run_id}"] .progress-text`);
        progressTexts.forEach(text => {
            text.textContent = current_task || `${progress}% complete`;
        });
        
        // Update status badges
        const statusBadges = document.querySelectorAll(`[data-run-id="${run_id}"] .status-badge`);
        statusBadges.forEach(badge => {
            badge.className = `status-badge status-${status}`;
            badge.textContent = status.toUpperCase();
        });
        
        // Handle completion
        if (status === 'completed') {
            showToast('Evaluation completed successfully!', 'success');
            this.refreshRunData(run_id);
        } else if (status === 'failed') {
            showToast(`Evaluation failed: ${error}`, 'error');
        }
        
        // Update notification count
        this.updateNotificationCount();
    }
    
    handleEvaluationComplete(data) {
        const { run_id } = data;
        showToast('Evaluation completed!', 'success');
        this.refreshRunData(run_id);
    }
    
    handleEvaluationError(data) {
        const { run_id, error } = data;
        showToast(`Evaluation failed: ${error}`, 'error');
        this.refreshRunData(run_id);
    }
    
    // UI Helper Methods
    showModal(content, title = 'Modal') {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal">
                <div class="modal-header">
                    <h3 class="modal-title">${title}</h3>
                    <button class="modal-close" type="button">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M18 6L6 18M6 6l12 12"/>
                        </svg>
                    </button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Animate in
        setTimeout(() => modal.classList.add('show'), 10);
        
        return modal;
    }
    
    closeModal() {
        const modals = document.querySelectorAll('.modal-overlay');
        modals.forEach(modal => {
            modal.classList.add('closing');
            setTimeout(() => modal.remove(), 200);
        });
    }
    
    toggleDropdown(dropdown) {
        // Close other dropdowns
        document.querySelectorAll('.dropdown.open').forEach(d => {
            if (d !== dropdown) d.classList.remove('open');
        });
        
        dropdown.classList.toggle('open');
    }
    
    initializeTooltips() {
        const tooltips = document.querySelectorAll('[data-tooltip]');
        tooltips.forEach(element => {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip-content';
            tooltip.textContent = element.dataset.tooltip;
            
            element.classList.add('tooltip');
            element.appendChild(tooltip);
        });
    }
    
    updateProgressBars() {
        const progressBars = document.querySelectorAll('.progress-bar[data-progress]');
        progressBars.forEach(bar => {
            const progress = parseFloat(bar.dataset.progress) || 0;
            bar.style.width = `${progress}%`;
        });
    }
    
    initializeCharts() {
        const chartContainers = document.querySelectorAll('.chart-wrapper[data-chart]');
        chartContainers.forEach(container => {
            const chartType = container.dataset.chart;
            const chartData = JSON.parse(container.dataset.chartData || '{}');
            
            this.createChart(container, chartType, chartData);
        });
    }
    
    createChart(container, type, data) {
        const canvas = document.createElement('canvas');
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        
        let config = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        };
        
        if (type === 'line') {
            config = {
                type: 'line',
                data: data,
                options: {
                    ...config,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            };
        } else if (type === 'doughnut') {
            config = {
                type: 'doughnut',
                data: data,
                options: {
                    ...config,
                    cutout: '60%'
                }
            };
        } else if (type === 'bar') {
            config = {
                type: 'bar',
                data: data,
                options: {
                    ...config,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            };
        }
        
        const chart = new Chart(ctx, config);
        this.charts.set(container.id, chart);
        
        return chart;
    }
    
    initializeDropdowns() {
        // Close dropdowns when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.matches('.dropdown-toggle') && !e.target.closest('.dropdown-menu')) {
                document.querySelectorAll('.dropdown.open').forEach(dropdown => {
                    dropdown.classList.remove('open');
                });
            }
        });
    }
    
    // Page-specific initialization
    initializeHomePage() {
        this.loadRecentRuns();
        this.loadSystemStats();
    }
    
    initializeDashboardPage() {
        this.loadAllRuns();
        this.startAutoRefresh('runs');
    }
    
    initializeNewEvaluationPage() {
        this.loadAvailableSuites();
        this.loadTargetSystems();
        
        // Dynamic form validation
        const form = document.getElementById('evaluation-form');
        if (form) {
            this.setupFormValidation(form);
        }
    }
    
    initializeRunDetailPage(runId) {
        this.loadRunDetails(runId);
        this.startAutoRefresh(`run-${runId}`);
        
        // Join the run room for real-time updates
        if (typeof socket !== 'undefined') {
            socket.emit('join_run', { run_id: runId });
        }
    }
    
    // Data loading methods
    async loadDashboardData() {
        try {
            const runs = await this.getAllRuns();
            this.updateDashboardCards(runs);
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
        }
    }
    
    async loadRecentRuns() {
        try {
            const runs = await this.getAllRuns();
            const recentRuns = runs.slice(0, 5);
            this.updateRecentRunsList(recentRuns);
        } catch (error) {
            console.error('Failed to load recent runs:', error);
        }
    }
    
    async loadAllRuns() {
        try {
            const runs = await this.getAllRuns();
            this.updateRunsTable(runs);
        } catch (error) {
            console.error('Failed to load runs:', error);
        }
    }
    
    async loadRunDetails(runId) {
        try {
            const runData = await this.getRunStatus(runId);
            this.updateRunDetailPage(runData);
        } catch (error) {
            console.error('Failed to load run details:', error);
        }
    }
    
    async loadSystemStats() {
        // Mock system stats - in real implementation would come from API
        const stats = {
            totalRuns: 156,
            successRate: 94.2,
            averageTime: '2.4m',
            activeRuns: 3
        };
        
        this.updateSystemStats(stats);
    }
    
    // Update methods
    updateDashboardCards(runs) {
        const totalRuns = runs.length;
        const completedRuns = runs.filter(r => r.status === 'completed').length;
        const failedRuns = runs.filter(r => r.status === 'failed').length;
        const activeRuns = runs.filter(r => r.status === 'running').length;
        
        const stats = {
            total: totalRuns,
            completed: completedRuns,
            failed: failedRuns,
            active: activeRuns,
            successRate: totalRuns > 0 ? (completedRuns / totalRuns * 100).toFixed(1) : 0
        };
        
        // Update stat cards
        Object.entries(stats).forEach(([key, value]) => {
            const element = document.querySelector(`[data-stat="${key}"]`);
            if (element) {
                element.textContent = value + (key === 'successRate' ? '%' : '');
            }
        });
    }
    
    updateRecentRunsList(runs) {
        const container = document.getElementById('recent-runs-list');
        if (!container) return;
        
        if (runs.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-title">No evaluations yet</div>
                    <div class="empty-description">Start your first evaluation to see results here.</div>
                    <a href="/new-evaluation" class="btn btn-primary">Create Evaluation</a>
                </div>
            `;
            return;
        }
        
        container.innerHTML = runs.map(run => this.createRunCard(run)).join('');
    }
    
    updateRunsTable(runs) {
        const tbody = document.querySelector('#runs-table tbody');
        if (!tbody) return;
        
        if (runs.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center">
                        <div class="empty-state">
                            <div class="empty-title">No evaluations found</div>
                            <div class="empty-description">Start your first evaluation to see results here.</div>
                        </div>
                    </td>
                </tr>
            `;
            return;
        }
        
        tbody.innerHTML = runs.map(run => this.createRunTableRow(run)).join('');
    }
    
    updateRunDetailPage(runData) {
        // Update run header
        const title = document.getElementById('run-title');
        if (title) title.textContent = `Evaluation ${runData.run_id.slice(-8)}`;
        
        // Update status
        const status = document.getElementById('run-status');
        if (status) {
            status.className = `status-badge status-${runData.status}`;
            status.innerHTML = `
                <span class="status-indicator"></span>
                ${runData.status.toUpperCase()}
            `;
        }
        
        // Update progress
        const progress = document.getElementById('run-progress');
        if (progress) {
            const progressBar = progress.querySelector('.progress-bar');
            const progressText = progress.querySelector('.progress-text');
            
            if (progressBar) progressBar.style.width = `${runData.progress || 0}%`;
            if (progressText) progressText.textContent = `${runData.progress || 0}% complete`;
        }
        
        // Update results if available
        if (runData.results) {
            this.updateRunResults(runData.results);
        }
    }
    
    updateSystemStats(stats) {
        Object.entries(stats).forEach(([key, value]) => {
            const element = document.querySelector(`[data-system-stat="${key}"]`);
            if (element) {
                element.textContent = value;
            }
        });
    }
    
    updateNotificationCount() {
        const badge = document.getElementById('notification-count');
        if (badge) {
            const activeRuns = document.querySelectorAll('.status-badge.status-running').length;
            
            if (activeRuns > 0) {
                badge.textContent = activeRuns;
                badge.style.display = 'block';
            } else {
                badge.style.display = 'none';
            }
        }
    }
    
    // Helper methods for creating HTML elements
    createRunCard(run) {
        const statusClass = `status-${run.status}`;
        const progress = run.progress || 0;
        
        return `
            <div class="evaluation-card" data-run-id="${run.run_id}">
                <div class="evaluation-header">
                    <div>
                        <h4 class="evaluation-title">Evaluation ${run.run_id.slice(-8)}</h4>
                        <p class="evaluation-subtitle">${new Date(run.start_time).toLocaleString()}</p>
                    </div>
                    <div class="evaluation-actions">
                        <span class="status-badge ${statusClass}">
                            <span class="status-indicator"></span>
                            ${run.status.toUpperCase()}
                        </span>
                    </div>
                </div>
                <div class="evaluation-body">
                    <div class="progress">
                        <div class="progress-bar" style="width: ${progress}%"></div>
                    </div>
                    <div class="progress-text">
                        <span>${progress}% complete</span>
                        <span>${run.current_task || ''}</span>
                    </div>
                    <div class="evaluation-actions" style="margin-top: 1rem;">
                        <a href="/run/${run.run_id}" class="btn btn-sm btn-primary">View Details</a>
                        ${run.status === 'completed' ? `<button class="btn btn-sm btn-outline btn-download-results" data-run-id="${run.run_id}">Download</button>` : ''}
                    </div>
                </div>
            </div>
        `;
    }
    
    createRunTableRow(run) {
        const statusClass = `status-${run.status}`;
        const startTime = new Date(run.start_time).toLocaleString();
        const duration = run.end_time 
            ? this.calculateDuration(run.start_time, run.end_time)
            : 'Running...';
        
        return `
            <tr data-run-id="${run.run_id}">
                <td>
                    <a href="/run/${run.run_id}" class="text-primary">
                        ${run.run_id.slice(-8)}
                    </a>
                </td>
                <td>
                    <span class="status-badge ${statusClass}">
                        <span class="status-indicator"></span>
                        ${run.status.toUpperCase()}
                    </span>
                </td>
                <td>${startTime}</td>
                <td>${duration}</td>
                <td>
                    <div class="progress" style="width: 100px;">
                        <div class="progress-bar" style="width: ${run.progress || 0}%"></div>
                    </div>
                </td>
                <td>
                    <div class="d-flex gap-sm">
                        <a href="/run/${run.run_id}" class="btn btn-sm btn-outline">View</a>
                        ${run.status === 'completed' ? `<button class="btn btn-sm btn-outline btn-download-results" data-run-id="${run.run_id}">Download</button>` : ''}
                    </div>
                </td>
            </tr>
        `;
    }
    
    calculateDuration(startTime, endTime) {
        const start = new Date(startTime);
        const end = new Date(endTime);
        const duration = Math.abs(end - start);
        
        const minutes = Math.floor(duration / (1000 * 60));
        const seconds = Math.floor((duration % (1000 * 60)) / 1000);
        
        return `${minutes}m ${seconds}s`;
    }
    
    // Auto-refresh functionality
    startAutoRefresh(key, interval = 5000) {
        if (this.activePolling.has(key)) {
            clearInterval(this.activePolling.get(key));
        }
        
        const intervalId = setInterval(() => {
            if (key === 'runs') {
                this.loadAllRuns();
            } else if (key.startsWith('run-')) {
                const runId = key.replace('run-', '');
                this.loadRunDetails(runId);
            }
        }, interval);
        
        this.activePolling.set(key, intervalId);
    }
    
    stopAutoRefresh(key) {
        if (this.activePolling.has(key)) {
            clearInterval(this.activePolling.get(key));
            this.activePolling.delete(key);
        }
    }
    
    async refreshRunData(runId) {
        if (this.currentRunId === runId) {
            await this.loadRunDetails(runId);
        }
        
        // Also refresh dashboard if we're on it
        if (window.location.pathname === '/dashboard' || window.location.pathname === '/') {
            await this.loadDashboardData();
        }
    }
    
    setupFormValidation(form) {
        const inputs = form.querySelectorAll('input, select, textarea');
        
        inputs.forEach(input => {
            input.addEventListener('blur', () => this.validateField(input));
            input.addEventListener('input', () => this.clearFieldError(input));
        });
    }
    
    validateField(field) {
        const value = field.value.trim();
        const isRequired = field.hasAttribute('required');
        
        if (isRequired && !value) {
            this.showFieldError(field, 'This field is required');
            return false;
        }
        
        // Additional validation based on field type
        if (field.type === 'email' && value && !this.isValidEmail(value)) {
            this.showFieldError(field, 'Please enter a valid email address');
            return false;
        }
        
        this.clearFieldError(field);
        return true;
    }
    
    showFieldError(field, message) {
        this.clearFieldError(field);
        
        field.classList.add('error');
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'field-error';
        errorDiv.textContent = message;
        
        field.parentNode.appendChild(errorDiv);
    }
    
    clearFieldError(field) {
        field.classList.remove('error');
        
        const errorDiv = field.parentNode.querySelector('.field-error');
        if (errorDiv) {
            errorDiv.remove();
        }
    }
    
    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
}

// Utility functions
function showLoading(text = 'Processing...') {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        const loadingText = overlay.querySelector('.loading-text');
        if (loadingText) {
            loadingText.textContent = text;
        }
        overlay.style.display = 'flex';
    }
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

function showToast(message, type = 'info', duration = 5000) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-icon">
            ${getToastIcon(type)}
        </div>
        <div class="toast-content">
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M18 6L6 18M6 6l12 12"/>
            </svg>
        </button>
    `;
    
    container.appendChild(toast);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (toast.parentNode) {
            toast.remove();
        }
    }, duration);
}

function getToastIcon(type) {
    const icons = {
        success: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/></svg>',
        error: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/></svg>',
        warning: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4M12 17h.01"/></svg>',
        info: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>'
    };
    
    return icons[type] || icons.info;
}

function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

function updateEvaluationProgress(data) {
    if (window.app) {
        window.app.handleEvaluationProgress(data);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new OpenEvalsApp();
}); 