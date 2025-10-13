
document.addEventListener('DOMContentLoaded', function() {
    console.log('Hospital Appointment System loaded');
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // scrolling 
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const requiredFields = form.querySelectorAll('[required]');
            let valid = true;

            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    field.classList.add('is-invalid');
                    valid = false;
                } else {
                    field.classList.remove('is-invalid');
                }
            });

            if (!valid) {
                e.preventDefault();
                showToast('Please fill in all required fields', 'warning');
            }
        });
    });

    // Real-time form validation
    document.querySelectorAll('input, select').forEach(field => {
        field.addEventListener('blur', function() {
            if (this.hasAttribute('required') && !this.value.trim()) {
                this.classList.add('is-invalid');
            } else {
                this.classList.remove('is-invalid');
                this.classList.add('is-valid');
            }
        });

        field.addEventListener('input', function() {
            if (this.value.trim()) {
                this.classList.remove('is-invalid');
                this.classList.add('is-valid');
            }
        });
    });

    // Dashboard auto-refresh (every 5 minutes)
    if (window.location.pathname === '/dashboard') {
        setInterval(() => {
            // Could be enhanced to only refresh charts/data
            console.log('Dashboard data refresh check...');
        }, 300000);
    }

    // Add loading states to buttons
    document.querySelectorAll('button[type="submit"]').forEach(button => {
        button.addEventListener('click', function() {
            const originalText = this.innerHTML;
            this.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            this.disabled = true;

            // Revert after 5 seconds if still on page (form submission should navigate away)
            setTimeout(() => {
                this.innerHTML = originalText;
                this.disabled = false;
            }, 5000);
        });
    });

    // Add animation to dashboard stats
    if (window.location.pathname === '/dashboard') {
        animateDashboardStats();
    }
});

// Animate dashboard statistics
function animateDashboardStats() {
    const stats = document.querySelectorAll('.dashboard-stat h3');
    stats.forEach(stat => {
        const finalValue = parseInt(stat.textContent);
        if (!isNaN(finalValue)) {
            animateValue(stat, 0, finalValue, 2000);
        }
    });
}

function animateValue(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const value = Math.floor(progress * (end - start) + start);
        element.textContent = value.toLocaleString();
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Toast notification system
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container') || createToastContainer();
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas fa-${getToastIcon(type)} me-2"></i>
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    // Remove toast after it's hidden
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

function getToastIcon(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    container.style.zIndex = '9999';
    document.body.appendChild(container);
    return container;
}

// API functions
async function makePrediction(predictionData) {
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(predictionData)
        });
        
        if (!response.ok) {
            throw new Error('Prediction request failed');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Prediction API error:', error);
        showToast('Prediction service unavailable', 'danger');
        return null;
    }
}

// Export functionality
function exportTableToCSV(tableId, filename) {
    const table = document.getElementById(tableId);
    if (!table) {
        showToast('Table not found for export', 'warning');
        return;
    }

    const rows = table.querySelectorAll('tr');
    const csv = [];
    
    for (const row of rows) {
        const rowData = [];
        const cols = row.querySelectorAll('td, th');
        
        for (const col of cols) {
            // Remove any icons or HTML content and get clean text
            const text = col.textContent.replace(/"/g, '""').trim();
            rowData.push(`"${text}"`);
        }
        
        csv.push(rowData.join(','));
    }
    
    const csvString = csv.join('\n');
    const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    
    a.href = url;
    a.download = filename || 'appointments_export.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showToast('Data exported successfully', 'success');
}

// Chart helper functions (if using client-side charts in future)
function createSimpleChart(canvasId, data, options = {}) {
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) {
        console.error('Canvas element not found:', canvasId);
        return null;
    }
    
    return new Chart(ctx, {
        type: options.type || 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            ...options
        }
    });
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function formatDate(date) {
    return new Date(date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

// Health check
async function checkSystemHealth() {
    try {
        const response = await fetch('/health');
        const health = await response.json();
        
        if (health.status !== 'healthy') {
            showToast('System is experiencing issues', 'warning');
        }
        
        return health;
    } catch (error) {
        console.error('Health check failed:', error);
        showToast('System health check failed', 'danger');
        return { status: 'unreachable' };
    }
}

// Model retraining function
async function retrainModel() {
    if (!confirm('Are you sure you want to retrain the model? This may take a few moments.')) {
        return;
    }

    showToast('Retraining model...', 'info');
    
    try {
        const response = await fetch('/retrain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showToast('Model retrained successfully!', 'success');
            // Reload page after a short delay to see updated metrics
            setTimeout(() => location.reload(), 2000);
        } else {
            showToast('Error retraining model: ' + data.message, 'danger');
        }
    } catch (error) {
        showToast('Error retraining model: ' + error, 'danger');
    }
}

// Initialize health check on page load
checkSystemHealth();

// Make functions available globally
window.exportTableToCSV = exportTableToCSV;
window.retrainModel = retrainModel;
window.showToast = showToast;