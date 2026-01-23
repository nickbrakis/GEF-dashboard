// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// Chart instances storage
const charts = {};

// Color palette for metrics
const METRIC_COLORS = {
    mase: {
        primary: 'rgba(160, 100, 255, 0.8)',
        secondary: 'rgba(160, 100, 255, 0.2)',
        border: 'rgba(160, 100, 255, 1)'
    },
    rmse: {
        primary: 'rgba(100, 200, 255, 0.8)',
        secondary: 'rgba(100, 200, 255, 0.2)',
        border: 'rgba(100, 200, 255, 1)'
    },
    mape: {
        primary: 'rgba(255, 150, 100, 0.8)',
        secondary: 'rgba(255, 150, 100, 0.2)',
        border: 'rgba(255, 150, 100, 1)'
    }
};

// DOM Elements
const fetchButton = document.getElementById('fetchButton');
const experimentNameInput = document.getElementById('experimentName');
const parentFilterInput = document.getElementById('parentFilter');
const loadingState = document.getElementById('loadingState');
const errorState = document.getElementById('errorState');
const errorMessage = document.getElementById('errorMessage');
const statsSection = document.getElementById('statsSection');
const chartsSection = document.getElementById('chartsSection');
const runsSection = document.getElementById('runsSection');
const statsGrid = document.getElementById('statsGrid');
const chartsGrid = document.getElementById('chartsGrid');
const runsTable = document.getElementById('runsTable');

// Event Listeners
fetchButton.addEventListener('click', fetchMetrics);

// Allow Enter key to trigger fetch
experimentNameInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') fetchMetrics();
});

// Main function to fetch metrics
async function fetchMetrics() {
    const experimentName = experimentNameInput.value.trim();

    if (!experimentName) {
        showError('Please enter an experiment name');
        return;
    }

    // Get selected metrics
    const selectedMetrics = [];
    if (document.getElementById('metricMase').checked) selectedMetrics.push('mase');
    if (document.getElementById('metricRmse').checked) selectedMetrics.push('rmse');
    if (document.getElementById('metricMape').checked) selectedMetrics.push('mape');

    if (selectedMetrics.length === 0) {
        showError('Please select at least one metric');
        return;
    }

    const parentFilter = parentFilterInput.value.trim() || null;

    // Show loading state
    showLoading();

    try {
        const response = await fetch(`${API_BASE_URL}/metrics`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                experiment_name: experimentName,
                metrics: selectedMetrics,
                parent_filter: parentFilter
            })
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Failed to fetch metrics');
        }

        // Display results
        displayResults(data);

    } catch (error) {
        showError(error.message);
    }
}

// Show loading state
function showLoading() {
    hideAllSections();
    loadingState.classList.remove('hidden');
}

// Show error state
function showError(message) {
    hideAllSections();
    errorMessage.textContent = message;
    errorState.classList.remove('hidden');
}

// Hide all sections
function hideAllSections() {
    loadingState.classList.add('hidden');
    errorState.classList.add('hidden');
    statsSection.classList.add('hidden');
    chartsSection.classList.add('hidden');
    runsSection.classList.add('hidden');
}

// Display results
function displayResults(data) {
    hideAllSections();

    const metrics = data.metrics;

    // Check if we have any valid data
    const hasData = Object.values(metrics).some(m => m.valid_runs > 0);

    if (!hasData) {
        showError('No valid metric data found for this experiment');
        return;
    }

    // Display statistics
    displayStatistics(metrics);

    // Display charts
    displayCharts(metrics);

    // Display runs table
    displayRunsTable(metrics);

    // Show sections
    statsSection.classList.remove('hidden');
    chartsSection.classList.remove('hidden');
    runsSection.classList.remove('hidden');
}

// Display statistics cards
function displayStatistics(metrics) {
    statsGrid.innerHTML = '';

    Object.entries(metrics).forEach(([metricName, metricData]) => {
        if (metricData.error || metricData.valid_runs === 0) return;

        const statCard = document.createElement('div');
        statCard.className = 'stat-card';
        statCard.innerHTML = `
            <div class="stat-label">${metricName.toUpperCase()}</div>
            <div class="stat-value">${metricData.average.toFixed(4)}</div>
            <div class="stat-metric">Average</div>
            <div style="margin-top: 1rem; font-size: 0.875rem; color: var(--color-text-secondary);">
                <div>Min: ${metricData.min.toFixed(4)}</div>
                <div>Max: ${metricData.max.toFixed(4)}</div>
                <div>Runs: ${metricData.valid_runs}</div>
            </div>
        `;
        statsGrid.appendChild(statCard);
    });
}

// Display charts
function displayCharts(metrics) {
    chartsGrid.innerHTML = '';

    // Destroy existing charts
    Object.values(charts).forEach(chart => chart.destroy());
    Object.keys(charts).forEach(key => delete charts[key]);

    Object.entries(metrics).forEach(([metricName, metricData]) => {
        if (metricData.error || metricData.valid_runs === 0) return;

        // Create chart container
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';

        const chartTitle = document.createElement('h3');
        chartTitle.className = 'chart-title';
        chartTitle.textContent = metricName.toUpperCase();

        const canvas = document.createElement('canvas');
        canvas.id = `chart-${metricName}`;

        chartContainer.appendChild(chartTitle);
        chartContainer.appendChild(canvas);
        chartsGrid.appendChild(chartContainer);

        // Create chart
        const ctx = canvas.getContext('2d');
        const colors = METRIC_COLORS[metricName] || METRIC_COLORS.mase;

        charts[metricName] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: metricData.values.map((_, i) => `Run ${i + 1}`),
                datasets: [{
                    label: metricName.toUpperCase(),
                    data: metricData.values,
                    backgroundColor: colors.primary,
                    borderColor: colors.border,
                    borderWidth: 2,
                    borderRadius: 8,
                    hoverBackgroundColor: colors.border
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        padding: 12,
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: colors.border,
                        borderWidth: 1,
                        displayColors: false,
                        callbacks: {
                            label: function (context) {
                                return `${metricName.toUpperCase()}: ${context.parsed.y.toFixed(4)}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)',
                            drawBorder: false
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)',
                            font: {
                                size: 11
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false,
                            drawBorder: false
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)',
                            font: {
                                size: 11
                            },
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                },
                animation: {
                    duration: 800,
                    easing: 'easeInOutQuart'
                }
            }
        });
    });
}

// Display runs table
function displayRunsTable(metrics) {
    // Get all runs from the first metric that has data
    const firstMetricWithData = Object.values(metrics).find(m => m.runs && m.runs.length > 0);

    if (!firstMetricWithData) {
        runsTable.innerHTML = '<p style="text-align: center; color: var(--color-text-secondary);">No run data available</p>';
        return;
    }

    const runs = firstMetricWithData.runs;

    // Create table
    const table = document.createElement('table');
    table.innerHTML = `
        <thead>
            <tr>
                <th>Parent Run</th>
                <th>Eval Run</th>
                ${Object.keys(metrics).map(m => `<th>${m.toUpperCase()}</th>`).join('')}
            </tr>
        </thead>
        <tbody>
            ${runs.map((run, index) => `
                <tr>
                    <td>${run.parent_run_name}</td>
                    <td>${run.eval_run_name}</td>
                    ${Object.entries(metrics).map(([metricName, metricData]) => {
        const value = metricData.runs && metricData.runs[index]
            ? metricData.runs[index].metric_value
            : 'N/A';
        return `<td>${typeof value === 'number' ? value.toFixed(4) : value}</td>`;
    }).join('')}
                </tr>
            `).join('')}
        </tbody>
    `;

    runsTable.innerHTML = '';
    runsTable.appendChild(table);
}

// Initialize
console.log('MLflow Metrics Visualization App loaded');
console.log('Ready to fetch metrics from MLflow experiments');
