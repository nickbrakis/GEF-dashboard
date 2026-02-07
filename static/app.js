// API Configuration
const API_BASE_URL = `${window.location.origin}/api`;

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
    },
    mae: {
        primary: 'rgba(100, 255, 150, 0.8)',
        secondary: 'rgba(100, 255, 150, 0.2)',
        border: 'rgba(100, 255, 150, 1)'
    }
};

// DOM Elements - Single Mode
const fetchButton = document.getElementById('fetchButton');
const experimentNameInput = document.getElementById('experimentName');
const parentFilterInput = document.getElementById('parentFilter');
const loadSingleExperimentsBtn = document.getElementById('loadSingleExperimentsBtn');
const singleExperimentSelect = document.getElementById('singleExperimentSelect');
const downloadCsvButton = document.getElementById('downloadCsvButton');

// DOM Elements - Compare Mode
const compareButton = document.getElementById('compareButton');
const experimentSelect = document.getElementById('experimentSelect');
const loadExperimentsBtn = document.getElementById('loadExperimentsBtn');
const compareParentFilterInput = document.getElementById('compareParentFilter');




// DOM Elements - Plot Compare Mode
const globalExperimentNameInput = document.getElementById('globalExperimentName');
const globalParentFilterInput = document.getElementById('globalParentFilter'); // New
const groupedExperimentNameInput = document.getElementById('groupedExperimentName');
const generatePlotBtn = document.getElementById('generatePlotBtn');
const plotPlaceholder = document.getElementById('plotPlaceholder');
const comparisonPlotImage = document.getElementById('comparisonPlotImage');

// DOM Elements - Mode Toggle
const singleModeBtn = document.getElementById('singleModeBtn');
const compareModeBtn = document.getElementById('compareModeBtn');
const plotCompareModeBtn = document.getElementById('plotCompareModeBtn');
const leaderboardModeBtn = document.getElementById('leaderboardModeBtn'); // New
const chatModeBtn = document.getElementById('chatModeBtn');
const singleMode = document.getElementById('singleMode');
const compareMode = document.getElementById('compareMode');
const plotCompareMode = document.getElementById('plotCompareMode');
const leaderboardMode = document.getElementById('leaderboardMode'); // New
const chatMode = document.getElementById('chatMode');
const leaderboardBody = document.getElementById('leaderboardBody'); // New

// DOM Elements - Shared
const loadingState = document.getElementById('loadingState');
const errorState = document.getElementById('errorState');
const errorMessage = document.getElementById('errorMessage');
const statsSection = document.getElementById('statsSection');
const chartsSection = document.getElementById('chartsSection');
const runsSection = document.getElementById('runsSection');
const statsGrid = document.getElementById('statsGrid');
const chartsGrid = document.getElementById('chartsGrid');
const runsTable = document.getElementById('runsTable');
const csvPreviewSection = document.getElementById('csvPreviewSection');
const csvPreviewTable = document.getElementById('csvPreviewTable');
const closePreviewBtn = document.getElementById('closePreviewBtn');

// DOM Elements - Chat Mode
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const chatSendBtn = document.getElementById('chatSendBtn');
const chatModelSelect = document.getElementById('chatModelSelect');
const chatExampleButtons = document.querySelectorAll('.chat-example');
const CHAT_CONVERSATION_STORAGE_KEY = 'mlflow_chat_conversation_id';
const CHAT_MODEL_STORAGE_KEY = 'mlflow_chat_model';
let chatConversationId = localStorage.getItem(CHAT_CONVERSATION_STORAGE_KEY) || '';

if (chatModelSelect) {
    const savedModel = localStorage.getItem(CHAT_MODEL_STORAGE_KEY);
    if (savedModel) {
        chatModelSelect.value = savedModel;
    }
    chatModelSelect.addEventListener('change', () => {
        localStorage.setItem(CHAT_MODEL_STORAGE_KEY, chatModelSelect.value);
    });
}

function setChatConversationId(conversationId) {
    chatConversationId = (conversationId || '').trim();
    if (chatConversationId) {
        localStorage.setItem(CHAT_CONVERSATION_STORAGE_KEY, chatConversationId);
    } else {
        localStorage.removeItem(CHAT_CONVERSATION_STORAGE_KEY);
    }
}

async function resetChatConversation() {
    if (!chatConversationId) return;
    try {
        await fetch(`${API_BASE_URL}/chat/reset`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ conversation_id: chatConversationId })
        });
    } catch (_) {
        // Best effort reset; keep UX responsive even if reset fails.
    } finally {
        setChatConversationId('');
    }
}

// Event Listeners - Single Mode
fetchButton.addEventListener('click', fetchMetrics);
loadSingleExperimentsBtn.addEventListener('click', loadSingleExperiments);
downloadCsvButton.addEventListener('click', downloadConcatenatedCSV);

// Event Listeners - Compare Mode
compareButton.addEventListener('click', compareExperiments);
loadExperimentsBtn.addEventListener('click', loadAvailableExperiments);

// Event Listeners - Mode Toggle


// Event Listeners - Mode Toggle
singleModeBtn.addEventListener('click', () => switchMode('single'));
compareModeBtn.addEventListener('click', () => switchMode('compare'));
plotCompareModeBtn.addEventListener('click', () => switchMode('plotCompare'));
closePreviewBtn.addEventListener('click', () => csvPreviewSection.classList.add('hidden'));

// Event Listeners - Plot Compare Mode
if (generatePlotBtn) {
    generatePlotBtn.addEventListener('click', generateComparisonPlot);
}

// Event Listeners - Leaderboard Mode
leaderboardModeBtn.addEventListener('click', () => switchMode('leaderboard'));
chatModeBtn.addEventListener('click', () => switchMode('chat'));

// Allow Enter key to trigger fetch
experimentNameInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') fetchMetrics();
});

if (chatInput) {
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatMessage();
    });
}

if (chatSendBtn) {
    chatSendBtn.addEventListener('click', sendChatMessage);
}

if (chatExampleButtons && chatExampleButtons.length > 0) {
    chatExampleButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            const prompt = btn.dataset.prompt || btn.textContent.trim();
            if (chatInput) chatInput.value = prompt;
            sendChatMessage();
        });
    });
}

// Sync dropdown selection with text input
singleExperimentSelect.addEventListener('change', () => {
    const selectedOption = singleExperimentSelect.selectedOptions[0];
    if (selectedOption && !selectedOption.disabled) {
        experimentNameInput.value = selectedOption.value;
    }
});

// Mode switching function
// Mode switching function
function switchMode(mode) {
    // Reset all buttons
    singleModeBtn.classList.remove('active');
    compareModeBtn.classList.remove('active');
    plotCompareModeBtn.classList.remove('active');
    if (leaderboardModeBtn) leaderboardModeBtn.classList.remove('active');
    if (chatModeBtn) chatModeBtn.classList.remove('active');

    // Hide all sections
    singleMode.classList.add('hidden');
    compareMode.classList.add('hidden');
    plotCompareMode.classList.add('hidden');
    if (leaderboardMode) leaderboardMode.classList.add('hidden');
    if (chatMode) chatMode.classList.add('hidden');

    // Activate selected mode
    if (mode === 'single') {
        singleModeBtn.classList.add('active');
        singleMode.classList.remove('hidden');
    } else if (mode === 'compare') {
        compareModeBtn.classList.add('active');
        compareMode.classList.remove('hidden');
    } else if (mode === 'plotCompare') {
        plotCompareModeBtn.classList.add('active');
        plotCompareMode.classList.remove('hidden');
    } else if (mode === 'leaderboard') {
        if (leaderboardModeBtn) leaderboardModeBtn.classList.add('active');
        if (leaderboardMode) leaderboardMode.classList.remove('hidden');
        fetchLeaderboard();
    } else if (mode === 'chat') {
        if (chatModeBtn) chatModeBtn.classList.add('active');
        if (chatMode) chatMode.classList.remove('hidden');
    }

    // Hide results when switching modes (common cleanup)
    hideAllSections();
}

function appendChatBubble(role, contentNode) {
    if (!chatMessages) return;
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;
    bubble.appendChild(contentNode);
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendChatText(role, text) {
    const textNode = document.createElement('p');
    textNode.textContent = text;
    appendChatBubble(role, textNode);
}

function appendChatTable(title, headers, rows) {
    const wrapper = document.createElement('div');
    if (title) {
        const heading = document.createElement('div');
        heading.className = 'chat-title';
        heading.textContent = title;
        wrapper.appendChild(heading);
    }

    const tableWrap = document.createElement('div');
    tableWrap.className = 'runs-table';
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    headers.forEach((h) => {
        const th = document.createElement('th');
        th.textContent = h;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    rows.forEach((row) => {
        const tr = document.createElement('tr');
        row.forEach((cell) => {
            const td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    tableWrap.appendChild(table);
    wrapper.appendChild(tableWrap);
    appendChatBubble('assistant', wrapper);
}

function renderChatResponse(payload) {
    if (!payload || !payload.success) {
        appendChatText('assistant', payload?.error || 'Sorry, I could not process that.');
        return;
    }

    if (payload.type === 'experiments') {
        const rows = payload.items.map((exp) => [exp.name, exp.experiment_id, exp.lifecycle_stage]);
        appendChatTable('Experiments', ['Name', 'ID', 'Stage'], rows);
        return;
    }

    if (payload.type === 'runs') {
        const rows = payload.items.map((run) => [
            run.run_name || '(unnamed)',
            run.run_id,
            run.status || ''
        ]);
        appendChatTable(`Runs for ${payload.experiment_name}`, ['Run Name', 'Run ID', 'Status'], rows);
        return;
    }

    if (payload.type === 'run') {
        const info = payload.item;
        const lines = [
            `Run: ${info.run_name || '(unnamed)'}`,
            `ID: ${info.run_id}`,
            `Status: ${info.status || ''}`
        ];
        appendChatText('assistant', lines.join(' | '));
        return;
    }

    if (payload.type === 'help') {
        appendChatText('assistant', payload.message);
        return;
    }

    if (payload.type === 'llm') {
        appendChatText('assistant', payload.message || 'No response.');
        return;
    }

    if (payload.message) {
        appendChatText('assistant', payload.message);
        return;
    }

    appendChatText('assistant', JSON.stringify(payload, null, 2));
}

async function sendChatMessage() {
    if (!chatInput || !chatSendBtn) return;
    const message = chatInput.value.trim();
    if (!message) return;

    if (message === '/reset') {
        appendChatText('user', message);
        chatInput.value = '';
        await resetChatConversation();
        appendChatText('assistant', 'Conversation memory cleared.');
        return;
    }

    appendChatText('user', message);
    chatInput.value = '';
    chatSendBtn.disabled = true;

    const thinking = document.createElement('p');
    thinking.textContent = 'Thinking...';
    const thinkingBubble = document.createElement('div');
    thinkingBubble.className = 'chat-bubble assistant chat-thinking';
    thinkingBubble.appendChild(thinking);
    chatMessages.appendChild(thinkingBubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message,
                conversation_id: chatConversationId || undefined,
                model: chatModelSelect ? chatModelSelect.value : undefined
            })
        });
        const data = await response.json();
        if (data?.conversation_id) {
            setChatConversationId(data.conversation_id);
        }
        thinkingBubble.remove();
        renderChatResponse(data);
    } catch (error) {
        thinkingBubble.remove();
        appendChatText('assistant', error.message || 'Chat request failed.');
    } finally {
        chatSendBtn.disabled = false;
        chatInput.focus();
    }
}

// Main function to fetch metrics (single experiment mode)
async function fetchMetrics() {
    // Get experiment name from dropdown if selected, otherwise from text input
    let experimentName = '';
    const selectedOption = singleExperimentSelect.selectedOptions[0];

    if (selectedOption && !selectedOption.disabled) {
        experimentName = selectedOption.value;
    } else {
        experimentName = experimentNameInput.value.trim();
    }

    if (!experimentName) {
        showError('Please select an experiment from the dropdown or enter an experiment name');
        return;
    }

    // Get selected metrics
    const selectedMetrics = [];
    if (document.getElementById('metricMase').checked) selectedMetrics.push('mase');
    if (document.getElementById('metricRmse').checked) selectedMetrics.push('rmse');
    if (document.getElementById('metricMape').checked) selectedMetrics.push('mape');
    if (document.getElementById('metricMae').checked) selectedMetrics.push('mae');

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

// Hide loading state
function hideLoading() {
    loadingState.classList.add('hidden');
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
    downloadCsvButton.classList.add('hidden');
    downloadCsvButton.classList.add('hidden');
    csvPreviewSection.classList.add('hidden');
    plotSection.classList.add('hidden');
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
    downloadCsvButton.classList.remove('hidden');
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

// Load available experiments
async function loadAvailableExperiments() {
    try {
        loadExperimentsBtn.disabled = true;
        loadExperimentsBtn.textContent = 'Loading...';

        const response = await fetch(`${API_BASE_URL}/experiments`);
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Failed to load experiments');
        }

        // Populate dropdown
        experimentSelect.innerHTML = '';
        data.experiments.forEach(exp => {
            const option = document.createElement('option');
            option.value = exp.name;
            option.textContent = exp.name;
            experimentSelect.appendChild(option);
        });

        loadExperimentsBtn.textContent = 'Reload Experiments';
        loadExperimentsBtn.disabled = false;

    } catch (error) {
        showError(error.message);
        loadExperimentsBtn.textContent = 'Load Available Experiments';
        loadExperimentsBtn.disabled = false;
    }
}

// Load available experiments for single mode
async function loadSingleExperiments() {
    try {
        loadSingleExperimentsBtn.disabled = true;
        loadSingleExperimentsBtn.textContent = 'Loading...';

        const response = await fetch(`${API_BASE_URL}/experiments`);
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Failed to load experiments');
        }

        // Populate dropdown
        singleExperimentSelect.innerHTML = '';
        data.experiments.forEach(exp => {
            const option = document.createElement('option');
            option.value = exp.name;
            option.textContent = exp.name;
            singleExperimentSelect.appendChild(option);
        });

        loadSingleExperimentsBtn.textContent = 'Reload Experiments';
        loadSingleExperimentsBtn.disabled = false;

    } catch (error) {
        showError(error.message);
        loadSingleExperimentsBtn.textContent = 'Load Available Experiments';
        loadSingleExperimentsBtn.disabled = false;
    }
}

// Compare experiments function
async function compareExperiments() {
    // Get selected experiments from dropdown
    const selectedOptions = Array.from(experimentSelect.selectedOptions);
    const experimentNames = selectedOptions.map(opt => opt.value);

    if (experimentNames.length < 2) {
        showError('Please select at least 2 experiments from the dropdown');
        return;
    }

    // Get selected metric
    const selectedMetric = document.querySelector('input[name="compareMetric"]:checked').value;
    const parentFilter = compareParentFilterInput
        ? (compareParentFilterInput.value.trim() || null)
        : null;

    // Show loading state
    showLoading();

    try {
        const response = await fetch(`${API_BASE_URL}/compare`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                experiment_names: experimentNames,
                metric: selectedMetric,
                parent_filter: parentFilter
            })
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Failed to compare experiments');
        }

        // Display comparison results
        displayComparisonResults(data);

    } catch (error) {
        showError(error.message);
    }
}

// Display comparison results
function displayComparisonResults(data) {
    hideAllSections();

    const experiments = data.experiments;
    const metricName = data.metric_name;

    // Check if we have any valid data
    const hasData = experiments.some(exp => exp.average !== null);

    if (!hasData) {
        showError('No valid metric data found for these experiments');
        return;
    }

    // Display comparison statistics
    displayComparisonStatistics(experiments, metricName);

    // Display comparison chart
    displayComparisonChart(experiments, metricName);

    // Show sections
    statsSection.classList.remove('hidden');
    chartsSection.classList.remove('hidden');
}

// Display comparison statistics
function displayComparisonStatistics(experiments, metricName) {
    statsGrid.innerHTML = '';

    experiments.forEach(exp => {
        if (exp.error || exp.average === null) return;

        const statCard = document.createElement('div');
        statCard.className = 'stat-card';
        statCard.innerHTML = `
            <div class="stat-label">${exp.experiment_name}</div>
            <div class="stat-value">${exp.average.toFixed(4)}</div>
            <div class="stat-metric">${metricName.toUpperCase()} Average</div>
            <div style="margin-top: 1rem; font-size: 0.875rem; color: var(--color-text-secondary);">
                <div>Min: ${exp.min.toFixed(4)}</div>
                <div>Max: ${exp.max.toFixed(4)}</div>
                <div>Runs: ${exp.valid_runs}</div>
            </div>
        `;
        statsGrid.appendChild(statCard);
    });
}

// Display comparison chart
function displayComparisonChart(experiments, metricName) {
    chartsGrid.innerHTML = '';

    // Destroy existing charts
    Object.values(charts).forEach(chart => chart.destroy());
    Object.keys(charts).forEach(key => delete charts[key]);

    // Filter out experiments with no data
    const validExperiments = experiments.filter(exp => exp.average !== null);

    if (validExperiments.length === 0) return;

    // Create chart container
    const chartContainer = document.createElement('div');
    chartContainer.className = 'chart-container';
    chartContainer.style.gridColumn = '1 / -1'; // Full width

    const chartTitle = document.createElement('h3');
    chartTitle.className = 'chart-title';
    chartTitle.textContent = `${metricName.toUpperCase()} Comparison`;
    chartTitle.style.marginBottom = '0';

    // Add download button
    const downloadBtn = document.createElement('button');
    downloadBtn.className = 'btn btn-download';
    downloadBtn.innerHTML = '<span class="btn-text">Download PNG</span>';

    // Create title container with button
    const titleContainer = document.createElement('div');
    titleContainer.style.display = 'flex';
    titleContainer.style.justifyContent = 'space-between';
    titleContainer.style.alignItems = 'center';
    titleContainer.style.marginBottom = 'var(--spacing-md)';
    titleContainer.appendChild(chartTitle);
    titleContainer.appendChild(downloadBtn);

    const canvas = document.createElement('canvas');
    canvas.id = 'comparison-chart';

    chartContainer.appendChild(titleContainer);
    chartContainer.appendChild(canvas);
    chartsGrid.appendChild(chartContainer);

    // Set download button onclick after canvas is created
    downloadBtn.onclick = () => downloadChartAsPNG(canvas, `${metricName}_comparison`);

    // Create chart
    const ctx = canvas.getContext('2d');
    const colors = METRIC_COLORS[metricName] || METRIC_COLORS.mase;

    charts['comparison'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: validExperiments.map(exp => exp.experiment_name),
            datasets: [{
                label: `${metricName.toUpperCase()} Average`,
                data: validExperiments.map(exp => exp.average),
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
                // Add white background for PNG export
                backgroundColor: 'transparent',
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
                            const exp = validExperiments[context.dataIndex];
                            return [
                                `Average: ${exp.average.toFixed(4)}`,
                                `Min: ${exp.min.toFixed(4)}`,
                                `Max: ${exp.max.toFixed(4)}`,
                                `Runs: ${exp.valid_runs}`
                            ];
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
}

// Download chart as PNG
function downloadChartAsPNG(canvas, filename) {
    // Create a link element
    const link = document.createElement('a');
    link.download = `${filename}_${new Date().toISOString().split('T')[0]}.png`;

    // Convert canvas to data URL with white background for academic papers
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Create a temporary canvas with white background
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext('2d');

    // Fill with white background
    tempCtx.fillStyle = '#FFFFFF';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

    // Draw the original canvas on top
    tempCtx.drawImage(canvas, 0, 0);

    // Set the link href to the canvas data URL
    link.href = tempCanvas.toDataURL('image/png');

    // Trigger download
    link.click();
}

// Download concatenated CSV
async function downloadConcatenatedCSV() {
    // Get experiment name from dropdown if selected, otherwise from text input
    let experimentName = '';
    const selectedOption = singleExperimentSelect.selectedOptions[0];

    if (selectedOption && !selectedOption.disabled) {
        experimentName = selectedOption.value;
    } else {
        experimentName = experimentNameInput.value.trim();
    }

    if (!experimentName) {
        showError('Please select an experiment first');
        return;
    }

    const parentFilter = parentFilterInput.value.trim() || null;

    // Disable button and show loading state
    const originalText = downloadCsvButton.querySelector('.btn-text').textContent;
    downloadCsvButton.disabled = true;
    downloadCsvButton.querySelector('.btn-text').textContent = 'Generating CSV...';

    try {
        const response = await fetch(`${API_BASE_URL}/export-csv`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                experiment_name: experimentName,
                parent_filter: parentFilter
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to generate CSV');
        }

        // Get the CSV content
        const blob = await response.blob();

        // Create download link
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${experimentName}_concatenated_${new Date().toISOString().split('T')[0]}.csv`;

        // Trigger download
        document.body.appendChild(link);
        link.click();

        // Display preview
        const reader = new FileReader();
        reader.onload = function (e) {
            const text = e.target.result;
            displayCSVPreview(text);
        };
        reader.readAsText(blob);

        // Cleanup
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);

        // Reset button
        downloadCsvButton.disabled = false;
        downloadCsvButton.querySelector('.btn-text').textContent = originalText;

    } catch (error) {
        showError(error.message);
        downloadCsvButton.disabled = false;
        downloadCsvButton.querySelector('.btn-text').textContent = originalText;
    }
}

// Display CSV Preview
function displayCSVPreview(csvText) {
    const lines = csvText.trim().split('\n');
    if (lines.length === 0) return;

    const headers = lines[0].split(',');
    const rows = lines.slice(1);

    const table = document.createElement('table');
    table.innerHTML = `
        <thead>
            <tr>
                ${headers.map(h => `<th>${h.replace(/"/g, '')}</th>`).join('')}
            </tr>
        </thead>
        <tbody>
            ${rows.map(row => {
        const columns = row.split(',');
        return `
                    <tr>
                        ${columns.map(c => `<td>${c.replace(/"/g, '')}</td>`).join('')}
                    </tr>
                `;
    }).join('')}
        </tbody>
    `;

    csvPreviewTable.innerHTML = '';
    csvPreviewTable.appendChild(table);
    csvPreviewSection.classList.remove('hidden');
    csvPreviewSection.scrollIntoView({ behavior: 'smooth' });
}

// Generate Comparison Plot
async function generateComparisonPlot() {
    const globalName = globalExperimentNameInput.value.trim();
    const groupedName = groupedExperimentNameInput.value.trim();
    const globalParentFilter = globalParentFilterInput.value.trim() || null;

    if (!globalName || !groupedName) {
        showError('Please enter both Global and Grouped experiment names');
        return;
    }

    showLoading();

    try {
        const response = await fetch(`${API_BASE_URL}/plot-comparison`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                global_experiment: globalName,
                grouped_experiment: groupedName,
                global_parent_filter: globalParentFilter
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to generate plot');
        }

        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);

        comparisonPlotImage.src = imageUrl;
        comparisonPlotImage.style.display = 'block';
        plotPlaceholder.style.display = 'none';

        hideAllSections();
        plotSection.classList.remove('hidden');

    } catch (error) {
        showError(error.message);
    }
}

// Initialize
console.log('MLflow Metrics Visualization App loaded');
console.log('Ready to fetch metrics from MLflow experiments');



async function fetchLeaderboard() {
    // We handle errors and generic cleanup here
    if (errorMessage) errorMessage.parentElement.classList.add('hidden');

    // Show inline spinner in the leaderboard table
    if (leaderboardBody) {
        leaderboardBody.innerHTML = `
            <tr>
                <td colspan="7" style="text-align: center; padding: 3rem;">
                    <div class="spinner"></div>
                    <p style="color: var(--color-text-secondary); margin-top: 1rem;">Ranking top models across GEF experiments...</p>
                </td>
            </tr>
        `;
    }

    // Also show spinners in mini leaderboards
    ['mlp', 'nbeats', 'lstm', 'lgbm'].forEach(key => {
        const container = document.getElementById(`${key}Leaderboard`);
        if (container) {
            container.innerHTML = '<div class="spinner" style="width: 30px; height: 30px; margin: 2rem auto;"></div>';
        }
    });

    try {
        const response = await fetch(`${API_BASE_URL}/leaderboard`);
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error);
        }

        // --- Render Global Leaderboard ---
        const globalData = data.global;
        if (!globalData || globalData.length === 0) {
            leaderboardBody.innerHTML = '<tr><td colspan="7" style="text-align: center;">No data found for core GEF experiments.</td></tr>';
        } else {
            leaderboardBody.innerHTML = '';
            globalData.forEach((row, index) => {
                const tr = document.createElement('tr');
                if (index === 0) {
                    tr.style.background = 'rgba(255, 215, 0, 0.1)';
                    tr.style.fontWeight = 'bold';
                }
                tr.innerHTML = `
                    <td>${index === 0 ? '🏆' : index + 1}</td>
                    <td>${row.experiment}</td>
                    <td title="${row.parent_run}">${row.parent_run}</td>
                    <td>${(row.mase !== null ? row.mase.toFixed(4) : 'N/A')}</td>
                    <td>${(row.mae !== null ? row.mae.toFixed(4) : 'N/A')}</td>
                    <td>${(row.mape !== null ? row.mape.toFixed(4) : 'N/A')}</td>
                    <td>${(row.rmse !== null ? row.rmse.toFixed(4) : 'N/A')}</td>
                `;
                leaderboardBody.appendChild(tr);
            });
        }

        // --- Render Architectural Leaderboards ---
        const archMap = data.architectures;
        if (archMap) {
            renderMiniLeaderboard('mlpLeaderboard', archMap['MLP']);
            renderMiniLeaderboard('nbeatsLeaderboard', archMap['NBEATS']);
            renderMiniLeaderboard('lstmLeaderboard', archMap['LSTM']);
            renderMiniLeaderboard('lgbmLeaderboard', archMap['LightGBM']);
        }

    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

function renderMiniLeaderboard(containerId, models) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!models || models.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--color-text-secondary); padding: 1.5rem; font-size: 0.9rem;">No experiments found for this architecture.</p>';
        return;
    }

    const table = document.createElement('table');
    table.className = 'compact-table';
    table.innerHTML = `
        <thead>
            <tr>
                <th>Experiment</th>
                <th>MASE</th>
            </tr>
        </thead>
        <tbody>
            ${models.map(m => `
                <tr>
                    <td title="${m.experiment}">${m.experiment}</td>
                    <td>${m.mase.toFixed(4)}</td>
                </tr>
            `).join('')}
        </tbody>
    `;
    container.innerHTML = '';
    container.appendChild(table);
}
