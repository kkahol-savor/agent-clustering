// Global state
let currentJobId = null;
let currentEvalId = null;
let uploadedFileName = null;

// Algorithm parameter configuration
const algorithmParams = {
    'kmeans': [
        { name: 'n_clusters', label: 'Number of Clusters', type: 'number', min: 2, default: 5 },
        { name: 'init', label: 'Initialization Method', type: 'select', options: ['k-means++', 'random'], default: 'k-means++' },
        { name: 'max_iter', label: 'Maximum Iterations', type: 'number', min: 10, default: 300 }
    ],
    'dbscan': [
        { name: 'eps', label: 'Epsilon (neighborhood size)', type: 'number', min: 0.1, step: 0.1, default: 0.5 },
        { name: 'min_samples', label: 'Minimum Samples', type: 'number', min: 1, default: 5 }
    ],
    'hdbscan': [
        { name: 'min_cluster_size', label: 'Minimum Cluster Size', type: 'number', min: 2, default: 5 },
        { name: 'min_samples', label: 'Minimum Samples', type: 'number', min: 1, default: 5 }
    ],
    'gmm': [
        { name: 'n_components', label: 'Number of Components', type: 'number', min: 1, default: 5 },
        { name: 'covariance_type', label: 'Covariance Type', type: 'select', options: ['full', 'tied', 'diag', 'spherical'], default: 'full' }
    ],
    'agglomerative': [
        { name: 'n_clusters', label: 'Number of Clusters', type: 'number', min: 2, default: 5 },
        { name: 'linkage', label: 'Linkage', type: 'select', options: ['ward', 'complete', 'average', 'single'], default: 'ward' }
    ]
};

// Function to reset the chat interface
function resetChat() {
    // Clear chat messages
    const chatBox = document.getElementById('chat-box');
    if (chatBox) {
        chatBox.innerHTML = '';
    }
    
    // Reset global variables
    currentJobId = null;
    currentEvalId = null;
    
    // Hide any open modals
    const resultsModal = document.getElementById('results-modal');
    if (resultsModal) {
        resultsModal.classList.add('hidden');
    }
    
    // Add welcome message
    addBotMessage('Welcome to the Clustering Analysis System. Upload a data file to begin.');
    
    console.log('Chat interface reset');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM loaded in clustering.js");
    
    // Clear any existing chat messages that might be persisting
    const chatBox = document.getElementById('chat-box');
    if (chatBox) {
        console.log("Clearing chat box");
        chatBox.innerHTML = '';
    }
    
    // Add welcome message
    addBotMessage('Welcome to the Clustering Analysis System. Upload a data file to begin.');
    
    // Hide any modals on startup
    const allModals = document.querySelectorAll('.modal, #results-modal');
    allModals.forEach(modal => {
        console.log("Hiding modal:", modal.id);
        modal.classList.add('hidden');
    });
    
    // Get DOM Elements - with null checks
    const resetButton = document.getElementById('reset-chat');
    const uploadForm = document.getElementById('upload-form');
    const uploadStatus = document.getElementById('upload-status');
    const clusteringControls = document.getElementById('clustering-controls');
    const algorithmSelect = document.getElementById('algorithm-select');
    const parameterControls = document.getElementById('parameter-controls');
    const runClusteringBtn = document.getElementById('run-clustering');
    const resultsModal = document.getElementById('results-modal');
    const resultsContent = document.getElementById('results-content');
    const downloadReportBtn = document.getElementById('download-report');
    const optimizeParamsBtn = document.getElementById('optimize-params');
    const closeModalBtn = document.querySelector('.close-modal');
    
    console.log("Elements found:", {
        resetButton: !!resetButton,
        uploadForm: !!uploadForm,
        algorithmSelect: !!algorithmSelect,
        runClusteringBtn: !!runClusteringBtn,
        resultsModal: !!resultsModal,
        closeModalBtn: !!closeModalBtn
    });
    
    // Setup event listeners - with null checks
    if (resetButton) {
        resetButton.addEventListener('click', resetChat);
    }
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFileUpload);
    }
    
    if (algorithmSelect) {
        algorithmSelect.addEventListener('change', updateParameterControls);
        // Initial parameter controls
        updateParameterControls();
    }
    
    if (runClusteringBtn) {
        runClusteringBtn.addEventListener('click', runClustering);
    }
    
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', downloadReport);
    }
    
    if (optimizeParamsBtn) {
        optimizeParamsBtn.addEventListener('click', optimizeParameters);
    }
    
    // Add event listeners for modal close
    if (closeModalBtn && resultsModal) {
        closeModalBtn.addEventListener('click', () => {
            console.log("Close button clicked");
            resultsModal.classList.add('hidden');
        });
        
        // Close when clicking outside content
        resultsModal.addEventListener('click', (e) => {
            if (e.target === resultsModal) {
                console.log("Clicked outside modal content");
                resultsModal.classList.add('hidden');
            }
        });
    }
    
    // Add ESC key listener to close modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            console.log("ESC key pressed");
            const modal = document.getElementById('results-modal');
            if (modal && !modal.classList.contains('hidden')) {
                modal.classList.add('hidden');
            }
        }
    });
});

// Handle file upload
async function handleFileUpload(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('data-file');
    const uploadStatus = document.getElementById('upload-status');
    const clusteringControls = document.getElementById('clustering-controls');
    
    if (!fileInput || !uploadStatus) return;
    
    const file = fileInput.files[0];
    
    if (!file) {
        uploadStatus.textContent = 'Please select a file to upload';
        return;
    }
    
    uploadStatus.textContent = 'Uploading...';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/uploadfile/', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.filename) {
            uploadedFileName = result.filename;
            uploadStatus.textContent = `File "${result.original_filename}" uploaded successfully`;
            if (clusteringControls) clusteringControls.classList.remove('hidden');
            
            // Add a message to the chat
            addBotMessage(`I've received your data file "${result.original_filename}". You can now configure and run your clustering analysis.`);
        } else {
            uploadStatus.textContent = 'Error uploading file';
        }
    } catch (error) {
        uploadStatus.textContent = `Error: ${error.message}`;
    }
}

// Update parameter controls based on selected algorithm
function updateParameterControls() {
    const algorithmSelect = document.getElementById('algorithm-select');
    const parameterControls = document.getElementById('parameter-controls');
    
    if (!algorithmSelect || !parameterControls) return;
    
    const algorithm = algorithmSelect.value;
    const params = algorithmParams[algorithm];
    
    parameterControls.innerHTML = '';
    
    params.forEach(param => {
        const paramGroup = document.createElement('div');
        paramGroup.className = 'parameter-group';
        
        const label = document.createElement('label');
        label.textContent = param.label;
        paramGroup.appendChild(label);
        
        if (param.type === 'select') {
            const select = document.createElement('select');
            select.id = `param-${param.name}`;
            select.name = param.name;
            
            param.options.forEach(option => {
                const optElement = document.createElement('option');
                optElement.value = option;
                optElement.textContent = option;
                if (option === param.default) {
                    optElement.selected = true;
                }
                select.appendChild(optElement);
            });
            
            paramGroup.appendChild(select);
        } else if (param.type === 'number') {
            const input = document.createElement('input');
            input.type = 'number';
            input.id = `param-${param.name}`;
            input.name = param.name;
            input.min = param.min || 0;
            if (param.step) input.step = param.step;
            input.value = param.default;
            
            paramGroup.appendChild(input);
        }
        
        parameterControls.appendChild(paramGroup);
    });
}

// Run clustering with selected parameters
async function runClustering() {
    const algorithmSelect = document.getElementById('algorithm-select');
    
    if (!algorithmSelect) return;
    
    if (!uploadedFileName) {
        addBotMessage('Please upload a data file first.');
        return;
    }
    
    const algorithm = algorithmSelect.value;
    const params = algorithmParams[algorithm];
    
    // Collect parameter values
    const hyperparameters = {};
    params.forEach(param => {
        const element = document.getElementById(`param-${param.name}`);
        if (element) {
            hyperparameters[param.name] = element.type === 'number' ? 
                Number(element.value) : 
                element.value;
        }
    });
    
    addBotMessage(`Running ${algorithm} clustering on your data...`);
    
    try {
        const response = await fetch('/api/v1/clustering/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data_filename: uploadedFileName,
                algorithm: algorithm,
                hyperparameters: hyperparameters
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to start clustering');
        }
        
        const result = await response.json();
        
        if (result.job_id) {
            currentJobId = result.job_id;
            addBotMessage(`Clustering job started with ID: ${currentJobId}. Waiting for results...`);
            pollJobStatus(currentJobId);
        } else {
            throw new Error('No job ID received');
        }
    } catch (error) {
        console.error('Clustering error:', error);
        addBotMessage(`Error: ${error.message}`);
    }
}

// Poll job status until complete
async function pollJobStatus(jobId) {
    try {
        const response = await fetch(`/api/v1/clustering/status/${jobId}`);
        const result = await response.json();
        
        if (result.status === 'completed') {
            addBotMessage('Clustering complete! Fetching results...');
            fetchClusteringResults(jobId);
        } else if (result.status === 'in_progress') {
            // Continue polling
            setTimeout(() => pollJobStatus(jobId), 2000);
        } else {
            addBotMessage(`Error: Clustering job ${result.status}`);
        }
    } catch (error) {
        addBotMessage(`Error checking job status: ${error.message}`);
    }
}

// Fetch clustering results
async function fetchClusteringResults(jobId) {
    try {
        const response = await fetch(`/api/v1/clustering/results/${jobId}`);
        const result = await response.json();
        
        if (result.results) {
            displayResults(result.results);
            addBotMessage('Clustering results are ready! You can view them in detail or run evaluation.');
        } else {
            addBotMessage('Error retrieving clustering results.');
        }
    } catch (error) {
        addBotMessage(`Error: ${error.message}`);
    }
}

// Display results in modal
function displayResults(results) {
    const resultsModal = document.getElementById('results-modal');
    const resultsContent = document.getElementById('results-content');
    
    if (!resultsModal || !resultsContent) {
        addBotMessage("Can't display results: UI elements missing");
        console.error("Missing DOM elements for displaying results");
        return;
    }
    
    // Format the results
    let content = '<div class="results-summary">';
    
    if (results.labels) {
        const clusterCounts = {};
        results.labels.forEach(label => {
            if (clusterCounts[label] === undefined) {
                clusterCounts[label] = 0;
            }
            clusterCounts[label]++;
        });
        
        content += '<h4>Cluster Distribution</h4>';
        content += '<ul>';
        Object.entries(clusterCounts).forEach(([cluster, count]) => {
            content += `<li>Cluster ${cluster}: ${count} items</li>`;
        });
        content += '</ul>';
    }
    
    if (results.metrics) {
        content += '<h4>Quality Metrics</h4>';
        content += '<ul>';
        Object.entries(results.metrics).forEach(([metric, value]) => {
            content += `<li>${metric.replace('_', ' ')}: ${value.toFixed(4)}</li>`;
        });
        content += '</ul>';
    }
    
    content += '</div>';
    
    // Set content and show modal
    resultsContent.innerHTML = content;
    resultsModal.classList.remove('hidden');
}

// Download PDF report
async function downloadReport() {
    if (!currentJobId) {
        addBotMessage('No clustering results available to generate a report.');
        return;
    }
    
    addBotMessage('Generating PDF report...');
    
    try {
        const response = await fetch('/api/v1/evaluator/evaluate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                job_id: currentJobId,
                optimize: false,
                // Add these fields to match expected schema
                notify_sms: {
                    enabled: false
                },
                optimization_config: {
                    max_iterations: 10,
                    search_algorithm: "optuna"
                }
            })
        });
        
        // Rest of function unchanged
        const result = await response.json();
        
        if (result.eval_id) {
            currentEvalId = result.eval_id;
            addBotMessage(`Report generation started with ID: ${currentEvalId}. Waiting for completion...`);
            
            // Poll for evaluation completion
            pollEvaluationStatus(currentEvalId);
        } else {
            addBotMessage('Error starting report generation.');
        }
    } catch (error) {
        addBotMessage(`Error: ${error.message}`);
    }
}

// Poll evaluation status
async function pollEvaluationStatus(evalId) {
    try {
        const response = await fetch(`/api/v1/evaluator/status/${evalId}`);
        const result = await response.json();
        
        if (result.status === 'completed') {
            addBotMessage('Report generation complete!');
            openReportDownload(evalId);
        } else if (result.status === 'in_progress') {
            // Continue polling
            setTimeout(() => pollEvaluationStatus(evalId), 2000);
        } else {
            addBotMessage(`Error: Report generation ${result.status}`);
        }
    } catch (error) {
        addBotMessage(`Error checking evaluation status: ${error.message}`);
    }
}

// Open report download
function openReportDownload(evalId) {
    const reportUrl = `/api/v1/evaluator/report/${evalId}`;
    addBotMessage(`<a href="${reportUrl}" target="_blank">Download your report</a>`);
}

// Optimize clustering parameters
async function optimizeParameters() {
    const algorithmSelect = document.getElementById('algorithm-select');
    
    if (!algorithmSelect) return;
    
    if (!uploadedFileName) {
        addBotMessage('Please upload a data file first.');
        return;
    }
    
    const algorithm = algorithmSelect.value;
    
    addBotMessage(`Starting hyperparameter optimization for ${algorithm}...`);
    
    try {
        // First run basic clustering to get a job_id
        const initialResponse = await fetch('/api/v1/clustering/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data_filename: uploadedFileName,
                algorithm: algorithm,
                hyperparameters: getDefaultHyperparameters(algorithm)
            })
        });
        
        const initialResult = await initialResponse.json();
        
        if (initialResult.job_id) {
            currentJobId = initialResult.job_id;
            
            // Wait for initial clustering to complete
            await waitForJobCompletion(currentJobId);
            
            // Start optimization
            const optimizeResponse = await fetch('/api/v1/evaluator/evaluate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    job_id: currentJobId,
                    optimize: true,
                    optimization_config: {
                        max_iterations: 10,
                        search_algorithm: 'optuna'
                    }
                })
            });
            
            const optimizeResult = await optimizeResponse.json();
            
            if (optimizeResult.eval_id) {
                currentEvalId = optimizeResult.eval_id;
                addBotMessage(`Optimization started with ID: ${currentEvalId}. This may take a few minutes...`);
                
                // Poll for optimization completion
                pollOptimizationStatus(currentEvalId);
            } else {
                addBotMessage('Error starting optimization.');
            }
        } else {
            addBotMessage('Error starting initial clustering job.');
        }
    } catch (error) {
        addBotMessage(`Error: ${error.message}`);
    }
}

// Poll optimization status
async function pollOptimizationStatus(evalId) {
    try {
        const response = await fetch(`/api/v1/evaluator/status/${evalId}`);
        const result = await response.json();
        
        if (result.status === 'completed') {
            addBotMessage('Optimization complete!');
            
            if (result.best_params) {
                displayOptimizedParameters(result.best_params);
            } else {
                addBotMessage('No optimized parameters returned.');
            }
        } else if (result.status === 'in_progress') {
            // Continue polling
            setTimeout(() => pollOptimizationStatus(evalId), 3000);
        } else {
            addBotMessage(`Error: Optimization ${result.status}`);
        }
    } catch (error) {
        addBotMessage(`Error checking optimization status: ${error.message}`);
    }
}

// Wait for job completion
async function waitForJobCompletion(jobId) {
    return new Promise((resolve, reject) => {
        async function checkStatus() {
            try {
                const response = await fetch(`/api/v1/clustering/status/${jobId}`);
                const result = await response.json();
                
                if (result.status === 'completed') {
                    resolve();
                } else if (result.status === 'in_progress') {
                    // Continue polling
                    setTimeout(checkStatus, 2000);
                } else {
                    reject(new Error(`Clustering job ${result.status}`));
                }
            } catch (error) {
                reject(error);
            }
        }
        
        checkStatus();
    });
}

// Display optimized parameters
function displayOptimizedParameters(bestParams) {
    addBotMessage('Optimized parameters found:');
    
    let paramText = '<ul>';
    Object.entries(bestParams).forEach(([param, value]) => {
        paramText += `<li>${param}: ${value}</li>`;
    });
    paramText += '</ul>';
    
    addBotMessage(paramText);
    addBotMessage('You can run clustering again with these optimized parameters for better results.');
    
    // Update the UI with the optimized parameters
    updateUIWithOptimizedParams(bestParams);
}

// Update UI with optimized parameters
function updateUIWithOptimizedParams(bestParams) {
    Object.entries(bestParams).forEach(([param, value]) => {
        const element = document.getElementById(`param-${param}`);
        if (element) {
            element.value = value;
        }
    });
}

// Get default hyperparameters for an algorithm
function getDefaultHyperparameters(algorithm) {
    const params = algorithmParams[algorithm];
    const defaults = {};
    
    params.forEach(param => {
        defaults[param.name] = param.default;
    });
    
    return defaults;
}

// Add bot message to chat with null check
function addBotMessage(message) {
    const chatBox = document.getElementById('chat-box');
    if (!chatBox) {
        console.error("Chat box element not found");
        return;
    }
    
    const botMessage = document.createElement('div');
    botMessage.innerHTML = `<strong>System:</strong> ${message}`;
    chatBox.appendChild(botMessage);
    chatBox.scrollTop = chatBox.scrollHeight;
}