// System metrics handling
class MetricsManager {
    constructor() {
        this.updateInterval = 5000; // 5 seconds
        this.maxDataPoints = 20;
        this.cpuData = Array(this.maxDataPoints).fill(0);
        this.memoryData = Array(this.maxDataPoints).fill(0);
        this.charts = {};
        this.initializeCharts();
        this.startMetricsPolling();
    }

    initializeCharts() {
        // Initialize CPU chart
        const cpuCtx = document.getElementById('cpuChart')?.getContext('2d');
        if (cpuCtx) {
            this.charts.cpu = new Chart(cpuCtx, {
                type: 'line',
                data: {
                    labels: Array(this.maxDataPoints).fill(''),
                    datasets: [{
                        label: 'CPU Usage (%)',
                        data: this.cpuData,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: { y: { beginAtZero: true, max: 100 } },
                    plugins: { legend: { display: false } }
                }
            });
        }

        // Initialize Memory chart
        const memoryCtx = document.getElementById('memoryChart')?.getContext('2d');
        if (memoryCtx) {
            this.charts.memory = new Chart(memoryCtx, {
                type: 'line',
                data: {
                    labels: Array(this.maxDataPoints).fill(''),
                    datasets: [{
                        label: 'Memory Usage (MB)',
                        data: this.memoryData,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: { y: { beginAtZero: true } },
                    plugins: { legend: { display: false } }
                }
            });
        }
    }

    updateChart(chartName, newValue) {
        const chart = this.charts[chartName];
        if (!chart) return;

        const data = chartName === 'cpu' ? this.cpuData : this.memoryData;
        data.push(newValue);
        if (data.length > this.maxDataPoints) {
            data.shift();
        }
        
        chart.data.datasets[0].data = data.slice();
        chart.update();
    }

    async updateMetrics() {
        try {
            const response = await fetch('/api/metrics');
            const data = await response.json();
            
            // Update CPU metrics
            const cpuPercent = data.cpu_percent;
            this.updateProgressBar('cpu', cpuPercent);
            this.updateChart('cpu', cpuPercent);
            
            // Update Memory metrics
            const memoryUsed = data.memory_used_mb;
            const memoryTotal = data.memory_total_mb;
            const memoryPercent = (memoryUsed / memoryTotal) * 100;
            this.updateProgressBar('memory', memoryPercent);
            this.updateChart('memory', memoryUsed);
            
            // Update memory text
            document.getElementById('memoryUsed').textContent = `${(memoryUsed / 1024).toFixed(1)} GB`;
            document.getElementById('memoryTotal').textContent = `${(memoryTotal / 1024).toFixed(1)} GB`;
            
            // Update GPU metrics if available
            if (data.gpu_memory_used !== null) {
                const gpuUsed = data.gpu_memory_used;
                const gpuTotal = data.gpu_memory_total;
                const gpuPercent = (gpuUsed / gpuTotal) * 100;
                this.updateProgressBar('gpu', gpuPercent);
                document.getElementById('gpuUsed').textContent = `${(gpuUsed / 1024).toFixed(1)} GB`;
                document.getElementById('gpuTotal').textContent = `${(gpuTotal / 1024).toFixed(1)} GB`;
                document.getElementById('gpuMetrics').classList.remove('hidden');
            } else {
                document.getElementById('gpuMetrics')?.classList.add('hidden');
            }
        } catch (error) {
            console.error('Error updating metrics:', error);
        }
    }

    updateProgressBar(type, percent) {
        const bar = document.getElementById(`${type}Progress`);
        if (bar) {
            bar.style.width = `${percent}%`;
            document.getElementById(`${type}Percent`).textContent = `${percent.toFixed(1)}%`;
        }
    }

    startMetricsPolling() {
        this.updateMetrics();
        setInterval(() => this.updateMetrics(), this.updateInterval);
    }
}

// Model management
class ModelManager {
    constructor() {
        this.updateInterval = 5000;
        this.startModelPolling();
    }

    async loadModel(modelId, params) {
        try {
            const response = await fetch(`/api/models/${modelId}/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            const result = await response.json();
            this.showNotification(result.message, result.status === 'success');
            await this.updateModelList();
        } catch (error) {
            this.showNotification('Error loading model: ' + error.message, false);
        }
    }

    async unloadModel(modelId) {
        try {
            const response = await fetch(`/api/models/${modelId}/unload`, {
                method: 'POST'
            });
            const result = await response.json();
            this.showNotification(result.message, result.status === 'success');
            await this.updateModelList();
        } catch (error) {
            this.showNotification('Error unloading model: ' + error.message, false);
        }
    }

    async updateModelList() {
        try {
            const response = await fetch('/api/models');
            const models = await response.json();
            const container = document.getElementById('modelList');
            
            if (!container) return;
            
            if (Object.keys(models).length === 0) {
                container.innerHTML = '<div class="text-gray-500">No models loaded</div>';
                return;
            }

            container.innerHTML = Object.entries(models)
                .map(([modelId, info]) => this.createModelCard(modelId, info))
                .join('');
        } catch (error) {
            console.error('Error updating model list:', error);
        }
    }

    createModelCard(modelId, info) {
        return `
            <div class="model-card">
                <div class="model-header">
                    <h3 class="model-title">${modelId}</h3>
                    <button onclick="modelManager.unloadModel('${modelId}')" 
                            class="btn btn-danger">Unload</button>
                </div>
                <div class="model-stats">
                    <div>Loaded: ${new Date(info.load_time).toLocaleString()}</div>
                    <div>Last used: ${new Date(info.last_used).toLocaleString()}</div>
                    <div>Context size: ${info.parameters.num_ctx}</div>
                    <div>GPU Layers: ${info.parameters.num_gpu}</div>
                </div>
            </div>
        `;
    }

    showNotification(message, isSuccess = true) {
        const notification = document.createElement('div');
        notification.className = `notification ${isSuccess ? 'success' : 'error'}`;
        notification.textContent = message;
        document.body.appendChild(notification);
        setTimeout(() => notification.remove(), 3000);
    }

    startModelPolling() {
        this.updateModelList();
        setInterval(() => this.updateModelList(), this.updateInterval);
    }
}

// Initialize managers when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.metricsManager = new MetricsManager();
    window.modelManager = new ModelManager();
}); 