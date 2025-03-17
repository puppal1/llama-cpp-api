// Model management functions
const modelManager = {
    // CPU-optimized model configurations
    modelConfigs: {
        'mistral-7b-instruct-v0': {
            num_ctx: 2048,
            num_batch: 512,
            num_thread: 8,  // Increased threads for CPU
            num_gpu: 0,     // No GPU layers
            mlock: false,
            mmap: true,
            // Model specific parameters
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            num_predict: 256
        },
        'Phi-3-mini-4k-instruct-q4': {
            num_ctx: 4096,  // Phi-3 supports larger context
            num_batch: 256,
            num_thread: 8,  // Increased threads for CPU
            num_gpu: 0,     // No GPU layers
            mlock: false,
            mmap: true,
            // Model specific parameters
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            num_predict: 256
        }
    },

    async loadModel(modelId) {
        try {
            const modelPath = document.getElementById('modelPath').value;
            const config = this.modelConfigs[modelId] || {};
            const settings = { ...this.currentSettings, ...config };

            const response = await fetch('/api/load_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_path: modelPath,
                    model_id: modelId,
                    parameters: settings
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            if (result.success) {
                this.activeModel = modelId;
                document.getElementById('activeModel').textContent = `Active Model: ${modelId}`;
                document.dispatchEvent(new CustomEvent('modelLoaded', { detail: { modelId } }));
                this.updateModelList();
                this.loadCurrentSettings();
            } else {
                throw new Error(result.error || 'Failed to load model');
            }
        } catch (error) {
            console.error('Error loading model:', error);
            alert(`Error loading model: ${error.message}`);
        }
    },

    async updateModelList() {
        try {
            const response = await fetch('/api/models');
            const models = await response.json();
            const modelList = document.getElementById('modelList');
            modelList.innerHTML = '';
            
            for (const [modelId, info] of Object.entries(models)) {
                const config = this.modelConfigs[modelId] || {};
                const modelDiv = document.createElement('div');
                modelDiv.className = 'model-item';
                modelDiv.innerHTML = `
                    <h5>${modelId}</h5>
                    <div class="model-info">
                        <p>Status: ${info.status}</p>
                        <p>Context: ${config.num_ctx || info.parameters.num_ctx}</p>
                        <p>Threads: ${config.num_thread || info.parameters.num_thread}</p>
                        <p>Loaded: ${new Date(info.load_time).toLocaleString()}</p>
                    </div>
                    <div class="model-actions">
                        <button class="btn btn-danger btn-sm" onclick="modelManager.unloadModel('${modelId}')">Unload</button>
                    </div>
                `;
                modelList.appendChild(modelDiv);
            }
        } catch (error) {
            console.error('Error updating model list:', error);
        }
    },

    async unloadModel(modelId) {
        try {
            const response = await fetch(`/api/models/${modelId}/unload`, {
                method: 'POST'
            });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.detail || 'Failed to unload model');
            }
            
            // Reset active model display
            const activeModelElement = document.getElementById('activeModel');
            activeModelElement.textContent = 'No model loaded';
            activeModelElement.dataset.modelId = '';
            activeModelElement.dataset.config = '';
            
            alert('Model unloaded successfully!');
            this.updateModelList();
        } catch (error) {
            alert('Error unloading model: ' + error.message);
        }
    },

    async loadCurrentSettings() {
        if (!this.activeModel) return;

        const response = await fetch('/api/get_model_settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_id: this.activeModel
            })
        });

        if (response.ok) {
            const settings = await response.json();
            this.currentSettings = settings;
            this.updateSettingsUI();
        }
    },

    updateSettingsUI() {
        const forms = ['modelSettingsForm', 'advancedSettingsForm'];
        forms.forEach(formId => {
            const form = document.getElementById(formId);
            if (!form) return;

            for (const input of form.elements) {
                if (!input.name) continue;

                if (input.type === 'checkbox') {
                    input.checked = this.currentSettings[input.name] || false;
                } else {
                    input.value = this.currentSettings[input.name] !== undefined 
                        ? this.currentSettings[input.name] 
                        : input.value;
                }
            }
        });
    },

    async saveSettings() {
        if (!this.activeModel) {
            alert('Please load a model first');
            return;
        }

        const forms = ['modelSettingsForm', 'advancedSettingsForm'];
        const newSettings = {};

        forms.forEach(formId => {
            const form = document.getElementById(formId);
            if (!form) return;

            for (const input of form.elements) {
                if (!input.name) continue;

                if (input.type === 'checkbox') {
                    newSettings[input.name] = input.checked;
                } else if (input.type === 'number') {
                    newSettings[input.name] = parseFloat(input.value);
                } else if (input.type === 'range') {
                    newSettings[input.name] = parseFloat(input.value);
                } else {
                    newSettings[input.name] = input.value;
                }
            }
        });

        try {
            const response = await fetch('/api/update_model_settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_id: this.activeModel,
                    parameters: newSettings
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            if (result.success) {
                this.currentSettings = newSettings;
                alert('Settings saved successfully');
            } else {
                throw new Error(result.error || 'Failed to save settings');
            }
        } catch (error) {
            console.error('Error saving settings:', error);
            alert(`Error saving settings: ${error.message}`);
        }
    }
}; 