class ChatManager {
    constructor() {
        this.messages = [];
        this.isGenerating = false;
        this.setupEventListeners();
    }

    setupEventListeners() {
        const input = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');

        if (input && sendButton) {
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            sendButton.addEventListener('click', () => this.sendMessage());
        }
    }

    async sendMessage() {
        if (this.isGenerating) return;

        const input = document.getElementById('messageInput');
        const text = input.value.trim();
        
        if (!text) return;
        
        // Clear input
        input.value = '';
        
        // Add user message
        this.addMessage('user', text);
        
        // Get active model and its configuration
        const activeModelElement = document.getElementById('activeModel');
        const modelId = activeModelElement?.dataset?.modelId;
        const modelConfig = activeModelElement?.dataset?.config ? JSON.parse(activeModelElement.dataset.config) : null;

        if (!modelId) {
            this.showError('No model is currently loaded');
            return;
        }

        this.isGenerating = true;
        this.showGenerating();

        try {
            const response = await fetch(`/api/models/${modelId}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: this.messages,
                    parameters: {
                        temperature: modelConfig?.temperature || 0.7,
                        top_p: modelConfig?.top_p || 0.9,
                        top_k: modelConfig?.top_k || 40,
                        num_predict: modelConfig?.num_predict || 512,
                        repeat_penalty: modelConfig?.repeat_penalty || 1.1,
                        repeat_last_n: modelConfig?.repeat_last_n || 64
                    }
                })
            });

            if (!response.ok) {
                throw new Error('Failed to get response from model');
            }

            const result = await response.json();
            this.removeGenerating();
            
            if (result.choices && result.choices[0]?.message) {
                this.addMessage('assistant', result.choices[0].message.content);
            } else {
                throw new Error('Invalid response format');
            }
        } catch (error) {
            this.removeGenerating();
            this.showError('Error: ' + error.message);
        } finally {
            this.isGenerating = false;
        }
    }

    addMessage(role, content) {
        this.messages.push({ role, content });
        
        const messagesDiv = document.getElementById('chatMessages');
        if (!messagesDiv) return;

        const messageElement = document.createElement('div');
        messageElement.className = `message ${role}`;
        messageElement.textContent = content;
        
        messagesDiv.appendChild(messageElement);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    showGenerating() {
        const messagesDiv = document.getElementById('chatMessages');
        if (!messagesDiv) return;

        const loadingElement = document.createElement('div');
        loadingElement.id = 'generatingMessage';
        loadingElement.className = 'message assistant loading';
        loadingElement.innerHTML = `
            <div class="loading-indicator">
                <span>Generating</span>
                <span class="loading-dots"></span>
            </div>
            <span class="cancel-button" onclick="chatManager.cancelGeneration()">Cancel</span>
        `;
        
        messagesDiv.appendChild(loadingElement);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    removeGenerating() {
        document.getElementById('generatingMessage')?.remove();
    }

    showError(message) {
        const errorElement = document.createElement('div');
        errorElement.className = 'message error';
        errorElement.textContent = message;
        
        const messagesDiv = document.getElementById('chatMessages');
        if (messagesDiv) {
            messagesDiv.appendChild(errorElement);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
    }

    cancelGeneration() {
        // TODO: Implement cancellation when API supports it
        this.isGenerating = false;
        this.removeGenerating();
        this.showError('Generation cancelled');
    }

    clearChat() {
        this.messages = [];
        const messagesDiv = document.getElementById('chatMessages');
        if (messagesDiv) {
            messagesDiv.innerHTML = '';
        }
    }
}

// Initialize chat manager when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chatManager = new ChatManager();
});

// Chat functionality
const chat = {
    async sendMessage(message) {
        const activeModel = document.getElementById('activeModel').textContent;
        if (activeModel === 'No model loaded') {
            alert('Please load a model first');
            return;
        }

        // Get the model ID from the loaded model text
        const modelId = activeModel.replace('Active model: ', '');

        try {
            // Add user message to chat
            this.addMessage('user', message);

            // Send request to API
            const response = await fetch(`/api/models/${modelId}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    messages: [
                        {
                            role: 'user',
                            content: message
                        }
                    ],
                    parameters: {
                        temperature: 0.7,
                        top_p: 0.95,
                        top_k: 40,
                        repeat_penalty: 1.1,
                        num_predict: 1000
                    }
                })
            });

            if (!response.ok) {
                throw new Error('Failed to get response from model');
            }

            const result = await response.json();
            
            // Add assistant's response to chat
            this.addMessage('assistant', result.choices[0].message.content);

        } catch (error) {
            console.error('Error in chat:', error);
            alert('Error: ' + error.message);
        }
    },

    addMessage(role, content) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        messageDiv.textContent = content;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
};

// Set up event listeners
document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');

    // Send message on button click
    sendButton.addEventListener('click', () => {
        const message = messageInput.value.trim();
        if (message) {
            chat.sendMessage(message);
            messageInput.value = '';
        }
    });

    // Send message on Enter key
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const message = messageInput.value.trim();
            if (message) {
                chat.sendMessage(message);
                messageInput.value = '';
            }
        }
    });
});

// Update active model display when model is loaded
document.addEventListener('modelLoaded', (event) => {
    const activeModelElement = document.getElementById('activeModel');
    activeModelElement.textContent = `Active model: ${event.detail.modelId}`;
}); 