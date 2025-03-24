#!/bin/bash

# Set server port (default: 8000)
SERVER_PORT=${1:-8000}

# Set base directory to the script's directory
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Base directory: $BASE_DIR"

# Set models directory
export MODELS_DIR="$BASE_DIR/models"
echo "Models directory: $MODELS_DIR"

# Check if models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo "Creating models directory..."
    mkdir -p "$MODELS_DIR"
fi

# List model files
echo "Listing model files in $MODELS_DIR:"
find "$MODELS_DIR" -name "*.gguf" -type f | while read -r model; do
    echo "  - $(basename "$model")"
done

# Check if any models were found
if [ -z "$(find "$MODELS_DIR" -name "*.gguf" -type f 2>/dev/null)" ]; then
    echo "No GGUF model files found"
fi

# Kill any existing server on the same port
echo "Checking for server instances on port $SERVER_PORT..."
if lsof -i :$SERVER_PORT -sTCP:LISTEN > /dev/null 2>&1; then
    echo "Server found on port $SERVER_PORT"
    
    # Try to identify the process
    PID=$(lsof -i :$SERVER_PORT -sTCP:LISTEN -t)
    
    if [ -n "$PID" ]; then
        echo "Process ID: $PID"
        
        # Check if it's a Python process
        if ps -p $PID -o comm= | grep -q "python"; then
            echo "Confirmed Python process, likely a server instance."
            echo "Killing process $PID..."
            kill -9 $PID
            if [ $? -eq 0 ]; then
                echo "Process terminated successfully."
            else
                echo "Failed to terminate process. You may need sudo privileges."
            fi
        else
            echo "Warning: Found a non-Python process on port $SERVER_PORT."
            echo "Process ID: $PID"
            echo "Attempting to kill the process..."
            kill -9 $PID
        fi
    else
        echo "Could not identify the process ID."
    fi
else
    echo "No server found running on port $SERVER_PORT."
fi

# Start the server
echo
echo "==============================="
echo "Starting server on port $SERVER_PORT..."
echo "URL: http://localhost:$SERVER_PORT/"
echo "API URL: http://localhost:$SERVER_PORT/api/v2/models"
echo "==============================="
echo

python -m uvicorn llama_cpp_api_package.main:app --host 0.0.0.0 --port $SERVER_PORT --log-level info 