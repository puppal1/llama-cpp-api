#!/bin/bash

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
if [ -z "$(find "$MODELS_DIR" -name "*.gguf" -type f)" ]; then
    echo "No GGUF model files found"
fi

# Start the server
echo "Starting server..."
python -m uvicorn llama_cpp_api_package.main:app --host 0.0.0.0 --port 8000 --log-level debug 