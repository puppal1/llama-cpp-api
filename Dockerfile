FROM python:3.9-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    make && \
    mkdir -p /usr/local/lib && \
    cp libllama.so /usr/local/lib/ && \
    cp libggml*.so /usr/local/lib/ && \
    ldconfig

# Set up the application
WORKDIR /app
COPY . /app/

# Create binary directories
RUN mkdir -p /app/llama_cpp_api_package/bin/linux && \
    cp /usr/local/lib/libllama.so /app/llama_cpp_api_package/bin/linux/ && \
    cp /usr/local/lib/libggml*.so /app/llama_cpp_api_package/bin/linux/

# Install the package
RUN pip install -e .

# Create models directory
RUN mkdir -p /app/llama_cpp_api_package/models

# Set environment variables
ENV LLAMA_API_HOST=0.0.0.0
ENV LLAMA_API_PORT=8000
ENV LLAMA_MODEL_PATH=/app/llama_cpp_api_package/models

# Expose the port
EXPOSE 8000

# Run the application
CMD ["llama-cpp-api"] 