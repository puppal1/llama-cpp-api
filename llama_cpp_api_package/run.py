#!/usr/bin/env python
"""
Llama.cpp API Package - Main Runner Script
This script provides a simple interface to run the llama.cpp API server.
"""

import argparse
import uvicorn

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Llama.cpp API Package")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    print(f"Starting Llama.cpp API server at http://{args.host}:{args.port}")
    
    # Use uvicorn to run the server with the app from main.py
    uvicorn.run(
        "llama_cpp_api_package.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main() 