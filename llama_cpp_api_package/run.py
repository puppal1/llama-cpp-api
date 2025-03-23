#!/usr/bin/env python
"""
Llama.cpp API Package - Main Runner Script
This script provides a simple interface to run the llama.cpp API server.
"""

import argparse
import uvicorn
from llama_cpp_api_package.main import app

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Llama.cpp API Package")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"Starting Llama.cpp API server at http://{args.host}:{args.port}")
    
    # Use uvicorn to run the server with the app from main.py
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main() 