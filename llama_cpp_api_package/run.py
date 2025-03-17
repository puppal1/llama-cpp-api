#!/usr/bin/env python
"""
Llama.cpp API Package - Main Runner Script
This script provides a simple interface to run the llama.cpp API server,
download models, and update the web interface.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Llama.cpp API Package")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model from Hugging Face")
    download_parser.add_argument("--model", required=True, help="Model repository (e.g., microsoft/Phi-3-mini-4k-instruct-gguf)")
    download_parser.add_argument("--filename", required=True, help="Filename to download (e.g., Phi-3-mini-4k-instruct-q4.gguf)")
    download_parser.add_argument("--token", help="Hugging Face token for gated models")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update the web interface with available models")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    
    args = parser.parse_args()
    
    if args.command == "server":
        print(f"Starting Llama.cpp API server at http://{args.host}:{args.port}")
        print(f"Web interface available at http://{args.host}:{args.port}/static/index.html")
        # Use the correct path to llama_server.py
        server_path = os.path.join(os.path.dirname(__file__), "llama_server.py")
        subprocess.run([sys.executable, server_path, "--host", args.host, "--port", str(args.port)])
        
    elif args.command == "download":
        print(f"Downloading model {args.filename} from {args.model}...")
        cmd = [sys.executable, "-m", "utils.download_models", 
               "--model", args.model, 
               "--filename", args.filename]
        if args.token:
            cmd.extend(["--token", args.token])
        subprocess.run(cmd)
        
        # After download, update the web interface
        print("\nUpdating web interface with new model...")
        subprocess.run([sys.executable, "-m", "utils.update_web_interface"])
        
    elif args.command == "update":
        print("Updating web interface with available models...")
        subprocess.run([sys.executable, "-m", "utils.update_web_interface"])
        
    elif args.command == "list":
        models_dir = Path("models")
        if not models_dir.exists() or not any(models_dir.glob("*.gguf")):
            print("No models found. Use the 'download' command to download models.")
            return
        
        print("Available models:")
        for model_file in models_dir.glob("*.gguf"):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  - {model_file.name} ({size_mb:.1f} MB)")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 