from setuptools import setup, find_packages
import os
import platform

def get_binary_files():
    # Define OS-specific binary files
    binary_files = {
        'Windows': ['*.dll'],
        'Linux': ['*.so'],
        'Darwin': ['*.dylib']
    }
    current_os = platform.system()
    return binary_files.get(current_os, [])

# Read README with explicit UTF-8 encoding
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except Exception as e:
    print(f"Warning: Could not read README.md: {e}")
    long_description = "A web interface for llama.cpp models"

setup(
    name="llama_cpp_api_package",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "llama-cpp-python",
        "psutil",
        "pytest",
        "httpx"
    ],
) 