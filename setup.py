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

setup(
    name="llama_cpp_api",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
        "psutil>=5.8.0",
        "numpy>=1.21.0",
        "pydantic>=1.8.0",
    ],
    package_data={
        'llama_cpp_api_package': [
            'static/css/*',
            'static/js/*',
            'static/index.html',
            'models/*.gguf',
            f'bin/{platform.system().lower()}/*'  # OS-specific binaries
        ]
    },
    entry_points={
        'console_scripts': [
            'llama-cpp-api=llama_cpp_api_package.run:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A web interface for llama.cpp models",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    keywords="llama, machine learning, api",
    url="https://github.com/yourusername/llama_cpp_api",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 