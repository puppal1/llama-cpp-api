from setuptools import setup, find_packages

setup(
    name="llama_cpp_api",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "llama-cpp-python",
        "psutil",
    ],
    python_requires=">=3.8",
) 