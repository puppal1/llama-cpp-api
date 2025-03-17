@echo off
echo Llama.cpp API Package
echo ---------------------

if "%1"=="" (
    echo Available commands:
    echo   server  - Start the API server
    echo   download - Download a model
    echo   update  - Update the web interface
    echo   list    - List available models
    echo.
    echo Example: llama_api.bat server
    exit /b
)

python run.py %* 