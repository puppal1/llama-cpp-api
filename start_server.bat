@echo off
setlocal

:: Set server port (default: 8000)
set "SERVER_PORT=8000"
if not "%~1"=="" set "SERVER_PORT=%~1"

:: Set base directory to the current directory
set "BASE_DIR=%~dp0"
set "BASE_DIR=%BASE_DIR:~0,-1%"
echo Base directory: %BASE_DIR%

:: Set models directory
set "MODELS_DIR=%BASE_DIR%\models"
echo Models directory: %MODELS_DIR%

:: Check if models directory exists
if not exist "%MODELS_DIR%" (
    echo Creating models directory...
    mkdir "%MODELS_DIR%"
)

:: List model files
echo Listing model files in %MODELS_DIR%:
dir /b "%MODELS_DIR%\*.gguf" 2>nul || echo No GGUF model files found

:: Print server information
echo Starting server on port %SERVER_PORT%...
echo URL: http://localhost:%SERVER_PORT%/
echo API URL: http://localhost:%SERVER_PORT%/api/v2/models

:: Start the server
python -m uvicorn llama_cpp_api_package.main:app --host 0.0.0.0 --port %SERVER_PORT% --log-level debug

endlocal 