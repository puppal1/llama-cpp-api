@echo off
setlocal enabledelayedexpansion

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
set "MODELS_DIR=%MODELS_DIR%"

:: Check if models directory exists
if not exist "%MODELS_DIR%" (
    echo Creating models directory...
    mkdir "%MODELS_DIR%"
)

:: List model files
echo Listing model files in %MODELS_DIR%:
dir /b "%MODELS_DIR%\*.gguf" 2>nul || echo No GGUF model files found

:: Kill any existing server on the same port
echo Checking for existing server instances on port %SERVER_PORT%...
netstat -ano | findstr ":%SERVER_PORT% " | findstr "LISTENING" > nul
if %ERRORLEVEL% EQU 0 (
    echo Server found on port %SERVER_PORT%
    
    :: Try to identify the process
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%SERVER_PORT% " ^| findstr "LISTENING"') do (
        set PID=%%a
    )
    
    if defined PID (
        echo Process ID: !PID!
        
        :: Check if it's a Python process
        for /f "tokens=1" %%b in ('tasklist /fi "PID eq !PID!" ^| findstr "python"') do (
            set IS_PYTHON=%%b
        )
        
        if defined IS_PYTHON (
            echo Confirmed Python process, likely a server instance.
            echo Killing process !PID!...
            taskkill /F /PID !PID!
            if %ERRORLEVEL% EQU 0 (
                echo Process terminated successfully.
            ) else (
                echo Failed to terminate process. You may need administrator privileges.
            )
        ) else (
            echo Warning: Found a non-Python process on port %SERVER_PORT%.
            echo Process ID: !PID!
            echo Attempting to kill the process...
            taskkill /F /PID !PID!
        )
    ) else (
        echo Could not identify the process ID.
    )
) else (
    echo No server found running on port %SERVER_PORT%.
)

:: Start the server
echo.
echo ===============================
echo Starting server on port %SERVER_PORT%...
echo URL: http://localhost:%SERVER_PORT%/
echo API URL: http://localhost:%SERVER_PORT%/api/v2/models
echo ===============================
echo.

python -m uvicorn llama_cpp_api_package.main:app --host 0.0.0.0 --port %SERVER_PORT% --log-level info

endlocal 