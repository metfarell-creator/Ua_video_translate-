@echo off
setlocal EnableDelayedExpansion
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

call ".venv\Scripts\activate.bat" || ( echo [ERROR] Спочатку install_portable.bat ; exit /b 1 )
if exist "tools\ffmpeg\bin" set "PATH=%CD%\tools\ffmpeg\bin;%PATH%"
if not defined HF_HOME set "HF_HOME=%CD%\.cache\huggingface"

python ui\gradio_app.py
