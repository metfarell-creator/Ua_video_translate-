@echo off
setlocal EnableDelayedExpansion
title SoniTranslate_UA_Portable - Installer

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

if not exist ".cache\huggingface" mkdir ".cache\huggingface"

echo ================================================
echo   SoniTranslate_UA_Portable: Installer
echo ================================================

where python >nul 2>nul || (
  echo [ERROR] Не знайдено Python 3.10/3.11 у PATH.
  pause & exit /b 1
)

if not exist ".venv" (
  echo [INFO] Створення .venv ...
  python -m venv .venv
)
call ".venv\Scripts\activate.bat" || ( echo [ERROR] venv ; exit /b 1 )

python -m pip install --upgrade pip setuptools wheel

echo [INFO] Встановлення базових залежностей (без Torch) ...
python -m pip install -r requirements.txt --no-deps

set "PT_OK="
echo [INFO] Спроба Torch CUDA 12.1 ...
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 && set "PT_OK=1"
if not defined PT_OK (
  echo [WARN] CUDA 12.1 недоступна. Пробуємо CUDA 11.8 ...
  python -m pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118 && set "PT_OK=1"
)
if not defined PT_OK (
  echo [WARN] Переходимо на CPU-збірки Torch ...
  python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu && set "PT_OK=1"
)
if not defined PT_OK (
  echo [ERROR] Torch install failed.
  pause & exit /b 1
)

echo [INFO] Доставляємо залежності ...
python -m pip install -r requirements.txt

echo [INFO] FFmpeg portable (Windows) ...
call tools\ffmpeg_setup.bat

echo [INFO] Prefetch моделей ...
set "HF_HOME=%CD%\.cache\huggingface"
python tools\prefetch_models.py

echo.
echo ===== Готово! Запускайте run_ui.bat =====
echo.
pause
