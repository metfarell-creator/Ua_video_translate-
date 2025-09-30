@echo off
setlocal
set "SCRIPTD=%~dp0"
set "FFDIR=%SCRIPTD%..\tools\ffmpeg"
set "FFBIN=%FFDIR%\bin\ffmpeg.exe"

if exist "%FFBIN%" (
  echo [INFO] FFmpeg вже є: %FFBIN%
  exit /b 0
)

echo [INFO] Завантаження portable FFmpeg ...
set "TMPZIP=%TEMP%\ffmpeg-release-essentials.zip"
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile '%TMPZIP%' } catch { exit 1 }"
if errorlevel 1 (
  echo [WARN] Не вдалося завантажити FFmpeg. Додайте свій FFmpeg у PATH.
  exit /b 1
)

mkdir "%FFDIR%" 2>nul
powershell -NoProfile -Command "Expand-Archive -Force '%TMPZIP%' '%FFDIR%'"

for /d %%D in ("%FFDIR%\ffmpeg-*") do (
  if exist "%%D\bin\ffmpeg.exe" (
    move /y "%%D\bin" "%FFDIR%\bin" >nul
    rmdir /s /q "%%D"
    goto :done
  )
)

:done
if exist "%FFBIN%" (
  echo [INFO] FFmpeg встановлено: %FFBIN%
  exit /b 0
) else (
  echo [WARN] FFmpeg не знайдено після розпакування.
  exit /b 1
)
