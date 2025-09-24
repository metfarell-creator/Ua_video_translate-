@echo off
REM Portable FFmpeg download helper
setlocal enabledelayedexpansion
set FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-release-essentials.zip
set TARGET=tools\ffmpeg

if exist "%TARGET%" (
  echo FFmpeg already exists in %TARGET%
  goto :eof
)

echo Downloading FFmpeg portable package...
powershell -Command "Invoke-WebRequest %FFMPEG_URL% -OutFile ffmpeg.zip"
mkdir %TARGET%
powershell -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; [IO.Compression.ZipFile]::ExtractToDirectory('ffmpeg.zip', '%TARGET%')"
del ffmpeg.zip

echo Done. Update PATH with %TARGET%\ffmpeg-*/bin
