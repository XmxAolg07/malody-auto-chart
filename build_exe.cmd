@echo off
rem Build the Windows package (onedir) with PyInstaller.
rem Usage:  build_exe.cmd "C:\path\to\python.exe"
rem Prereq: the given python must have PyInstaller available
rem         (pip install pyinstaller, or PYTHONPATH pointing at a local copy).
setlocal
if "%~1"=="" (set "PY=python") else (set "PY=%~1")

"%PY%" -m PyInstaller --noconfirm --clean --onedir --console ^
  --name MalodyAudioAnalyzer ^
  --paths . --paths vendor ^
  --add-data "vendor;vendor" ^
  --hidden-import soundfile --hidden-import cffi --hidden-import _cffi_backend ^
  --exclude-module tkinter --exclude-module matplotlib ^
  --exclude-module pandas --exclude-module scipy ^
  cli.py

if errorlevel 1 (
  echo Build failed.
  exit /b 1
)
echo.
echo Built: dist\MalodyAudioAnalyzer\MalodyAudioAnalyzer.exe
echo Zip it (example):
echo   Compress-Archive dist\MalodyAudioAnalyzer dist\MalodyAudioAnalyzer-win64.zip -Force
