@echo off
rem Build the Windows GUI package (onedir) with PyInstaller.
rem Usage:  build_exe.cmd "C:\path\to\python.exe"
rem Prereq: the given python must have PyInstaller available
rem         (pip install pyinstaller, or PYTHONPATH pointing at a local copy).
setlocal
if "%~1"=="" (set "PY=python") else (set "PY=%~1")

"%PY%" -m PyInstaller --noconfirm --clean --onedir --windowed ^
  --name MalodyAutoChart ^
  --icon "assets\icon.ico" ^
  --paths . --paths vendor ^
  --add-data "vendor;vendor" ^
  --add-data "assets;assets" ^
  --hidden-import tkinterdnd2 --hidden-import soundfile ^
  --hidden-import cffi --hidden-import _cffi_backend ^
  --exclude-module sounddevice --exclude-module _sounddevice ^
  --exclude-module _sounddevice_data ^
  --exclude-module matplotlib --exclude-module pandas --exclude-module scipy ^
  gui.py

if errorlevel 1 (
  echo Build failed.
  exit /b 1
)
echo.
echo Built: dist\MalodyAutoChart\MalodyAutoChart.exe
echo Zip it (example):
echo   Compress-Archive dist\MalodyAutoChart dist\MalodyAutoChart-win64.zip -Force
