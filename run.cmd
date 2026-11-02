@echo off
rem Convenience launcher: drag & drop an audio file onto this file,
rem or run:  run.cmd "song.ogg" --width 2000 --zoom 20
chcp 65001 >nul
setlocal
set "HERE=%~dp0"
set "BUNDLED=%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
if exist "%BUNDLED%" (
  set "PY=%BUNDLED%"
) else (
  set "PY=python"
)
"%PY%" "%HERE%cli.py" %*
if errorlevel 1 pause
