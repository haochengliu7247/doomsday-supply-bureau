@echo off
setlocal
cd /d "%~dp0"
set "GRADIO_ANALYTICS_ENABLED=False"

where uv >nul 2>&1
if errorlevel 1 (
  echo [ERROR] uv was not found. Install uv or add it to PATH.
  exit /b 1
)

uv sync --frozen --python 3.12
if errorlevel 1 exit /b 1

uv run python app.py
