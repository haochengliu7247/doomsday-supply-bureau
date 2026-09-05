@echo off
setlocal
if not defined DSB_AI_ROOT set "DSB_AI_ROOT=C:\AI"
set "COMFY_ROOT=%DSB_AI_ROOT%\ComfyUI_windows_portable"

if not exist "%COMFY_ROOT%\python_embeded\python.exe" (
  echo [ERROR] ComfyUI was not found at %COMFY_ROOT%.
  echo Run scripts\install_comfyui.ps1 first.
  exit /b 1
)

curl.exe --silent --fail "http://127.0.0.1:8188/system_stats" >nul 2>&1
if not errorlevel 1 (
  echo [READY] ComfyUI is already listening at http://127.0.0.1:8188.
  exit /b 0
)

cd /d "%COMFY_ROOT%"
"%COMFY_ROOT%\python_embeded\python.exe" -s ComfyUI\main.py ^
  --windows-standalone-build ^
  --listen 127.0.0.1 ^
  --port 8188 ^
  --disable-auto-launch ^
  --reserve-vram 1 ^
  --preview-method none
