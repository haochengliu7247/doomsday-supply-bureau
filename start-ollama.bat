@echo off
setlocal
if not defined DSB_AI_ROOT set "DSB_AI_ROOT=C:\AI"
set "OLLAMA_MODELS=%DSB_AI_ROOT%\OllamaModels"
set "OLLAMA_HOST=127.0.0.1:11434"

where ollama >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Ollama was not found in PATH.
  exit /b 1
)

if not exist "%OLLAMA_MODELS%" mkdir "%OLLAMA_MODELS%"
curl.exe --silent --fail "http://%OLLAMA_HOST%/api/version" >nul 2>&1
if not errorlevel 1 (
  echo [READY] Ollama is already listening at http://%OLLAMA_HOST%.
  echo [INFO] The running process keeps the model directory it started with.
  ollama list
  exit /b 0
)

echo [OLLAMA] Models: %OLLAMA_MODELS%
echo [OLLAMA] API: http://%OLLAMA_HOST%
ollama serve
