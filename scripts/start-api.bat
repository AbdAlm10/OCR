@echo off
setlocal EnableExtensions
cd /d "%~dp0..\backend"
if errorlevel 1 (
  echo Failed to enter backend folder.
  exit /b 1
)

if not exist requirements.txt (
  echo requirements.txt not found in %CD%
  exit /b 1
)

if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create .venv. Install Python 3.11 or 3.12 if possible.
    exit /b 1
  )
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
  echo Failed to activate .venv
  exit /b 1
)

echo Installing / updating dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Dependency install failed.
  echo Tip: this project works best with Python 3.11 or 3.12 ^(torch may not support 3.14 yet^).
  exit /b 1
)

echo Starting API on http://127.0.0.1:8000 ...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
