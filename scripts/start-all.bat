@echo off
echo Starting Katib OCR API on :8000 ...
start "Katib API" cmd /k "%~dp0start-api.bat"
timeout /t 2 >nul
echo Starting Katib UI on :3000 ...
start "Katib Web" cmd /k "%~dp0start-web.bat"
echo.
echo Open http://localhost:3000 when both windows are ready.
echo First API start downloads the model (~7.5GB).
