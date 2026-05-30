@echo off
REM ============================================================================
REM  Lancement du backend FakeGuard / FPD (Windows)
REM ============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Activation de l'environnement virtuel s'il existe
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

REM Chargement des variables d'environnement
if exist ".env" (
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            if not "%%a"=="" set "%%a=%%b"
        )
    )
)

if "%PORT%"=="" set PORT=5000

echo ============================================================
echo   FakeGuard / FPD backend - demarrage
python --version
echo   Port   : %PORT%
echo ============================================================

python app.py
endlocal
