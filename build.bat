@echo off
REM ============================================================
REM  PSM Pipeline -- Windows exe build
REM  Run from the repo root, INSIDE the project venv.
REM ============================================================

echo [1/4] Installing build dependencies...
pip install --upgrade pyinstaller pywebview || goto :fail

echo [2/4] Ensuring psm is importable...
python -c "import psm" 2>nul || pip install -e . || goto :fail

echo [3/4] Sanity check: app.py present?
if not exist app.py (
    echo ERROR: run this from the repo root ^(app.py not found^).
    goto :fail
)

echo [4/4] Building...
pyinstaller --noconfirm --clean psm_pipeline.spec || goto :fail

echo.
echo Build complete:  dist\PSM_Pipeline\PSM_Pipeline.exe
echo Distribute the ENTIRE dist\PSM_Pipeline folder (zip it).
exit /b 0

:fail
echo.
echo BUILD FAILED -- scroll up for the first error.
exit /b 1
