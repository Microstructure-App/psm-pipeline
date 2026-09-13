@echo off
setlocal
cd /d "%~dp0"

echo ==========================================================
echo  Uploading psm_pipeline to GitHub
echo  Repo: Microstructure-App/psm-pipeline
echo ==========================================================
echo.

git --version >nul 2>&1
if errorlevel 1 (
  echo [X] Git is not installed.
  echo.
  echo     Install it from https://git-scm.com/download/win
  echo     Click Next through every screen - the defaults are correct.
  echo     Then run this file again.
  echo.
  pause
  exit /b 1
)

echo [1/6] Git found.

if not exist ".git" (
  git init -b main
  echo [2/6] Created a new repository here.
) else (
  echo [2/6] Repository already exists here.
)

git config user.name  "Ishveer Randhawa"
git config user.email "ishveer.sr@gmail.com"
echo [3/6] Set your name and email for this repo.

git add -A
echo [4/6] Staged files ^(.gitignore excludes .venv, __pycache__, build, dist, outputs, *.vtk, *.zip^).

git diff --cached --quiet
if errorlevel 1 (
  git commit -m "Add psm pipeline: 2D SEM microstructure analysis tool"
  echo [5/6] Committed.
) else (
  echo [5/6] Nothing new to commit.
)

git remote remove origin >nul 2>&1
git remote add origin https://github.com/Microstructure-App/psm-pipeline.git
echo [6/6] Pointed at Microstructure-App/psm-pipeline.

echo.
echo Pushing... a browser window will open asking you to sign in to GitHub.
echo Sign in and approve - that happens once, then never again.
echo.

git push -u origin main

if errorlevel 1 (
  echo.
  echo ==========================================================
  echo  PUSH FAILED - copy the red text above and send it to Claude.
  echo ==========================================================
) else (
  echo.
  echo ==========================================================
  echo  DONE. Your files are now at:
  echo  https://github.com/Microstructure-App/psm-pipeline
  echo ==========================================================
)

echo.
pause
