@echo off
REM Project-S V2 - GitHub Upload Script
REM This script initializes git and uploads the project to GitHub

echo.
echo ========================================
echo Project-S V2 - GitHub Upload
echo ========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git for Windows from: https://git-scm.com/download/win
    pause
    exit /b 1
)

REM Initialize git repository
echo Initializing Git repository...
git init

REM Add remote origin
echo Adding GitHub remote...
git remote add origin https://github.com/toti85/project-s-agent-v2.git

REM Add all files
echo Adding files to Git...
git add .

REM Commit
echo Creating initial commit...
git commit -m "Initial commit: Project-S V2 - Advanced Autonomous AI Agent"

REM Push to GitHub
echo Pushing to GitHub...
git branch -M main
git push -u origin main

echo.
echo ========================================
echo Upload Complete!
echo ========================================
echo.
echo Your Project-S V2 repository is now available at:
echo https://github.com/toti85/project-s-agent-v2
echo.
pause
