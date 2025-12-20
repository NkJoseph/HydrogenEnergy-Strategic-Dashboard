@echo off
REM Render.com Deployment Script for Windows
REM This script helps prepare and deploy to Render.com

echo 🚀 Hydrogen Dashboard - Render Deployment Script
echo ==================================================

REM Check if git is available
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git is not installed. Please install git first.
    pause
    exit /b 1
)

REM Check if we're in a git repository
if not exist ".git" (
    echo ❌ Not in a git repository. Please initialize git first.
    pause
    exit /b 1
)

echo 📋 Checking required files...

REM Check for required files
set "required_files=api.py dashboard.py requirements.txt render.yaml runtime.txt"

for %%f in (%required_files%) do (
    if exist "%%f" (
        echo ✅ %%f exists
    ) else (
        echo ❌ %%f is missing
        pause
        exit /b 1
    )
)

echo.
echo 🔧 Preparing for deployment...

REM Add all files to git
git add .

REM Check if there are changes to commit
git diff --staged --quiet
if errorlevel 1 (
    echo 📝 Committing changes...
    git commit -m "Prepare for Render deployment - %date% %time%"
) else (
    echo ℹ️  No changes to commit
)

REM Push to origin
echo 📤 Pushing to GitHub...
git push origin main

echo.
echo ✅ Code pushed to GitHub successfully!
echo.
echo 🌐 Next steps:
echo 1. Go to https://render.com
echo 2. Sign up/Login with GitHub
echo 3. Click 'New +' → 'Blueprint'
echo 4. Connect your GitHub repository
echo 5. Render will automatically detect render.yaml
echo 6. Review and deploy your services
echo.
echo 📊 Your services will be available at:
echo    API: https://hydrogen-api.onrender.com
echo    Dashboard: https://hydrogen-dashboard.onrender.com
echo.
echo 🔍 Monitor deployment in Render dashboard
echo 📝 Check logs if deployment fails

pause
