@echo off
REM Quick script to push important files to GitHub
REM This script adds only the essential files for deployment

echo 🚀 Pushing Hydrogen Dashboard to GitHub
echo ======================================

REM Add important files
git add .gitignore
git add api.py dashboard.py requirements.txt render.yaml runtime.txt Procfile config.py start_api.py
git add README.md DEPLOYMENT.md DOCKER_DEPLOYMENT.md RENDER_DEPLOYMENT.md
git add Dockerfile* docker-compose*.yml .dockerignore deploy_render.* render_start.py start_render.py requirements-render*.txt nginx.conf
git add dataset/

REM Check status
echo.
echo 📋 Files to be committed:
git status --porcelain

REM Commit with timestamp
echo.
echo 📝 Committing changes...
git commit -m "Update deployment configuration - %date% %time%"

REM Push to GitHub
echo.
echo 📤 Pushing to GitHub...
git push origin production

echo.
echo ✅ Successfully pushed to GitHub!
echo.
echo 🌐 Your repository is ready for deployment:
echo    - Docker: Use docker-compose up --build
echo    - Render: Connect to https://render.com
echo.
pause
