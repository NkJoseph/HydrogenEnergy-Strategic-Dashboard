@echo off
REM Railway Size Optimization Script
REM Fixes size issues for Railway free tier

echo 🚀 Railway Size Optimization
echo ============================

echo.
echo 📋 Railway Free Tier Limits:
echo - Memory: 512MB RAM
echo - Storage: 1GB disk space
echo - Build time: 90 minutes/month
echo.

echo 🔧 Applying optimizations...

REM Use minimal requirements
echo 📦 Using minimal requirements...
copy requirements-railway-minimal.txt requirements.txt

REM Remove large files
echo 🗑️  Removing large files...
if exist pretrained-models\ (
    echo Removing pretrained-models directory...
    rmdir /s /q pretrained-models
)

if exist processed_data\ (
    echo Removing processed_data directory...
    rmdir /s /q processed_data
)

REM Add files to git
echo 📝 Adding optimized files...
git add requirements.txt
git add .railwayignore
git add railway.json

REM Check if there are changes to commit
git diff --staged --quiet
if errorlevel 1 (
    echo 📝 Committing optimizations...
    git commit -m "Optimize for Railway free tier - %date% %time%"
) else (
    echo ℹ️  No changes to commit
)

REM Push to GitHub
echo 📤 Pushing to GitHub...
git push origin production

echo.
echo ✅ Railway optimization complete!
echo.
echo 🌐 Next steps:
echo 1. Go to Railway dashboard
echo 2. Click 'Redeploy' or 'Deploy'
echo 3. Railway will use the optimized configuration
echo.
echo 📊 Optimizations applied:
echo - Minimal requirements (no heavy dependencies)
echo - Removed large files (pretrained-models, processed_data)
echo - Added .railwayignore file
echo - Optimized railway.json configuration
echo.
echo 💡 If still too big, try Vercel or Heroku instead!
echo.

pause
