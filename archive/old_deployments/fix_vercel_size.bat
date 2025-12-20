@echo off
REM Vercel Size Optimization Script
REM Fixes size issues for Vercel deployment

echo 🚀 Vercel Size Optimization
echo ==========================

echo.
echo 📋 Vercel Size Limits:
echo - Max file size: 4GB per file
echo - Total deployment: 100MB (free tier)
echo - Function timeout: 30 seconds
echo.

echo 🔧 Applying optimizations...

REM Use minimal requirements
echo 📦 Using minimal requirements...
copy requirements-china.txt requirements.txt

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

REM Create .vercelignore
echo 📝 Creating .vercelignore file...
echo pretrained-models/ > .vercelignore
echo processed_data/ >> .vercelignore
echo *.pt >> .vercelignore
echo *.pkl >> .vercelignore
echo *.zip >> .vercelignore
echo dataset/ >> .vercelignore
echo *.md >> .vercelignore
echo Dockerfile* >> .vercelignore
echo docker-compose*.yml >> .vercelignore
echo .dockerignore >> .vercelignore

REM Add files to git
echo 📝 Adding optimized files...
git add requirements.txt
git add .vercelignore
git add vercel.json

REM Check if there are changes to commit
git diff --staged --quiet
if errorlevel 1 (
    echo 📝 Committing optimizations...
    git commit -m "Optimize for Vercel deployment - %date% %time%"
) else (
    echo ℹ️  No changes to commit
)

REM Push to GitHub
echo 📤 Pushing to GitHub...
git push origin production

echo.
echo ✅ Vercel optimization complete!
echo.
echo 🌐 Next steps:
echo 1. Go to Vercel dashboard
echo 2. Click 'Redeploy' or 'Deploy'
echo 3. Vercel will use the optimized configuration
echo.
echo 📊 Optimizations applied:
echo - Minimal requirements (no heavy dependencies)
echo - Removed large files (pretrained-models, processed_data)
echo - Added .vercelignore file
echo - Optimized vercel.json configuration
echo.
echo 💡 If still too big, try Railway or Heroku instead!
echo.

pause
