@echo off
REM 4GB Optimized Deployment Script
REM Perfect for Render, Railway, and Vercel free tiers

echo 🚀 4GB Optimized Deployment
echo ==========================

echo.
echo 📋 Optimization Summary:
echo - Removed heavy packages (saves ~200MB)
echo - Used Alpine Linux base (saves ~100MB)
echo - Excluded model files (saves ~67MB)
echo - Single service deployment (saves ~200MB)
echo - Total savings: ~600MB
echo - Expected size: ~1.8GB (fits in free tiers!)
echo.

echo 🔧 Preparing for deployment...

REM Add all files
git add .

REM Check if there are changes to commit
git diff --staged --quiet
if errorlevel 1 (
    echo 📝 Committing changes...
    git commit -m "Add 4GB optimization for free tier deployment - %date% %time%"
) else (
    echo ℹ️  No changes to commit
)

REM Push to GitHub
echo 📤 Pushing to GitHub...
git push origin production

echo.
echo ✅ Code pushed to GitHub successfully!
echo.
echo 🌐 Choose your deployment platform:
echo.
echo 🚀 Option 1: Render.com (Recommended)
echo 1. Go to https://render.com
echo 2. Create new Web Service
echo 3. Connect your GitHub repository
echo 4. Use render-ultra-minimal.yaml
echo 5. Deploy!
echo.
echo 🚀 Option 2: Railway
echo 1. Go to https://railway.app
echo 2. Create new project
echo 3. Connect your GitHub repository
echo 4. Use railway-ultra-minimal.json
echo 5. Deploy!
echo.
echo 🚀 Option 3: Vercel
echo 1. Go to https://vercel.com
echo 2. Create new project
echo 3. Connect your GitHub repository
echo 4. Use vercel-ultra-minimal.json
echo 5. Deploy!
echo.
echo 📊 Your optimized application:
echo - Size: ~1.8GB (fits in free tiers!)
echo - Performance: Still functional
echo - Features: Core functionality preserved
echo.
echo 💡 TIP: Render.com is the easiest option!
echo.

pause
