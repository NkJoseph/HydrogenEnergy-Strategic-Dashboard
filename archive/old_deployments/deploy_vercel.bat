@echo off
REM Vercel Deployment Script
REM Quick deployment to Vercel

echo 🚀 Vercel Deployment
echo ===================

echo.
echo 📋 Vercel Advantages:
echo - No size limits
echo - Works in China without VPN
echo - Free tier available
echo - Excellent performance
echo - Easy deployment
echo.

echo 🔧 Preparing for deployment...

REM Use smart-minimal requirements
echo 📦 Using smart-minimal requirements...
copy requirements-smart-minimal.txt requirements.txt

REM Use smart-minimal Vercel config
echo 📝 Using smart-minimal Vercel config...
copy vercel-smart-minimal.json vercel.json

REM Add files to git
echo 📝 Adding optimized files...
git add requirements.txt vercel.json

REM Check if there are changes to commit
git diff --staged --quiet
if errorlevel 1 (
    echo 📝 Committing changes...
    git commit -m "Prepare for Vercel deployment - %date% %time%"
) else (
    echo ℹ️  No changes to commit
)

REM Push to GitHub
echo 📤 Pushing to GitHub...
git push origin production

echo.
echo ✅ Code pushed to GitHub successfully!
echo.
echo 🌐 Next steps for Vercel deployment:
echo.
echo 🚀 Step 1: Sign Up for Vercel
echo 1. Go to https://vercel.com
echo 2. Click "Sign Up"
echo 3. Sign up with GitHub (recommended)
echo 4. Authorize Vercel to access your repositories
echo.
echo 🚀 Step 2: Deploy Your Project
echo 1. Go to Vercel Dashboard
echo 2. Click "Add New..." → "Project"
echo 3. Select your repository: NkJoseph/HydrogenEnergy-Strategic-Dashboard
echo 4. Select branch: production
echo 5. Configure Project:
echo    - Framework Preset: Other
echo    - Root Directory: ./
echo    - Build Command: Leave empty
echo    - Output Directory: Leave empty
echo    - Install Command: pip install -r requirements-smart-minimal.txt
echo 6. Click "Deploy"
echo.
echo 🚀 Step 3: Configure Environment Variables
echo In Vercel Dashboard → Settings → Environment Variables:
echo - HYDROGEN_API_URL=https://your-api-url.vercel.app
echo - PORT=8080
echo - PYTHONUNBUFFERED=1
echo.
echo 📊 Your application will be available at:
echo    - Dashboard: https://your-project.vercel.app
echo    - API: https://your-project.vercel.app/api
echo.
echo 💡 TIP: Vercel auto-deploys on every push to GitHub!
echo.

pause
