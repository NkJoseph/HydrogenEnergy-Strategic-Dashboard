@echo off
REM Free Tier Deployment Script for Render.com
REM This script helps deploy without Blueprint (no payment required)

echo 🚀 Hydrogen Dashboard - Free Tier Deployment
echo =============================================

echo.
echo 📋 Free Tier Deployment Options:
echo.
echo 1. Manual Service Creation (Recommended)
echo    - Create API service manually
echo    - Create Dashboard service manually
echo    - No payment information required
echo.
echo 2. Single Service Deployment
echo    - Deploy only Dashboard
echo    - API embedded in Dashboard
echo    - Simplest option
echo.
echo 3. Alternative Platforms
echo    - Heroku (free tier)
echo    - Railway (free tier)
echo    - Vercel (free tier)
echo.

echo 🔧 Preparing for deployment...

REM Add all files
git add .

REM Check if there are changes to commit
git diff --staged --quiet
if errorlevel 1 (
    echo 📝 Committing changes...
    git commit -m "Prepare for free tier deployment - %date% %time%"
) else (
    echo ℹ️  No changes to commit
)

REM Push to GitHub
echo 📤 Pushing to GitHub...
git push origin production

echo.
echo ✅ Code pushed to GitHub successfully!
echo.
echo 🌐 Next steps for FREE TIER deployment:
echo.
echo 📋 Manual Service Creation (No Payment Required):
echo 1. Go to https://render.com
echo 2. Click 'New +' → 'Web Service' (NOT Blueprint)
echo 3. Connect your GitHub repository
echo 4. Create API service:
echo    - Name: hydrogen-api
echo    - Build: pip install -r requirements.txt
echo    - Start: uvicorn api:app --host 0.0.0.0 --port $PORT
echo    - Health: /health
echo    - Plan: Starter (Free)
echo.
echo 5. Create Dashboard service:
echo    - Name: hydrogen-dashboard
echo    - Build: pip install -r requirements.txt
echo    - Start: python start_dashboard_render.py
echo    - Health: /
echo    - Plan: Starter (Free)
echo    - Env: HYDROGEN_API_URL=https://hydrogen-api.onrender.com
echo.
echo 📊 Your services will be available at:
echo    API: https://hydrogen-api.onrender.com
echo    Dashboard: https://hydrogen-dashboard.onrender.com
echo.
echo 💡 TIP: Use manual service creation to avoid payment requirement!
echo.

pause
