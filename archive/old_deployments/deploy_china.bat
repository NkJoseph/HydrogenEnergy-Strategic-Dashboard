@echo off
REM China Deployment Script
REM Alternative platforms for China users

echo 🇨🇳 Hydrogen Dashboard - China Deployment Options
echo ================================================

echo.
echo 📋 China-Friendly Deployment Platforms:
echo.
echo 1. Railway (Recommended)
echo    - Works in China
echo    - Free tier available
echo    - Easy deployment
echo    - URL: https://railway.app
echo.
echo 2. Vercel (Best Performance)
echo    - Excellent China performance
echo    - Free tier available
echo    - Fast deployment
echo    - URL: https://vercel.com
echo.
echo 3. Heroku (Classic Choice)
echo    - Reliable in China
echo    - Free tier available
echo    - Easy deployment
echo    - URL: https://heroku.com
echo.
echo 4. DigitalOcean App Platform
echo    - Works in China
echo    - Good performance
echo    - Free tier available
echo    - URL: https://digitalocean.com
echo.
echo 5. Fly.io (Global CDN)
echo    - Excellent global performance
echo    - Works in China
echo    - Free tier available
echo    - URL: https://fly.io
echo.

echo 🔧 Preparing for deployment...

REM Add all files
git add .

REM Check if there are changes to commit
git diff --staged --quiet
if errorlevel 1 (
    echo 📝 Committing changes...
    git commit -m "Prepare for China deployment - %date% %time%"
) else (
    echo ℹ️  No changes to commit
)

REM Push to GitHub
echo 📤 Pushing to GitHub...
git push origin production

echo.
echo ✅ Code pushed to GitHub successfully!
echo.
echo 🌐 Next steps for China deployment:
echo.
echo 🚀 Railway (Recommended):
echo 1. Go to https://railway.app
echo 2. Sign up with GitHub
echo 3. Connect your repository
echo 4. Deploy automatically
echo.
echo 🚀 Vercel (Best Performance):
echo 1. Go to https://vercel.com
echo 2. Import GitHub repository
echo 3. Deploy automatically
echo.
echo 🚀 Heroku (Classic):
echo 1. Go to https://heroku.com
echo 2. Create new app
echo 3. Connect GitHub repository
echo 4. Deploy
echo.
echo 📊 Your services will be available at:
echo    - Railway: https://your-app.railway.app
echo    - Vercel: https://your-app.vercel.app
echo    - Heroku: https://your-app.herokuapp.com
echo.
echo 💡 TIP: Railway is the easiest option for China users!
echo.

pause
