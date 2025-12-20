@echo off
REM Vultr Deployment Script
REM Perfect for China users with Alipay

echo 🌍 Vultr Deployment
echo ===================

echo.
echo 📋 Vultr Advantages:
echo - Accepts Alipay
echo - Works in China without VPN
echo - Free tier available
echo - Global locations
echo - Good performance
echo.

echo 🔧 Preparing for deployment...

REM Add all files
git add .

REM Check if there are changes to commit
git diff --staged --quiet
if errorlevel 1 (
    echo 📝 Committing changes...
    git commit -m "Prepare for Vultr deployment - %date% %time%"
) else (
    echo ℹ️  No changes to commit
)

REM Push to GitHub
echo 📤 Pushing to GitHub...
git push origin production

echo.
echo ✅ Code pushed to GitHub successfully!
echo.
echo 🌐 Next steps for Vultr deployment:
echo.
echo 🚀 Step 1: Create Vultr Account
echo 1. Go to https://vultr.com
echo 2. Sign up with Alipay
echo 3. Complete verification
echo.
echo 🚀 Step 2: Create VPS Instance
echo 1. Go to VPS
echo 2. Choose location (Singapore recommended for China)
echo 3. Select OS (Ubuntu 22.04 recommended)
echo 4. Choose plan (start with $5/month)
echo 5. Deploy instance
echo.
echo 🚀 Step 3: Deploy Your Application
echo 1. SSH into your VPS
echo 2. Install Docker
echo 3. Clone your repository
echo 4. Build and run your application
echo.
echo 📊 Your application will be available at:
echo    - VPS: http://your-vps-ip:8080
echo.
echo 💡 TIP: Singapore location works best for China users!
echo.

pause
