@echo off
REM Tencent Cloud Deployment Script
REM Perfect for China users with WeChat Pay/Alipay

echo 🇨🇳 Tencent Cloud Deployment
echo ============================

echo.
echo 📋 Tencent Cloud Advantages:
echo - Accepts WeChat Pay and Alipay
echo - Works perfectly in China
echo - Free tier available
echo - Excellent performance in China
echo - Chinese language support
echo.

echo 🔧 Preparing for deployment...

REM Add all files
git add .

REM Check if there are changes to commit
git diff --staged --quiet
if errorlevel 1 (
    echo 📝 Committing changes...
    git commit -m "Prepare for Tencent Cloud deployment - %date% %time%"
) else (
    echo ℹ️  No changes to commit
)

REM Push to GitHub
echo 📤 Pushing to GitHub...
git push origin production

echo.
echo ✅ Code pushed to GitHub successfully!
echo.
echo 🌐 Next steps for Tencent Cloud deployment:
echo.
echo 🚀 Step 1: Create Tencent Cloud Account
echo 1. Go to https://cloud.tencent.com
echo 2. Sign up with WeChat or Alipay
echo 3. Complete verification
echo.
echo 🚀 Step 2: Choose Deployment Method
echo.
echo Option A: Container Service (Recommended)
echo - Go to Container Service
echo - Create Kubernetes cluster
echo - Upload your Docker image
echo - Deploy application
echo.
echo Option B: CVM (Full Control)
echo - Go to CVM
echo - Create instance
echo - Install Docker
echo - Deploy your application
echo.
echo Option C: Serverless (Serverless)
echo - Go to Serverless
echo - Create new function
echo - Upload your code
echo - Configure triggers
echo.
echo 🚀 Step 3: Deploy Your Application
echo 1. Use the provided Dockerfile (tencent-cloud-dockerfile)
echo 2. Use the Kubernetes config (tencent-k8s.yaml)
echo 3. Follow the deployment guide (CHINA_FRIENDLY_DEPLOYMENT.md)
echo.
echo 📊 Your application will be available at:
echo    - Container Service: https://your-cluster.tencentcloud.com
echo    - CVM: http://your-cvm-ip:8080
echo    - Serverless: https://your-function.tencentcloud.com
echo.
echo 💡 TIP: Container Service is the easiest option for beginners!
echo.

pause
