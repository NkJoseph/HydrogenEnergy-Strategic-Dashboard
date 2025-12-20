@echo off
REM Alibaba Cloud Deployment Script
REM Perfect for China users with Alipay/WeChat Pay

echo 🇨🇳 Alibaba Cloud Deployment
echo ============================

echo.
echo 📋 Alibaba Cloud Advantages:
echo - Accepts Alipay and WeChat Pay
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
    git commit -m "Prepare for Alibaba Cloud deployment - %date% %time%"
) else (
    echo ℹ️  No changes to commit
)

REM Push to GitHub
echo 📤 Pushing to GitHub...
git push origin production

echo.
echo ✅ Code pushed to GitHub successfully!
echo.
echo 🌐 Next steps for Alibaba Cloud deployment:
echo.
echo 🚀 Step 1: Create Alibaba Cloud Account
echo 1. Go to https://aliyun.com
echo 2. Sign up with Alipay or WeChat Pay
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
echo Option B: ECS (Full Control)
echo - Go to ECS
echo - Create instance
echo - Install Docker
echo - Deploy your application
echo.
echo Option C: Function Compute (Serverless)
echo - Go to Function Compute
echo - Create new function
echo - Upload your code
echo - Configure triggers
echo.
echo 🚀 Step 3: Deploy Your Application
echo 1. Use the provided Dockerfile (alibaba-cloud-dockerfile)
echo 2. Use the Kubernetes config (k8s-deployment.yaml)
echo 3. Follow the deployment guide (ALIBABA_CLOUD_DEPLOYMENT.md)
echo.
echo 📊 Your application will be available at:
echo    - Container Service: https://your-cluster.aliyuncs.com
echo    - ECS: http://your-ecs-ip:8080
echo    - Function Compute: https://your-function.aliyuncs.com
echo.
echo 💡 TIP: Container Service is the easiest option for beginners!
echo.

pause
