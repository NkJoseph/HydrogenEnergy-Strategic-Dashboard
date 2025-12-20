@echo off
REM Deploy ML Service to Google Cloud Run (Windows)
REM Update PROJECT_ID before running

set PROJECT_ID=your-gcp-project-id
set SERVICE_NAME=hydrogen-ml-service
set REGION=us-central1

echo 🚀 Deploying ML Service to Google Cloud Run
echo ============================================

echo.
echo 📋 Configuration:
echo    Project ID: %PROJECT_ID%
echo    Service Name: %SERVICE_NAME%
echo    Region: %REGION%
echo.

REM Check if gcloud is installed
where gcloud >nul 2>&1
if errorlevel 1 (
    echo ❌ Google Cloud SDK not found!
    echo Please install from: https://cloud.google.com/sdk/docs/install
    pause
    exit /b 1
)

REM Check if docker is installed
where docker >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker not found!
    echo Please install Docker Desktop
    pause
    exit /b 1
)

echo 🔧 Step 1: Building Docker image...
docker build -f Dockerfile.ml-service -t gcr.io/%PROJECT_ID%/%SERVICE_NAME%:latest .

if errorlevel 1 (
    echo ❌ Docker build failed!
    pause
    exit /b 1
)

echo.
echo 📤 Step 2: Pushing to Google Container Registry...
docker push gcr.io/%PROJECT_ID%/%SERVICE_NAME%:latest

if errorlevel 1 (
    echo ❌ Docker push failed!
    pause
    exit /b 1
)

echo.
echo 🚀 Step 3: Deploying to Cloud Run...
gcloud run deploy %SERVICE_NAME% ^
  --image gcr.io/%PROJECT_ID%/%SERVICE_NAME%:latest ^
  --platform managed ^
  --region %REGION% ^
  --min-instances 0 ^
  --max-instances 10 ^
  --memory 2Gi ^
  --cpu 2 ^
  --timeout 300 ^
  --allow-unauthenticated ^
  --port 8080

if errorlevel 1 (
    echo ❌ Cloud Run deployment failed!
    pause
    exit /b 1
)

echo.
echo ✅ Deployment complete!
echo.
echo 📊 Getting service URL...
for /f "tokens=*" %%i in ('gcloud run services describe %SERVICE_NAME% --region %REGION% --format "value(status.url)"') do set SERVICE_URL=%%i

echo.
echo 🌐 Your ML Service URL: %SERVICE_URL%
echo.
echo 📝 Next steps:
echo 1. Copy the URL above
echo 2. Update render.yaml with ML_SERVICE_URL=%SERVICE_URL%
echo 3. Deploy API to Render
echo 4. Deploy Dashboard to Vercel
echo.

pause
