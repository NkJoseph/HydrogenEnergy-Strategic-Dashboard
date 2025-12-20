@echo off
REM AWS Deployment Script for EU North 1 Region
echo ========================================
echo   AWS Deployment - EU North 1 (Stockholm)
echo ========================================
echo.

REM Set region
set AWS_REGION=eu-north-1

echo Select deployment method:
echo 1. AWS App Runner (Easiest - Recommended)
echo 2. AWS ECS with Fargate (Production)
echo 3. Exit
echo.
set /p DEPLOY_METHOD="Enter choice (1-3): "

if "%DEPLOY_METHOD%"=="1" goto APP_RUNNER
if "%DEPLOY_METHOD%"=="2" goto ECS
if "%DEPLOY_METHOD%"=="3" goto END
goto INVALID

:APP_RUNNER
echo.
echo ========================================
echo   AWS App Runner Deployment
echo ========================================
echo.
echo Follow these steps:
echo 1. Go to: https://eu-north-1.console.aws.amazon.com/apprunner
echo 2. Click "Create service"
echo 3. Select "Source code repository"
echo 4. Connect GitHub and select your repository
echo 5. Use configuration file: aws-apprunner-api.yaml
echo 6. Set service name: hydrogen-api-eu-north-1
echo 7. Set CPU: 0.5 vCPU, Memory: 1 GB, Port: 8000
echo 8. Add environment variables:
echo    - HOST=0.0.0.0
echo    - PORT=8000
echo    - PYTHONUNBUFFERED=1
echo.
echo After API is deployed, deploy dashboard with:
echo - Configuration file: aws-apprunner-dashboard.yaml
echo - Port: 8050
echo - Environment variable: HYDROGEN_API_URL=<YOUR_API_URL>
echo.
echo See AWS_QUICK_START.md for detailed instructions.
goto END

:ECS
echo.
echo ========================================
echo   AWS ECS Deployment
echo ========================================
echo.
echo This will guide you through ECS deployment steps.
echo.

REM Check AWS CLI
where aws >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: AWS CLI not found. Please install it first.
    echo Download from: https://aws.amazon.com/cli/
    goto END
)

REM Check Docker
where docker >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker not found. Please install it first.
    echo Download from: https://www.docker.com/products/docker-desktop
    goto END
)

echo Step 1: Getting AWS Account ID...
for /f "tokens=*" %%i in ('aws sts get-caller-identity --query Account --output text --region %AWS_REGION%') do set AWS_ACCOUNT_ID=%%i
if "%AWS_ACCOUNT_ID%"=="" (
    echo ERROR: Could not get AWS Account ID. Please configure AWS CLI first.
    echo Run: aws configure
    goto END
)
echo Account ID: %AWS_ACCOUNT_ID%
echo.

echo Step 2: Creating ECR repositories...
aws ecr create-repository --repository-name hydrogen-api --region %AWS_REGION% 2>nul
aws ecr create-repository --repository-name hydrogen-dashboard --region %AWS_REGION% 2>nul
echo ECR repositories created (or already exist).
echo.

echo Step 3: Logging into ECR...
for /f "tokens=*" %%i in ('aws ecr get-login-password --region %AWS_REGION%') do set ECR_PASSWORD=%%i
echo %ECR_PASSWORD% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to login to ECR.
    goto END
)
echo Logged into ECR successfully.
echo.

echo Step 4: Building Docker images...
echo Building API image...
docker build -f Dockerfile.aws-api -t hydrogen-api:latest .
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to build API image.
    goto END
)

echo Building Dashboard image...
docker build -f Dockerfile.aws-dashboard -t hydrogen-dashboard:latest .
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to build Dashboard image.
    goto END
)
echo Images built successfully.
echo.

echo Step 5: Getting repository URIs...
for /f "tokens=*" %%i in ('aws ecr describe-repositories --repository-names hydrogen-api --query "repositories[0].repositoryUri" --output text --region %AWS_REGION%') do set API_REPO_URI=%%i
for /f "tokens=*" %%i in ('aws ecr describe-repositories --repository-names hydrogen-dashboard --query "repositories[0].repositoryUri" --output text --region %AWS_REGION%') do set DASHBOARD_REPO_URI=%%i
echo API Repository: %API_REPO_URI%
echo Dashboard Repository: %DASHBOARD_REPO_URI%
echo.

echo Step 6: Tagging and pushing images...
docker tag hydrogen-api:latest %API_REPO_URI%:latest
docker tag hydrogen-dashboard:latest %DASHBOARD_REPO_URI%:latest

echo Pushing API image...
docker push %API_REPO_URI%:latest
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to push API image.
    goto END
)

echo Pushing Dashboard image...
docker push %DASHBOARD_REPO_URI%:latest
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to push Dashboard image.
    goto END
)
echo Images pushed successfully.
echo.

echo Step 7: Creating CloudWatch log groups...
aws logs create-log-group --log-group-name /ecs/hydrogen-api --region %AWS_REGION% 2>nul
aws logs create-log-group --log-group-name /ecs/hydrogen-dashboard --region %AWS_REGION% 2>nul
echo Log groups created (or already exist).
echo.

echo ========================================
echo   Next Steps:
echo ========================================
echo.
echo 1. Update task definitions:
echo    - Edit aws-ecs-task-definition-api.json
echo    - Replace YOUR_ACCOUNT_ID with: %AWS_ACCOUNT_ID%
echo    - Edit aws-ecs-task-definition-dashboard.json
echo    - Replace YOUR_ACCOUNT_ID with: %AWS_ACCOUNT_ID%
echo.
echo 2. Create IAM roles (if not exists):
echo    - Run the commands in AWS_DEPLOYMENT_GUIDE.md Step 6
echo.
echo 3. Register task definitions:
echo    aws ecs register-task-definition --cli-input-json file://aws-ecs-task-definition-api.json --region %AWS_REGION%
echo    aws ecs register-task-definition --cli-input-json file://aws-ecs-task-definition-dashboard.json --region %AWS_REGION%
echo.
echo 4. Create ECS cluster:
echo    aws ecs create-cluster --cluster-name hydrogen-cluster --region %AWS_REGION%
echo.
echo 5. Follow remaining steps in AWS_DEPLOYMENT_GUIDE.md
echo.
goto END

:INVALID
echo Invalid choice. Please select 1, 2, or 3.
goto END

:END
echo.
echo Deployment script completed.
echo For detailed instructions, see AWS_DEPLOYMENT_GUIDE.md
pause
