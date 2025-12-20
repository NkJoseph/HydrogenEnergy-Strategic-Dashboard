@echo off
REM Step-by-step AWS Deployment Script
echo ========================================
echo   AWS Deployment - Step by Step Guide
echo ========================================
echo.
echo Select deployment method:
echo 1. AWS App Runner (Easiest)
echo 2. AWS ECS with Fargate (Production)
echo 3. Exit
echo.
set /p DEPLOY_METHOD="Enter choice (1-3): "
echo.
echo See AWS_DEPLOY_STEPS.md for detailed instructions
pause
