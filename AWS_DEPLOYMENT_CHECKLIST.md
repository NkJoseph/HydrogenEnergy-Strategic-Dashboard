# AWS Deployment Checklist - EU North 1

Use this checklist to track your deployment progress.

## 📋 Pre-Deployment

### Prerequisites
- [ ] AWS Account created
- [ ] AWS CLI installed (`aws --version`)
- [ ] AWS CLI configured for `eu-north-1` region (`aws configure`)
- [ ] Docker installed (for ECS deployment)
- [ ] Git repository with code pushed to GitHub (for App Runner)
- [ ] Code tested locally and working

### Code Preparation
- [ ] All dependencies in `requirements.txt`
- [ ] `api.py` exists and works
- [ ] `dashboard.py` exists and works
- [ ] `start_api.py` exists and configured correctly
- [ ] Environment variables documented

---

## 🚀 AWS App Runner Deployment

### API Service
- [ ] Navigated to App Runner console (eu-north-1)
- [ ] Created new service
- [ ] Connected GitHub repository
- [ ] Selected branch (main/master)
- [ ] Set configuration file: `aws-apprunner-api.yaml`
- [ ] Set service name: `hydrogen-api-eu-north-1`
- [ ] Set CPU: 0.5 vCPU
- [ ] Set Memory: 1 GB
- [ ] Set Port: 8000
- [ ] Added environment variables:
  - [ ] `HOST=0.0.0.0`
  - [ ] `PORT=8000`
  - [ ] `PYTHONUNBUFFERED=1`
- [ ] Service deployed successfully
- [ ] **Copied API service URL** (save this!)
- [ ] Health check passing

### Dashboard Service
- [ ] Created new App Runner service
- [ ] Connected GitHub repository
- [ ] Selected branch (main/master)
- [ ] Set configuration file: `aws-apprunner-dashboard.yaml`
- [ ] Set service name: `hydrogen-dashboard-eu-north-1`
- [ ] Set CPU: 0.5 vCPU
- [ ] Set Memory: 1 GB
- [ ] Set Port: 8050
- [ ] Added environment variables:
  - [ ] `PORT=8050`
  - [ ] `HYDROGEN_API_URL=<YOUR_API_URL>`
  - [ ] `PYTHONUNBUFFERED=1`
- [ ] Service deployed successfully
- [ ] **Copied Dashboard service URL** (save this!)
- [ ] Health check passing

### Post-Deployment (App Runner)
- [ ] API accessible at: `https://YOUR_API_URL`
- [ ] API health endpoint works: `https://YOUR_API_URL/health`
- [ ] Dashboard accessible at: `https://YOUR_DASHBOARD_URL`
- [ ] Dashboard connects to API successfully
- [ ] Tested full workflow in dashboard

---

## 🐳 AWS ECS Deployment

### ECR Setup
- [ ] ECR repositories created:
  - [ ] `hydrogen-api` in `eu-north-1`
  - [ ] `hydrogen-dashboard` in `eu-north-1`
- [ ] Logged into ECR successfully

### Docker Images
- [ ] Built API image: `docker build -f Dockerfile.aws-api -t hydrogen-api:latest .`
- [ ] Built Dashboard image: `docker build -f Dockerfile.aws-dashboard -t hydrogen-dashboard:latest .`
- [ ] Tagged API image with ECR URI
- [ ] Tagged Dashboard image with ECR URI
- [ ] Pushed API image to ECR
- [ ] Pushed Dashboard image to ECR

### CloudWatch Logs
- [ ] Created log group: `/ecs/hydrogen-api`
- [ ] Created log group: `/ecs/hydrogen-dashboard`

### IAM Roles
- [ ] Created `ecsTaskExecutionRole` (or verified exists)
- [ ] Attached `AmazonECSTaskExecutionRolePolicy`
- [ ] Created `ecsTaskRole` (if needed)

### Task Definitions
- [ ] Updated `aws-ecs-task-definition-api.json`:
  - [ ] Replaced `YOUR_ACCOUNT_ID` with actual account ID
  - [ ] Verified ECR image URI is correct
- [ ] Updated `aws-ecs-task-definition-dashboard.json`:
  - [ ] Replaced `YOUR_ACCOUNT_ID` with actual account ID
  - [ ] Replaced `REPLACE_WITH_API_URL` with API URL
  - [ ] Verified ECR image URI is correct
- [ ] Registered API task definition
- [ ] Registered Dashboard task definition

### ECS Cluster
- [ ] Created cluster: `hydrogen-cluster` in `eu-north-1`

### Networking
- [ ] Identified default VPC
- [ ] Identified subnet(s)
- [ ] Created security group: `hydrogen-api-sg`
  - [ ] Allowed inbound TCP port 8000 from 0.0.0.0/0
- [ ] Created security group: `hydrogen-dashboard-sg`
  - [ ] Allowed inbound TCP port 8050 from 0.0.0.0/0

### ECS Services
- [ ] Created API service:
  - [ ] Service name: `hydrogen-api`
  - [ ] Task definition: `hydrogen-api`
  - [ ] Desired count: 1
  - [ ] Launch type: FARGATE
  - [ ] Network configuration set correctly
  - [ ] Service running successfully
- [ ] Created Dashboard service:
  - [ ] Service name: `hydrogen-dashboard`
  - [ ] Task definition: `hydrogen-dashboard`
  - [ ] Desired count: 1
  - [ ] Launch type: FARGATE
  - [ ] Network configuration set correctly
  - [ ] Service running successfully

### Post-Deployment (ECS)
- [ ] Obtained API public IP/URL
- [ ] Obtained Dashboard public IP/URL
- [ ] API accessible
- [ ] API health endpoint works
- [ ] Dashboard accessible
- [ ] Dashboard connects to API successfully
- [ ] Tested full workflow in dashboard
- [ ] CloudWatch logs accessible
- [ ] Monitoring set up (optional)

---

## ✅ Final Verification

### Functionality Tests
- [ ] API `/health` endpoint returns 200
- [ ] API `/regions` endpoint returns data
- [ ] Dashboard loads without errors
- [ ] Dashboard can fetch data from API
- [ ] All dashboard features work correctly
- [ ] No console errors in browser

### Performance
- [ ] API responds within acceptable time (< 2 seconds)
- [ ] Dashboard loads within acceptable time (< 5 seconds)
- [ ] No memory leaks observed

### Security
- [ ] Security groups configured correctly
- [ ] Only necessary ports exposed
- [ ] Environment variables set securely
- [ ] No sensitive data in logs

### Monitoring (Optional)
- [ ] CloudWatch alarms configured
- [ ] Log retention policy set
- [ ] Cost monitoring enabled

---

## 📝 Notes

**API URL:** `_________________________________`

**Dashboard URL:** `_________________________________`

**AWS Account ID:** `_________________________________`

**Region:** `eu-north-1`

**Deployment Date:** `_________________________________`

---

## 🆘 Troubleshooting

If you encounter issues, check:
1. CloudWatch logs for errors
2. Security group rules
3. Environment variables
4. Task/service status in ECS console
5. Network connectivity

See `AWS_DEPLOYMENT_GUIDE.md` → Troubleshooting section for details.

---

**Status:** ⬜ Not Started | 🟡 In Progress | ✅ Complete
