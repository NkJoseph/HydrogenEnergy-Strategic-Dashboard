# AWS Deployment Guide - EU North 1 Region

Complete guide for deploying the Hydrogen Simulation Dashboard to AWS EU North 1 (Stockholm) region.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start - AWS App Runner](#quick-start---aws-app-runner)
3. [Production Deployment - AWS ECS](#production-deployment---aws-ecs)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)
6. [Cost Estimates](#cost-estimates)

---

## Prerequisites

Before starting, ensure you have:

- [ ] AWS Account (sign up at https://aws.amazon.com)
- [ ] AWS CLI installed and configured (`aws configure`)
- [ ] Docker installed (for ECS deployment)
- [ ] Git repository with your code pushed to GitHub (for App Runner)
- [ ] Code tested locally and working

### Configure AWS CLI for EU North 1

```bash
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region: eu-north-1
# - Default output format: json
```

---

## Quick Start - AWS App Runner

**Best for:** Quick deployment, automatic scaling, minimal configuration

### Step 1: Prepare Your Code

1. **Push code to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for AWS EU North 1 deployment"
   git push origin main
   ```

2. **Verify files exist:**
   - `api.py`
   - `dashboard.py`
   - `start_api.py`
   - `requirements.txt`
   - `aws-apprunner-api.yaml`
   - `aws-apprunner-dashboard.yaml`

### Step 2: Deploy API Service

1. **Go to AWS App Runner Console:**
   - Navigate to: https://eu-north-1.console.aws.amazon.com/apprunner
   - Click **"Create service"**

2. **Source Configuration:**
   - Select **"Source code repository"**
   - Connect GitHub (if not already connected)
   - Select your repository
   - Branch: `main` or `master`
   - Deployment trigger: **Automatic**

3. **Build Settings:**
   - Build configuration: **Use a configuration file**
   - Configuration file: `aws-apprunner-api.yaml`
   - Runtime: **Python 3**

4. **Service Settings:**
   - Service name: `hydrogen-api-eu-north-1`
   - Virtual CPU: **0.5 vCPU**
   - Memory: **1 GB**
   - Port: **8000**
   - Environment variables:
     ```
     HOST=0.0.0.0
     PORT=8000
     PYTHONUNBUFFERED=1
     ```

5. **Create Service:**
   - Click **"Create & deploy"**
   - Wait 5-10 minutes for deployment
   - **Copy the service URL** (e.g., `https://xxxxx.eu-north-1.awsapprunner.com`)

### Step 3: Deploy Dashboard Service

1. **Create another App Runner service** for the dashboard

2. **Use same steps** but with these differences:
   - Service name: `hydrogen-dashboard-eu-north-1`
   - Configuration file: `aws-apprunner-dashboard.yaml`
   - Port: **8050**
   - Environment variables:
     ```
     PORT=8050
     HYDROGEN_API_URL=<YOUR_API_SERVICE_URL_FROM_STEP_2>
     PYTHONUNBUFFERED=1
     ```

3. **Update API URL:**
   - After dashboard service is created, go to **Configuration** → **Environment variables**
   - Update `HYDROGEN_API_URL` with your API service URL

### Step 4: Access Your Application

- **API:** `https://YOUR_API_SERVICE_URL`
- **Dashboard:** `https://YOUR_DASHBOARD_SERVICE_URL`

---

## Production Deployment - AWS ECS

**Best for:** Production deployments, more control, cost-effective for steady workloads

### Step 1: Install Prerequisites

```bash
# Verify AWS CLI
aws --version

# Verify Docker
docker --version
```

### Step 2: Configure AWS CLI for EU North 1

```bash
aws configure set region eu-north-1
```

### Step 3: Create ECR Repositories

```bash
# Set variables for EU North 1
export AWS_REGION=eu-north-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create repositories
aws ecr create-repository --repository-name hydrogen-api --region $AWS_REGION
aws ecr create-repository --repository-name hydrogen-dashboard --region $AWS_REGION

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

**Windows PowerShell:**
```powershell
$AWS_REGION = "eu-north-1"
$AWS_ACCOUNT_ID = (aws sts get-caller-identity --query Account --output text)

aws ecr create-repository --repository-name hydrogen-api --region $AWS_REGION
aws ecr create-repository --repository-name hydrogen-dashboard --region $AWS_REGION

aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
```

### Step 4: Build and Push Docker Images

```bash
# Build API image
docker build -f Dockerfile.aws-api -t hydrogen-api:latest .

# Build Dashboard image  
docker build -f Dockerfile.aws-dashboard -t hydrogen-dashboard:latest .

# Get repository URIs
API_REPO_URI=$(aws ecr describe-repositories --repository-names hydrogen-api --query "repositories[0].repositoryUri" --output text --region $AWS_REGION)
DASHBOARD_REPO_URI=$(aws ecr describe-repositories --repository-names hydrogen-dashboard --query "repositories[0].repositoryUri" --output text --region $AWS_REGION)

# Tag images
docker tag hydrogen-api:latest $API_REPO_URI:latest
docker tag hydrogen-dashboard:latest $DASHBOARD_REPO_URI:latest

# Push images
docker push $API_REPO_URI:latest
docker push $DASHBOARD_REPO_URI:latest
```

**Windows PowerShell:**
```powershell
docker build -f Dockerfile.aws-api -t hydrogen-api:latest .
docker build -f Dockerfile.aws-dashboard -t hydrogen-dashboard:latest .

$API_REPO_URI = (aws ecr describe-repositories --repository-names hydrogen-api --query "repositories[0].repositoryUri" --output text --region eu-north-1)
$DASHBOARD_REPO_URI = (aws ecr describe-repositories --repository-names hydrogen-dashboard --query "repositories[0].repositoryUri" --output text --region eu-north-1)

docker tag hydrogen-api:latest "$API_REPO_URI:latest"
docker tag hydrogen-dashboard:latest "$DASHBOARD_REPO_URI:latest"

docker push "$API_REPO_URI:latest"
docker push "$DASHBOARD_REPO_URI:latest"
```

### Step 5: Create CloudWatch Log Groups

```bash
aws logs create-log-group --log-group-name /ecs/hydrogen-api --region $AWS_REGION
aws logs create-log-group --log-group-name /ecs/hydrogen-dashboard --region $AWS_REGION
```

### Step 6: Create IAM Roles

**Task Execution Role:**

```bash
# Create trust policy file
cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document file://trust-policy.json

# Attach managed policy
aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

**Windows PowerShell:**
```powershell
@"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
"@ | Out-File -FilePath trust-policy.json -Encoding utf8

aws iam create-role --role-name ecsTaskExecutionRole --assume-role-policy-document file://trust-policy.json
aws iam attach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

### Step 7: Update Task Definitions

1. **Edit `aws-ecs-task-definition-api.json`:**
   - Replace `YOUR_ACCOUNT_ID` with your AWS account ID
   - The ECR repository URI is already set for `eu-north-1`

2. **Edit `aws-ecs-task-definition-dashboard.json`:**
   - Replace `YOUR_ACCOUNT_ID` with your AWS account ID
   - Replace `REPLACE_WITH_API_URL` with your API service URL (after creating it)

### Step 8: Register Task Definitions

```bash
aws ecs register-task-definition --cli-input-json file://aws-ecs-task-definition-api.json --region eu-north-1
aws ecs register-task-definition --cli-input-json file://aws-ecs-task-definition-dashboard.json --region eu-north-1
```

### Step 9: Create ECS Cluster

```bash
aws ecs create-cluster --cluster-name hydrogen-cluster --region eu-north-1
```

### Step 10: Create VPC and Networking

**Get default VPC:**
```bash
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region eu-north-1)
SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[0].SubnetId" --output text --region eu-north-1)
```

**Create Security Groups:**
```bash
# API Security Group
API_SG=$(aws ec2 create-security-group \
  --group-name hydrogen-api-sg \
  --description "Security group for Hydrogen API" \
  --vpc-id $VPC_ID \
  --query 'GroupId' --output text --region eu-north-1)

# Dashboard Security Group
DASHBOARD_SG=$(aws ec2 create-security-group \
  --group-name hydrogen-dashboard-sg \
  --description "Security group for Hydrogen Dashboard" \
  --vpc-id $VPC_ID \
  --query 'GroupId' --output text --region eu-north-1)

# Allow inbound traffic
aws ec2 authorize-security-group-ingress \
  --group-id $API_SG \
  --protocol tcp \
  --port 8000 \
  --cidr 0.0.0.0/0 \
  --region eu-north-1

aws ec2 authorize-security-group-ingress \
  --group-id $DASHBOARD_SG \
  --protocol tcp \
  --port 8050 \
  --cidr 0.0.0.0/0 \
  --region eu-north-1
```

### Step 11: Create ECS Services

```bash
# Create API service
aws ecs create-service \
  --cluster hydrogen-cluster \
  --service-name hydrogen-api \
  --task-definition hydrogen-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_ID],securityGroups=[$API_SG],assignPublicIp=ENABLED}" \
  --region eu-north-1

# Create Dashboard service
aws ecs create-service \
  --cluster hydrogen-cluster \
  --service-name hydrogen-dashboard \
  --task-definition hydrogen-dashboard \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_ID],securityGroups=[$DASHBOARD_SG],assignPublicIp=ENABLED}" \
  --region eu-north-1
```

### Step 12: Get Service URLs

```bash
# Get API public IP
aws ecs describe-services \
  --cluster hydrogen-cluster \
  --services hydrogen-api \
  --query "services[0].networkConfiguration.awsvpcConfiguration" \
  --region eu-north-1

# Check tasks
aws ecs list-tasks --cluster hydrogen-cluster --service-name hydrogen-api --region eu-north-1
```

---

## Configuration

### Environment Variables

**API Service:**
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `PYTHONUNBUFFERED`: Python output buffering (set to `1`)

**Dashboard Service:**
- `PORT`: Server port (default: `8050`)
- `HYDROGEN_API_URL`: URL of the API service (required)
- `PYTHONUNBUFFERED`: Python output buffering (set to `1`)

---

## Troubleshooting

### Common Issues

1. **Port binding errors:**
   - Ensure services bind to `0.0.0.0` not `127.0.0.1`
   - Check security groups allow inbound traffic

2. **Dashboard can't connect to API:**
   - Verify `HYDROGEN_API_URL` environment variable is set correctly
   - Check API is accessible from dashboard's network
   - Test API health endpoint: `curl https://YOUR_API_URL/health`

3. **Container fails to start:**
   - Check CloudWatch logs for errors
   - Verify Docker image is correct
   - Check environment variables

4. **Out of memory:**
   - Increase memory allocation in task definition
   - Check for memory leaks

### Debug Commands

```bash
# Check ECS services
aws ecs describe-services --cluster hydrogen-cluster --services hydrogen-api hydrogen-dashboard --region eu-north-1

# View task logs
aws logs tail /ecs/hydrogen-api --follow --region eu-north-1
aws logs tail /ecs/hydrogen-dashboard --follow --region eu-north-1

# Check service status
aws ecs describe-services --cluster hydrogen-cluster --services hydrogen-api --region eu-north-1

# Test API endpoint
curl https://YOUR_API_URL/health
```

---

## Cost Estimates

### App Runner (EU North 1)
- **API Service:** ~$0.007/vCPU-hour + $0.0008/GB-hour
- **Dashboard Service:** ~$0.007/vCPU-hour + $0.0008/GB-hour
- **Estimated monthly:** $10-30 (depending on traffic)

### ECS Fargate (EU North 1)
- **API Service:** ~$0.04/vCPU-hour + $0.004/GB-hour
- **Dashboard Service:** ~$0.04/vCPU-hour + $0.004/GB-hour
- **Estimated monthly:** $30-60 (for 24/7 operation)

### Free Tier
- **AWS Free Tier:** 12 months free for new accounts
- **EC2 t2.micro:** 750 hours/month free
- **ECS:** No free tier, but pay-as-you-go

---

## ✅ Deployment Checklist

### Pre-Deployment
- [ ] Code tested locally
- [ ] All dependencies in `requirements.txt`
- [ ] Environment variables documented
- [ ] AWS account created
- [ ] AWS CLI configured for `eu-north-1`

### App Runner Deployment
- [ ] Code pushed to GitHub
- [ ] API service created
- [ ] Dashboard service created
- [ ] Environment variables configured
- [ ] Services deployed successfully
- [ ] Health checks passing

### ECS Deployment
- [ ] ECR repositories created in `eu-north-1`
- [ ] Docker images built and pushed
- [ ] CloudWatch log groups created
- [ ] IAM roles created
- [ ] Task definitions registered
- [ ] ECS cluster created
- [ ] Security groups configured
- [ ] Services created and running
- [ ] Public IPs/URLs obtained

### Post-Deployment
- [ ] API accessible
- [ ] Dashboard accessible
- [ ] Dashboard connects to API
- [ ] Monitoring set up
- [ ] Alarms configured (optional)

---

## 🎯 Recommended Approach

**For beginners:** Use **AWS App Runner** (Option 1)
- Easier setup
- Automatic scaling
- Less configuration needed

**For production:** Use **AWS ECS** (Option 2)
- More control
- Better for steady workloads
- More cost-effective for 24/7 operation

---

## 📚 Additional Resources

- [AWS App Runner Documentation](https://docs.aws.amazon.com/apprunner/)
- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [AWS Pricing Calculator](https://calculator.aws/)
- [EU North 1 Region Information](https://aws.amazon.com/about-aws/global-infrastructure/regions_az/)

---

**Need help?** Check `AWS_DEPLOY_STEPS.md` for detailed step-by-step instructions.
