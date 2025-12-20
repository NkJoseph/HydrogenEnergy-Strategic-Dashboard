# AWS Deployment Steps - Complete Guide

This guide provides step-by-step instructions to deploy your Hydrogen Dashboard and API on AWS.

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] AWS Account (sign up at https://aws.amazon.com)
- [ ] AWS CLI installed and configured (`aws configure`)
- [ ] Docker installed (for ECS deployment)
- [ ] Git repository with your code pushed to GitHub (for App Runner)
- [ ] Code tested locally and working

---

## 🚀 Option 1: AWS App Runner (Easiest - Recommended)

**Best for:** Quick deployment, automatic scaling, minimal configuration

### Step 1: Prepare Your Code

1. **Push code to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for AWS deployment"
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
   - Navigate to: https://console.aws.amazon.com/apprunner
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
   - Service name: `hydrogen-api`
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
   - **Copy the service URL** (e.g., `https://xxxxx.us-east-1.awsapprunner.com`)

### Step 3: Deploy Dashboard Service

1. **Create another App Runner service** for the dashboard

2. **Use same steps** but with these differences:
   - Service name: `hydrogen-dashboard`
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

## 🐳 Option 2: AWS ECS with Fargate (Production Ready)

**Best for:** Production deployments, more control, cost-effective for steady workloads

### Step 1: Install Prerequisites

```bash
# Verify AWS CLI
aws --version

# Verify Docker
docker --version
```

### Step 2: Configure AWS CLI

```bash
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key  
# - Default region (e.g., us-east-1)
# - Default output format (json)
```

### Step 3: Create ECR Repositories

```bash
# Set variables
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create repositories
aws ecr create-repository --repository-name hydrogen-api --region $AWS_REGION
aws ecr create-repository --repository-name hydrogen-dashboard --region $AWS_REGION

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

**Windows PowerShell:**
```powershell
$AWS_REGION = "us-east-1"
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

$API_REPO_URI = (aws ecr describe-repositories --repository-names hydrogen-api --query "repositories[0].repositoryUri" --output text --region us-east-1)
$DASHBOARD_REPO_URI = (aws ecr describe-repositories --repository-names hydrogen-dashboard --query "repositories[0].repositoryUri" --output text --region us-east-1)

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

### Step 7: Update Task Definitions

1. **Edit `aws-ecs-task-definition-api.json`:**
   - Replace `YOUR_ACCOUNT_ID` with your AWS account ID
   - Replace `YOUR_ECR_REPO_URI` with your ECR repository URI

2. **Edit `aws-ecs-task-definition-dashboard.json`:**
   - Replace `YOUR_ACCOUNT_ID` with your AWS account ID
   - Replace `YOUR_ECR_REPO_URI` with your ECR repository URI
   - Replace `YOUR_API_ALB_DNS` with your API service URL (after creating it)

### Step 8: Register Task Definitions

```bash
aws ecs register-task-definition --cli-input-json file://aws-ecs-task-definition-api.json
aws ecs register-task-definition --cli-input-json file://aws-ecs-task-definition-dashboard.json
```

### Step 9: Create ECS Cluster

```bash
aws ecs create-cluster --cluster-name hydrogen-cluster
```

### Step 10: Create VPC and Networking

**Get default VPC:**
```bash
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)
SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[0].SubnetId" --output text)
```

**Create Security Groups:**
```bash
# API Security Group
API_SG=$(aws ec2 create-security-group \
  --group-name hydrogen-api-sg \
  --description "Security group for Hydrogen API" \
  --vpc-id $VPC_ID \
  --query 'GroupId' --output text)

# Dashboard Security Group
DASHBOARD_SG=$(aws ec2 create-security-group \
  --group-name hydrogen-dashboard-sg \
  --description "Security group for Hydrogen Dashboard" \
  --vpc-id $VPC_ID \
  --query 'GroupId' --output text)

# Allow inbound traffic
aws ec2 authorize-security-group-ingress \
  --group-id $API_SG \
  --protocol tcp \
  --port 8000 \
  --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id $DASHBOARD_SG \
  --protocol tcp \
  --port 8050 \
  --cidr 0.0.0.0/0
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
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_ID],securityGroups=[$API_SG],assignPublicIp=ENABLED}"

# Create Dashboard service
aws ecs create-service \
  --cluster hydrogen-cluster \
  --service-name hydrogen-dashboard \
  --task-definition hydrogen-dashboard \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_ID],securityGroups=[$DASHBOARD_SG],assignPublicIp=ENABLED}"
```

### Step 12: Get Service URLs

```bash
# Get API public IP
aws ecs describe-services \
  --cluster hydrogen-cluster \
  --services hydrogen-api \
  --query "services[0].networkConfiguration.awsvpcConfiguration"

# Check tasks
aws ecs list-tasks --cluster hydrogen-cluster --service-name hydrogen-api
```

---

## 📝 Quick Reference Commands

### Check Service Status

```bash
# ECS Services
aws ecs describe-services --cluster hydrogen-cluster --services hydrogen-api hydrogen-dashboard

# App Runner Services
aws apprunner list-services
aws apprunner describe-service --service-arn <SERVICE_ARN>
```

### View Logs

```bash
# CloudWatch Logs (ECS)
aws logs tail /ecs/hydrogen-api --follow
aws logs tail /ecs/hydrogen-dashboard --follow

# App Runner Logs
aws apprunner describe-service --service-arn <SERVICE_ARN> --query "Service.Status"
```

### Update Services

```bash
# ECS - Force new deployment
aws ecs update-service --cluster hydrogen-cluster --service hydrogen-api --force-new-deployment

# App Runner - Automatic on git push
```

---

## 🔧 Troubleshooting

### Common Issues

1. **Port binding errors:**
   - Ensure services bind to `0.0.0.0` not `127.0.0.1`
   - Check security groups allow inbound traffic

2. **Dashboard can't connect to API:**
   - Verify `HYDROGEN_API_URL` environment variable is set correctly
   - Check API is accessible from dashboard's network
   - Test API health endpoint

3. **Container fails to start:**
   - Check CloudWatch logs for errors
   - Verify Docker image is correct
   - Check environment variables

4. **Out of memory:**
   - Increase memory allocation in task definition
   - Check for memory leaks

### Debug Commands

```bash
# Check ECS tasks
aws ecs list-tasks --cluster hydrogen-cluster

# View task logs
aws logs tail /ecs/hydrogen-api --follow

# Check service status
aws ecs describe-services --cluster hydrogen-cluster --services hydrogen-api

# Test API endpoint
curl https://YOUR_API_URL/health
```

---

## 💰 Cost Estimates

### App Runner
- **API Service:** ~$0.007/vCPU-hour + $0.0008/GB-hour
- **Dashboard Service:** ~$0.007/vCPU-hour + $0.0008/GB-hour
- **Estimated monthly:** $10-30 (depending on traffic)

### ECS Fargate
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
- [ ] AWS CLI configured

### App Runner Deployment
- [ ] Code pushed to GitHub
- [ ] API service created
- [ ] Dashboard service created
- [ ] Environment variables configured
- [ ] Services deployed successfully
- [ ] Health checks passing

### ECS Deployment
- [ ] ECR repositories created
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

---

**Need help?** Check `AWS_DEPLOYMENT_CHECKLIST.md` for detailed checklist or AWS documentation.

