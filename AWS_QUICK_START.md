# AWS Quick Start - EU North 1

Quick deployment guide for AWS App Runner (easiest method).

## 🚀 5-Minute Deployment

### Prerequisites
- AWS Account
- Code pushed to GitHub

### Step 1: Deploy API

1. Go to: https://eu-north-1.console.aws.amazon.com/apprunner
2. Click **"Create service"**
3. Select **"Source code repository"** → Connect GitHub → Select repo
4. Build config: **Use configuration file** → `aws-apprunner-api.yaml`
5. Service name: `hydrogen-api-eu-north-1`
6. CPU: **0.5 vCPU**, Memory: **1 GB**, Port: **8000**
7. Environment variables:
   ```
   HOST=0.0.0.0
   PORT=8000
   PYTHONUNBUFFERED=1
   ```
8. Click **"Create & deploy"**
9. **Copy the service URL** (wait 5-10 minutes)

### Step 2: Deploy Dashboard

1. Create another App Runner service
2. Same steps, but:
   - Service name: `hydrogen-dashboard-eu-north-1`
   - Config file: `aws-apprunner-dashboard.yaml`
   - Port: **8050**
   - Environment variables:
     ```
     PORT=8050
     HYDROGEN_API_URL=<YOUR_API_URL_FROM_STEP_1>
     PYTHONUNBUFFERED=1
     ```

### Step 3: Done! 🎉

- **API:** `https://YOUR_API_URL`
- **Dashboard:** `https://YOUR_DASHBOARD_URL`

---

For detailed instructions, see `AWS_DEPLOYMENT_GUIDE.md`
