# 🚀 Three-Tier Deployment Guide

## Architecture Overview

```
User → Vercel (Dashboard) → Render (API) → Cloud Run (ML Model) → Response
```

### Components:
1. **Frontend (Dashboard)**: Deploy `dashboard.py` to Vercel (Free)
2. **Backend (API)**: Deploy `api.py` to Render (Free, spins down)
3. **ML Service**: Deploy `ml_service.py` to Google Cloud Run (Pay-per-use, min instances = 0)

---

## 📋 Prerequisites

### Required Accounts:
- ✅ **GitHub** - Your code repository
- ✅ **Vercel** - Free account for frontend
- ✅ **Render** - Free account for API
- ✅ **Google Cloud** - Free tier ($300 credit)

### Required Tools:
- ✅ **Docker** - For building ML service container
- ✅ **Google Cloud SDK** - For deploying to Cloud Run
- ✅ **Git** - For version control

---

## 🚀 Phase 1: Deploy ML Service to Google Cloud Run

### Step 1.1: Set Up Google Cloud

1. **Create Google Cloud Account**:
   - Go to [cloud.google.com](https://cloud.google.com)
   - Sign up (free tier includes $300 credit)
   - Create a new project (e.g., "hydrogen-dashboard")

2. **Install Google Cloud SDK**:
   ```bash
   # Windows: Download from:
   # https://cloud.google.com/sdk/docs/install-sdk
   ```

3. **Authenticate**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

4. **Enable Required APIs**:
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

### Step 1.2: Deploy ML Service

1. **Update `deploy-cloud-run.bat`**:
   - Open `deploy-cloud-run.bat`
   - Replace `your-gcp-project-id` with your actual Google Cloud project ID

2. **Run Deployment Script**:
   ```bash
   deploy-cloud-run.bat
   ```

3. **Get Service URL**:
   After deployment, the script will display your service URL.
   Copy it (format: `https://hydrogen-ml-service-XXXXX.run.app`)

### Step 1.3: Test ML Service

```bash
# Test health endpoint
curl https://your-ml-service.run.app/health

# Test prediction endpoint
curl -X POST https://your-ml-service.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"region": "Global", "scenario": "BAU", "years": [2025, 2026, 2027]}'
```

---

## 🚀 Phase 2: Deploy API to Render

### Step 2.1: Update Configuration

1. **Update `render.yaml`**:
   - Open `render.yaml`
   - Replace `https://hydrogen-ml-service-XXXXX.run.app` with your actual Cloud Run URL

2. **Commit Changes**:
   ```bash
   git add render.yaml
   git commit -m "Add ML_SERVICE_URL to Render configuration"
   git push origin production
   ```

### Step 2.2: Deploy to Render (Manual - Free Tier)

1. **Go to Render Dashboard**:
   - Visit [render.com](https://render.com)
   - Sign up/Login with GitHub

2. **Create Web Service**:
   - Click "New +" → "Web Service" (NOT Blueprint)
   - Connect GitHub repository: `NkJoseph/HydrogenEnergy-Strategic-Dashboard`
   - Select branch: `production`

3. **Configure Service**:
   - **Name**: `hydrogen-api`
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements-render-compatible.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Health Check Path**: `/health`
   - **Plan**: Starter (Free)

4. **Add Environment Variables**:
   - `ML_SERVICE_URL`: (your Cloud Run URL from Phase 1)
   - `PYTHONUNBUFFERED`: `1`

5. **Deploy**:
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)

6. **Get API URL**:
   - After deployment, note your API URL (format: `https://hydrogen-api.onrender.com`)

### Step 2.3: Test API

```bash
# Test health endpoint
curl https://your-api.onrender.com/health

# Test forecast endpoint
curl "https://your-api.onrender.com/forecast?region=Global&start=2025&end=2030&scenario=BAU"
```

---

## 🚀 Phase 3: Deploy Dashboard to Vercel

### Step 3.1: Update Configuration

1. **Update `vercel-smart-minimal.json`**:
   - Open `vercel-smart-minimal.json`
   - Replace `https://hydrogen-api.onrender.com` with your actual Render API URL

2. **Rename to `vercel.json`** (optional):
   ```bash
   copy vercel-smart-minimal.json vercel.json
   ```

3. **Commit Changes**:
   ```bash
   git add vercel-smart-minimal.json vercel.json
   git commit -m "Update Vercel config with API URL"
   git push origin production
   ```

### Step 3.2: Deploy to Vercel

1. **Go to Vercel Dashboard**:
   - Visit [vercel.com](https://vercel.com)
   - Sign up/Login with GitHub

2. **Import Project**:
   - Click "Add New..." → "Project"
   - Import GitHub repository: `NkJoseph/HydrogenEnergy-Strategic-Dashboard`
   - Select branch: `production`

3. **Configure Project**:
   - **Framework Preset**: Other
   - **Root Directory**: `./`
   - **Build Command**: Leave empty (Vercel auto-detects)
   - **Output Directory**: Leave empty
   - **Install Command**: `pip install -r requirements-smart-minimal.txt`

4. **Add Environment Variables**:
   - `HYDROGEN_API_URL`: (your Render API URL from Phase 2)
   - `PYTHONUNBUFFERED`: `1`

5. **Deploy**:
   - Click "Deploy"
   - Wait for deployment (2-5 minutes)

6. **Get Dashboard URL**:
   - After deployment, note your dashboard URL (format: `https://your-project.vercel.app`)

### Step 3.3: Test Dashboard

- Visit your Vercel URL
- Verify dashboard loads
- Test forecast functionality
- Check API connection

---

## ✅ Phase 4: Verify Complete System

### Test Flow:

1. **ML Service**:
   ```bash
   curl https://your-ml-service.run.app/health
   ```

2. **API**:
   ```bash
   curl https://your-api.onrender.com/health
   ```

3. **Dashboard**:
   - Visit: `https://your-project.vercel.app`
   - Test forecast functionality

### Expected Behavior:

- **Dashboard** → Calls API → API calls ML Service → Returns predictions
- **Fallback**: If ML service unavailable, API uses local models
- **Fallback**: If API unavailable, Dashboard shows error message

---

## 📊 Service URLs Reference

After deployment, you should have:

| Service | Platform | URL Format | Environment Variable |
|---------|----------|------------|---------------------|
| **ML Service** | Cloud Run | `https://hydrogen-ml-service-XXXXX.run.app` | None |
| **API** | Render | `https://hydrogen-api.onrender.com` | `ML_SERVICE_URL` |
| **Dashboard** | Vercel | `https://your-project.vercel.app` | `HYDROGEN_API_URL` |

---

## 🔧 Troubleshooting

### Issue 1: ML Service Not Responding
**Solution**:
- Check Cloud Run logs: `gcloud run services logs read hydrogen-ml-service --region us-central1`
- Verify service is deployed: `gcloud run services list`
- Test health endpoint: `curl https://your-ml-service.run.app/health`

### Issue 2: API Can't Reach ML Service
**Solution**:
- Verify `ML_SERVICE_URL` environment variable in Render dashboard
- Check API logs in Render dashboard
- Test ML service directly with curl

### Issue 3: Dashboard Can't Reach API
**Solution**:
- Verify `HYDROGEN_API_URL` environment variable in Vercel dashboard
- Check API is running (may be spun down on Render free tier)
- Wait 30-60 seconds for API to wake up

### Issue 4: Cold Start Delays
**Solution**:
- **Render API**: Normal (30-60 seconds) - acceptable for free tier
- **Cloud Run ML**: Normal (5-10 seconds) - acceptable with min instances = 0
- Consider upgrading to paid plans for faster cold starts

---

## 💰 Cost Estimate

### Free Tier:
- **Vercel**: Free (100GB bandwidth/month)
- **Render**: Free (spins down after 15 min inactivity)
- **Cloud Run**: ~$0.10 per 1000 requests (with min instances = 0)

### Estimated Monthly Cost:
- **Low traffic** (< 1000 requests/month): **$0-1**
- **Medium traffic** (10,000 requests/month): **$1-5**
- **High traffic** (100,000 requests/month): **$10-20**

---

## 🎯 Quick Deployment Checklist

### Phase 1: Cloud Run (ML Service)
- [ ] Create Google Cloud account
- [ ] Install Google Cloud SDK
- [ ] Enable Cloud Run API
- [ ] Update `deploy-cloud-run.bat` with project ID
- [ ] Run `deploy-cloud-run.bat`
- [ ] Copy ML service URL

### Phase 2: Render (API)
- [ ] Update `render.yaml` with ML service URL
- [ ] Commit and push changes
- [ ] Create Render account
- [ ] Create Web Service manually
- [ ] Add `ML_SERVICE_URL` environment variable
- [ ] Deploy and get API URL

### Phase 3: Vercel (Dashboard)
- [ ] Update `vercel-smart-minimal.json` with API URL
- [ ] Commit and push changes
- [ ] Create Vercel account
- [ ] Import GitHub repository
- [ ] Add `HYDROGEN_API_URL` environment variable
- [ ] Deploy and get dashboard URL

### Phase 4: Testing
- [ ] Test ML service health
- [ ] Test API health
- [ ] Test dashboard functionality
- [ ] Verify end-to-end flow

---

## 🎉 Success!

Your three-tier architecture is now deployed:
- ✅ **Frontend** on Vercel (always available)
- ✅ **API** on Render (spins down, but that's okay)
- ✅ **ML Service** on Cloud Run (pay-per-use, scales to zero)

**All services are working together!** 🚀
