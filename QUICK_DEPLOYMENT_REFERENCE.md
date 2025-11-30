# 🚀 Quick Deployment Reference

## Three-Tier Architecture Summary

```
User → Vercel (dashboard.py) → Render (api.py) → Cloud Run (ml_service.py)
```

## 📁 Files Created/Modified

### New Files:
- ✅ `ml_service.py` - ML service for Cloud Run
- ✅ `requirements-ml-service.txt` - ML service dependencies
- ✅ `Dockerfile.ml-service` - Dockerfile for Cloud Run
- ✅ `deploy-cloud-run.bat` - Cloud Run deployment script
- ✅ `THREE_TIER_DEPLOYMENT_GUIDE.md` - Complete deployment guide

### Modified Files:
- ✅ `api.py` - Added ML service integration
- ✅ `render.yaml` - Added ML_SERVICE_URL environment variable
- ✅ `vercel-smart-minimal.json` - Added HYDROGEN_API_URL

## 🚀 Quick Start

### 1. Deploy ML Service (Cloud Run)
```bash
# Update PROJECT_ID in deploy-cloud-run.bat
deploy-cloud-run.bat
# Copy the service URL
```

### 2. Deploy API (Render)
- Go to render.com
- Create Web Service manually
- Add ML_SERVICE_URL environment variable
- Deploy

### 3. Deploy Dashboard (Vercel)
- Go to vercel.com
- Import GitHub repository
- Add HYDROGEN_API_URL environment variable
- Deploy

## 📊 Environment Variables

| Service | Variable | Value |
|---------|----------|-------|
| **Render API** | `ML_SERVICE_URL` | Your Cloud Run URL |
| **Vercel Dashboard** | `HYDROGEN_API_URL` | Your Render API URL |

## ✅ Deployment Order

1. **Cloud Run** → Get ML service URL
2. **Render** → Get API URL, set ML_SERVICE_URL
3. **Vercel** → Set HYDROGEN_API_URL

## 🔗 Full Guide

See `THREE_TIER_DEPLOYMENT_GUIDE.md` for complete step-by-step instructions.
