# Render.com Blueprint Deployment Guide

## 🚨 Blueprint Error Fix

**Error**: `dependsOn: field dependsOn not found in type file.Service`

**Solution**: The `dependsOn` field is not supported in Render Blueprint. Use the updated `render.yaml` without this field.

## ✅ Fixed Configuration

### Updated render.yaml
- ✅ Removed unsupported `dependsOn` field
- ✅ Simplified configuration for Blueprint compatibility
- ✅ Maintained proper startup commands

## 🚀 Deployment Options

### Option 1: Use Updated render.yaml (Recommended)
The `render.yaml` has been fixed and should now work with Blueprint.

### Option 2: Use Simplified Configuration
Use `render-simple.yaml` for a minimal configuration.

### Option 3: Manual Service Creation
Create services manually without Blueprint.

## 📋 Step-by-Step Blueprint Deployment

### 1. Prepare Repository
```bash
# Ensure all files are committed
git add .
git commit -m "Fix render.yaml for Blueprint compatibility"
git push origin production
```

### 2. Create Blueprint on Render.com
1. **Go to [render.com](https://render.com)**
2. **Click "New +" → "Blueprint"**
3. **Connect GitHub repository**: `NkJoseph/HydrogenEnergy-Strategic-Dashboard`
4. **Select branch**: `production`
5. **Render will detect `render.yaml`**

### 3. Review Configuration
Render will show you the services to be created:
- **hydrogen-api** (Backend API)
- **hydrogen-dashboard** (Frontend Dashboard)

### 4. Deploy Services
1. **Click "Apply"** to create services
2. **Wait for deployment** (5-10 minutes)
3. **Check service URLs** in Render dashboard

## 🔧 Manual Service Creation (Alternative)

If Blueprint continues to have issues, create services manually:

### API Service
1. **Go to Render Dashboard**
2. **Click "New +" → "Web Service"**
3. **Connect GitHub repository**
4. **Configure**:
   - **Name**: `hydrogen-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`

### Dashboard Service
1. **Click "New +" → "Web Service"**
2. **Connect same GitHub repository**
3. **Configure**:
   - **Name**: `hydrogen-dashboard`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python start_dashboard_render.py`
   - **Health Check Path**: `/`
   - **Environment Variables**:
     - `HYDROGEN_API_URL`: `https://hydrogen-api.onrender.com`

## 🔍 Troubleshooting Blueprint Issues

### Common Blueprint Errors

#### Error 1: `dependsOn not found`
**Solution**: Use updated `render.yaml` without `dependsOn` field

#### Error 2: `Invalid service configuration`
**Solution**: Use `render-simple.yaml` for minimal configuration

#### Error 3: `Build failed`
**Solution**: Check Python dependencies in `requirements.txt`

### Debug Steps

#### 1. Validate render.yaml
```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('render.yaml'))"
```

#### 2. Test Locally
```bash
# Test API
python -c "import api; print('API imports successfully')"

# Test Dashboard
python -c "import dashboard; print('Dashboard imports successfully')"
```

#### 3. Check Render Logs
1. Go to Render Dashboard
2. Click on service
3. Go to "Logs" tab
4. Look for error messages

## 📊 Service Configuration Summary

### API Service
- **URL**: `https://hydrogen-api.onrender.com`
- **Health Check**: `/health`
- **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`

### Dashboard Service
- **URL**: `https://hydrogen-dashboard.onrender.com`
- **Health Check**: `/`
- **Start Command**: `python start_dashboard_render.py`

## 🎯 Quick Fix Summary

1. ✅ **Removed `dependsOn` field** from render.yaml
2. ✅ **Created simplified configuration** (render-simple.yaml)
3. ✅ **Updated startup commands** for proper binding
4. ✅ **Added troubleshooting guide**

The Blueprint deployment should now work correctly! 🎉
