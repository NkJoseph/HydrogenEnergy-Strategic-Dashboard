# Render.com Free Tier Deployment Guide

## 🚨 Blueprint Payment Requirement Issue

**Problem**: `Your render.yaml services require payment information on file. Please enter your payment details to continue.`

**Cause**: Render.com's Blueprint feature requires payment information even for free tier services.

## ✅ Free Tier Solutions

### Option 1: Manual Service Creation (Recommended for Free Tier)

Since Blueprint requires payment info, create services manually:

#### Step 1: Create API Service
1. **Go to [render.com](https://render.com)**
2. **Click "New +" → "Web Service"**
3. **Connect GitHub repository**: `NkJoseph/HydrogenEnergy-Strategic-Dashboard`
4. **Configure**:
   - **Name**: `hydrogen-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`
   - **Plan**: `Starter` (Free)

#### Step 2: Create Dashboard Service
1. **Click "New +" → "Web Service"**
2. **Connect same GitHub repository**
3. **Configure**:
   - **Name**: `hydrogen-dashboard`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python start_dashboard_render.py`
   - **Health Check Path**: `/`
   - **Plan**: `Starter` (Free)
   - **Environment Variables**:
     - `HYDROGEN_API_URL`: `https://hydrogen-api.onrender.com`

### Option 2: Single Service Deployment (Free Tier Friendly)

Deploy only the dashboard with embedded API:

#### Create Single Service
1. **Go to [render.com](https://render.com)**
2. **Click "New +" → "Web Service"**
3. **Connect GitHub repository**
4. **Configure**:
   - **Name**: `hydrogen-dashboard`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python dashboard.py`
   - **Health Check Path**: `/`
   - **Plan**: `Starter` (Free)

### Option 3: Use Alternative Free Platforms

#### Heroku (Free Tier Available)
```bash
# Create Procfile for Heroku
echo "web: python dashboard.py" > Procfile
echo "api: uvicorn api:app --host 0.0.0.0 --port \$PORT" >> Procfile
```

#### Railway (Free Tier)
```bash
# Create railway.json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python dashboard.py",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

#### Vercel (Free Tier)
```bash
# Create vercel.json
{
  "builds": [
    {
      "src": "dashboard.py",
      "use": "@vercel/python"
    }
  ]
}
```

## 🔧 Free Tier Optimizations

### Minimal Requirements File
Use `requirements-render-free.txt` for smaller builds:

```bash
# Copy minimal requirements
cp requirements-render-free.txt requirements.txt
git add requirements.txt
git commit -m "Use minimal requirements for free tier"
git push origin production
```

### Optimized Startup Script
Use `start_dashboard_render.py` for better free tier performance.

## 📋 Step-by-Step Manual Deployment

### 1. Prepare Repository
```bash
# Ensure all files are committed
git add .
git commit -m "Prepare for free tier deployment"
git push origin production
```

### 2. Create API Service on Render
1. **Go to Render Dashboard**
2. **Click "New +" → "Web Service"**
3. **Connect GitHub**: `NkJoseph/HydrogenEnergy-Strategic-Dashboard`
4. **Settings**:
   - **Name**: `hydrogen-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - **Health Check**: `/health`
   - **Plan**: `Starter` (Free)

### 3. Create Dashboard Service
1. **Click "New +" → "Web Service"**
2. **Connect same repository**
3. **Settings**:
   - **Name**: `hydrogen-dashboard`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python start_dashboard_render.py`
   - **Health Check**: `/`
   - **Plan**: `Starter` (Free)
   - **Environment Variables**:
     - `HYDROGEN_API_URL`: `https://hydrogen-api.onrender.com`

### 4. Deploy and Test
1. **Wait for both services to deploy** (5-10 minutes)
2. **Test API**: `https://hydrogen-api.onrender.com/health`
3. **Test Dashboard**: `https://hydrogen-dashboard.onrender.com/`

## 🚨 Free Tier Limitations

### Render.com Free Tier
- **Sleep Mode**: Services sleep after 15 minutes of inactivity
- **Memory**: 512MB RAM limit
- **Build Time**: 90 minutes per month
- **Cold Starts**: 30-60 seconds to wake up

### Workarounds
1. **Use minimal dependencies** (`requirements-render-free.txt`)
2. **Optimize startup time** (use startup scripts)
3. **Consider single service** deployment
4. **Use external monitoring** to keep services awake

## 🔍 Troubleshooting Free Tier Issues

### Issue 1: Services Keep Sleeping
**Solution**: Use external monitoring service (UptimeRobot, etc.)

### Issue 2: Build Timeouts
**Solution**: Use minimal requirements file

### Issue 3: Memory Issues
**Solution**: Optimize dependencies, remove heavy packages

### Issue 4: Cold Start Delays
**Solution**: Use startup scripts for faster initialization

## 🎯 Recommended Free Tier Approach

### For Maximum Compatibility:
1. **Use manual service creation** (not Blueprint)
2. **Deploy API and Dashboard separately**
3. **Use minimal requirements**
4. **Set up external monitoring**

### For Simplest Deployment:
1. **Deploy only Dashboard** (with embedded API)
2. **Use single service**
3. **Accept sleep mode limitations**

## 📊 Service URLs After Deployment

- **API**: `https://hydrogen-api.onrender.com`
- **Dashboard**: `https://hydrogen-dashboard.onrender.com`
- **Health Check**: `https://hydrogen-api.onrender.com/health`

The free tier deployment should work without payment information! 🎉
