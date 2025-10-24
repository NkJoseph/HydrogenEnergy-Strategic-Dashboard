# Render.com Port Binding Fix

## 🚨 Issue: Port Scan Timeout

**Error**: `Port scan timeout reached, no open ports detected on 0.0.0.0`

**Cause**: Application not binding to the correct host/port that Render.com expects.

## ✅ Solutions Applied

### 1. Updated render.yaml
- ✅ Added proper startup commands
- ✅ Ensured host binding to `0.0.0.0`
- ✅ Added logging for debugging

### 2. Created Render-specific Startup Script
- ✅ `start_dashboard_render.py` - Ensures proper binding
- ✅ `test_render_binding.py` - Test script for verification

### 3. Fixed Configuration

#### API Service (render.yaml)
```yaml
startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info
```

#### Dashboard Service (render.yaml)
```yaml
startCommand: python start_dashboard_render.py
```

## 🔧 Manual Fixes

### Option 1: Use Updated render.yaml
The `render.yaml` has been updated with proper startup commands.

### Option 2: Manual Service Configuration
If using manual service creation on Render.com:

#### API Service:
- **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1`
- **Environment Variables**: 
  - `PORT=8000`
  - `PYTHONUNBUFFERED=1`

#### Dashboard Service:
- **Start Command**: `python start_dashboard_render.py`
- **Environment Variables**:
  - `PORT=8050`
  - `PYTHONUNBUFFERED=1`
  - `HYDROGEN_API_URL=https://your-api-url.onrender.com`

### Option 3: Test Locally First
```bash
# Test API binding
PORT=8000 python -c "import api; import uvicorn; uvicorn.run(api.app, host='0.0.0.0', port=8000)"

# Test Dashboard binding
PORT=8050 python start_dashboard_render.py
```

## 🚀 Deployment Steps

### 1. Push Updated Code
```bash
git add .
git commit -m "Fix Render.com port binding issues"
git push origin production
```

### 2. Redeploy on Render.com
- Go to your Render.com dashboard
- Click "Manual Deploy" on both services
- Or delete and recreate services with updated configuration

### 3. Verify Deployment
- **API**: `https://your-api-url.onrender.com/health`
- **Dashboard**: `https://your-dashboard-url.onrender.com/`

## 🔍 Troubleshooting

### Check Service Logs
1. Go to Render.com dashboard
2. Click on your service
3. Go to "Logs" tab
4. Look for binding errors

### Common Issues

#### Issue 1: Still binding to localhost
**Solution**: Ensure `host="0.0.0.0"` in startup commands

#### Issue 2: Port not being read from environment
**Solution**: Use `$PORT` variable in start commands

#### Issue 3: Application not starting
**Solution**: Check Python dependencies and imports

### Debug Commands

#### Test Port Binding
```python
import os
print(f"PORT environment: {os.environ.get('PORT')}")
print(f"Host should be: 0.0.0.0")
```

#### Test Application Start
```bash
# Test API
python -c "import api; print('API imports successfully')"

# Test Dashboard
python -c "import dashboard; print('Dashboard imports successfully')"
```

## 📋 Render.com Configuration Checklist

### ✅ API Service
- [ ] Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1`
- [ ] Environment: `PORT=8000`
- [ ] Health Check: `/health`
- [ ] Python Version: 3.11

### ✅ Dashboard Service
- [ ] Start Command: `python start_dashboard_render.py`
- [ ] Environment: `PORT=8050`
- [ ] Health Check: `/`
- [ ] Python Version: 3.11
- [ ] Depends On: API Service

## 🎯 Quick Fix Summary

1. **Updated render.yaml** with proper startup commands
2. **Created start_dashboard_render.py** for proper binding
3. **Added test script** for verification
4. **Pushed to GitHub** - ready for redeployment

The port binding issue should now be resolved! 🎉
