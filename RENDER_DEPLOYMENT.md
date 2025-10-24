# Render.com Deployment Guide

This guide explains how to deploy the Hydrogen Simulation Dashboard on Render.com.

## 🚀 Quick Deployment

### Option 1: Using render.yaml (Recommended)

1. **Push your code to GitHub**
2. **Connect to Render.com**
3. **Deploy using render.yaml**

### Option 2: Manual Service Creation

Create two separate web services on Render.com.

## 📋 Prerequisites

- GitHub repository with your code
- Render.com account
- Python 3.11

## 🔧 Configuration Files

### render.yaml
```yaml
services:
  - type: web
    name: hydrogen-api
    env: python
    buildCommand: |
      pip install --upgrade pip
      pip install -r requirements.txt
    startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: PORT
        value: 8000
    healthCheckPath: /health
    plan: starter
    region: oregon

  - type: web
    name: hydrogen-dashboard
    env: python
    buildCommand: |
      pip install --upgrade pip
      pip install -r requirements.txt
    startCommand: python dashboard.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: PORT
        value: 8050
      - key: HYDROGEN_API_URL
        value: https://hydrogen-api.onrender.com
    healthCheckPath: /
    plan: starter
    region: oregon
    dependsOn:
      - hydrogen-api
```

## 🛠️ Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Ensure all files are committed:**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Verify these files exist:**
   - `render.yaml`
   - `requirements.txt`
   - `api.py`
   - `dashboard.py`
   - `runtime.txt`

### Step 2: Create Render Account

1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Connect your GitHub account

### Step 3: Deploy Services

#### Method A: Using render.yaml (Automatic)

1. **Create New Blueprint:**
   - Go to Render Dashboard
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Select the repository
   - Render will automatically detect `render.yaml`

2. **Configure Services:**
   - Review the services configuration
   - Update environment variables if needed
   - Click "Apply" to deploy

#### Method B: Manual Service Creation

1. **Deploy API Service:**
   - Go to Render Dashboard
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `hydrogen-api`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1`
     - **Health Check Path**: `/health`

2. **Deploy Dashboard Service:**
   - Click "New +" → "Web Service"
   - Connect the same GitHub repository
   - Configure:
     - **Name**: `hydrogen-dashboard`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `python dashboard.py`
     - **Health Check Path**: `/`
     - **Environment Variables**:
       - `HYDROGEN_API_URL`: `https://hydrogen-api.onrender.com`

### Step 4: Environment Variables

Set these environment variables in Render dashboard:

#### API Service:
```
PYTHON_VERSION=3.11.0
PORT=8000
PYTHONUNBUFFERED=1
```

#### Dashboard Service:
```
PYTHON_VERSION=3.11.0
PORT=8050
PYTHONUNBUFFERED=1
HYDROGEN_API_URL=https://hydrogen-api.onrender.com
```

## 🔍 Service Configuration Details

### API Service Configuration

- **Runtime**: Python 3.11
- **Build Command**: 
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- **Start Command**: 
  ```bash
  uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1
  ```
- **Health Check**: `/health`
- **Plan**: Starter (Free tier)

### Dashboard Service Configuration

- **Runtime**: Python 3.11
- **Build Command**: 
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- **Start Command**: 
  ```bash
  python dashboard.py
  ```
- **Health Check**: `/`
- **Plan**: Starter (Free tier)

## 📊 Monitoring and Logs

### View Logs
1. Go to your service in Render Dashboard
2. Click on "Logs" tab
3. Monitor real-time logs

### Health Checks
- **API**: `https://your-api-url.onrender.com/health`
- **Dashboard**: `https://your-dashboard-url.onrender.com/`

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**
   - Check Python version compatibility
   - Verify all dependencies in requirements.txt
   - Check build logs for specific errors

2. **Service Not Starting**
   - Verify start commands
   - Check environment variables
   - Review service logs

3. **API Connection Issues**
   - Ensure `HYDROGEN_API_URL` is correct
   - Check API service is running
   - Verify network connectivity

4. **Memory Issues**
   - Upgrade to paid plan for more memory
   - Optimize dependencies
   - Use lighter PyTorch version

### Debug Commands

```bash
# Check service status
curl https://your-api-url.onrender.com/health

# Test API endpoints
curl https://your-api-url.onrender.com/

# Test dashboard
curl https://your-dashboard-url.onrender.com/
```

## 💰 Pricing Considerations

### Free Tier Limitations
- **Sleep Mode**: Services sleep after 15 minutes of inactivity
- **Memory**: 512MB RAM limit
- **CPU**: Limited CPU resources
- **Build Time**: 90 minutes per month

### Paid Plans
- **Starter**: $7/month per service
- **Standard**: $25/month per service
- **Pro**: $85/month per service

## 🔄 Continuous Deployment

### Automatic Deployments
1. Connect GitHub repository
2. Enable auto-deploy on push
3. Services will rebuild automatically on code changes

### Manual Deployments
1. Go to service dashboard
2. Click "Manual Deploy"
3. Select branch/commit to deploy

## 📈 Performance Optimization

### For Free Tier
1. **Optimize Dependencies**:
   ```python
   # Use CPU-only PyTorch
   torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Reduce Memory Usage**:
   - Remove unused dependencies
   - Use lighter ML libraries
   - Optimize model loading

3. **Faster Cold Starts**:
   - Minimize startup time
   - Use efficient imports
   - Cache model loading

### For Paid Plans
1. **Enable Auto-scaling**
2. **Use CDN for static assets**
3. **Implement caching strategies**

## 🔐 Security Considerations

1. **Environment Variables**:
   - Never commit secrets to repository
   - Use Render's environment variable system
   - Rotate API keys regularly

2. **Network Security**:
   - Use HTTPS endpoints
   - Implement rate limiting
   - Add authentication if needed

## 📝 Custom Domain (Paid Plans)

1. **Add Custom Domain**:
   - Go to service settings
   - Add custom domain
   - Configure DNS records

2. **SSL Certificate**:
   - Automatically provided by Render
   - Automatic renewal

## 🔄 Backup and Recovery

### Data Persistence
- Render provides persistent disks for paid plans
- Free tier: No persistent storage
- Consider external storage for production

### Backup Strategy
1. **Code**: GitHub repository
2. **Data**: External storage (AWS S3, etc.)
3. **Models**: Version control or external storage

## 📞 Support

### Render Support
- **Documentation**: [render.com/docs](https://render.com/docs)
- **Community**: [render.com/community](https://render.com/community)
- **Support**: Available for paid plans

### Common Resources
- [Render Python Guide](https://render.com/docs/deploy-flask)
- [Environment Variables](https://render.com/docs/environment-variables)
- [Health Checks](https://render.com/docs/health-checks)
