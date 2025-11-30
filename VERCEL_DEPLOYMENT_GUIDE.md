# 🚀 Vercel Deployment Guide

## Complete Guide for Deploying Hydrogen Dashboard to Vercel

### ✅ **Why Vercel?**
- ✅ **No size limits** (unlike Railway)
- ✅ **Works in China** without VPN
- ✅ **Free tier available**
- ✅ **Excellent performance**
- ✅ **Easy deployment**

### 📋 **Prerequisites**
1. **GitHub account** with your repository
2. **Vercel account** (sign up at [vercel.com](https://vercel.com))
3. **Your code** pushed to GitHub

## 🚀 **Step-by-Step Deployment**

### **Step 1: Prepare Your Repository**

#### **Option A: Use Smart-Minimal (Recommended)**
```bash
# Copy smart-minimal requirements
cp requirements-smart-minimal.txt requirements.txt

# Copy smart-minimal Vercel config
cp vercel-smart-minimal.json vercel.json

# Commit and push
git add requirements.txt vercel.json
git commit -m "Prepare for Vercel deployment"
git push origin production
```

#### **Option B: Use Current Setup**
```bash
# Your current vercel.json is ready
# Just ensure requirements.txt is optimized
```

### **Step 2: Sign Up for Vercel**

1. **Go to [vercel.com](https://vercel.com)**
2. **Click "Sign Up"**
3. **Sign up with GitHub** (recommended)
4. **Authorize Vercel** to access your repositories

### **Step 3: Deploy Your Project**

#### **Method 1: Import from GitHub (Recommended)**

1. **Go to Vercel Dashboard**
2. **Click "Add New..." → "Project"**
3. **Select your repository**: `NkJoseph/HydrogenEnergy-Strategic-Dashboard`
4. **Select branch**: `production`
5. **Configure Project**:
   - **Framework Preset**: Other
   - **Root Directory**: `./`
   - **Build Command**: Leave empty (Vercel auto-detects)
   - **Output Directory**: Leave empty
   - **Install Command**: `pip install -r requirements-smart-minimal.txt`
6. **Click "Deploy"**

#### **Method 2: Using Vercel CLI**

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy
vercel --prod
```

### **Step 4: Configure Environment Variables**

In Vercel Dashboard → Your Project → Settings → Environment Variables:

#### **For Dashboard:**
```
HYDROGEN_API_URL=https://your-api-url.vercel.app
PORT=8080
PYTHONUNBUFFERED=1
```

#### **For API (if deploying separately):**
```
PORT=8000
PYTHONUNBUFFERED=1
```

### **Step 5: Deploy Both Services**

#### **Option A: Single Deployment (Dashboard Only)**
- **Deploy dashboard** as main service
- **API embedded** in dashboard
- **Simpler setup**

#### **Option B: Separate Deployments (Recommended)**
1. **Deploy API first**:
   - Create new project: `hydrogen-api`
   - Use `api.py` as entry point
   - Deploy

2. **Deploy Dashboard**:
   - Create new project: `hydrogen-dashboard`
   - Use `dashboard.py` as entry point
   - Set `HYDROGEN_API_URL` to API URL
   - Deploy

## 📁 **Vercel Configuration Files**

### **vercel.json (Current)**
```json
{
  "builds": [
    {
      "src": "dashboard.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/dashboard.py"
    }
  ],
  "env": {
    "PYTHON_VERSION": "3.11"
  }
}
```

### **vercel-smart-minimal.json (Recommended)**
```json
{
  "version": 2,
  "builds": [
    {
      "src": "dashboard.py",
      "use": "@vercel/python"
    },
    {
      "src": "api.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "api.py"
    },
    {
      "src": "/(.*)",
      "dest": "dashboard.py"
    }
  ],
  "env": {
    "PYTHONUNBUFFERED": "1",
    "PYTHON_VERSION": "3.11"
  }
}
```

## 🔧 **Troubleshooting**

### **Issue 1: Build Fails**
**Solution**: Use `requirements-smart-minimal.txt`
```bash
cp requirements-smart-minimal.txt requirements.txt
git add requirements.txt
git commit -m "Use smart-minimal requirements"
git push origin production
```

### **Issue 2: File Too Large**
**Solution**: Use `.vercelignore` to exclude large files
```bash
# Create .vercelignore
echo "pretrained-models/" > .vercelignore
echo "processed_data/" >> .vercelignore
echo "*.pt" >> .vercelignore
echo "*.pkl" >> .vercelignore
```

### **Issue 3: API Not Connecting**
**Solution**: Check environment variables
```bash
# In Vercel Dashboard → Settings → Environment Variables
HYDROGEN_API_URL=https://your-api-url.vercel.app
```

### **Issue 4: Timeout Errors**
**Solution**: Increase function timeout
```json
{
  "functions": {
    "dashboard.py": {
      "maxDuration": 30
    }
  }
}
```

## 📊 **Vercel Free Tier Limits**

- **Bandwidth**: 100GB/month
- **Function Execution**: 100GB-hours/month
- **Build Time**: Unlimited
- **Deployments**: Unlimited
- **Team Members**: Unlimited

## 🎯 **Recommended Configuration**

### **For Your Hydrogen Dashboard:**

1. **Use `vercel-smart-minimal.json`**
2. **Use `requirements-smart-minimal.txt`**
3. **Deploy as single project** (dashboard + API)
4. **Set environment variables** in Vercel dashboard

### **Project Structure:**
```
hydrogen-dashboard/
├── dashboard.py (main entry)
├── api.py (API routes)
├── vercel.json (configuration)
└── requirements-smart-minimal.txt
```

## 🚀 **Quick Deploy Commands**

### **Using Vercel CLI:**
```bash
# Install CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel --prod

# Set environment variables
vercel env add HYDROGEN_API_URL
```

### **Using GitHub Integration:**
1. **Connect repository** in Vercel dashboard
2. **Auto-deploy** on every push
3. **Preview deployments** for pull requests

## 📈 **Post-Deployment**

### **Check Deployment:**
1. **Go to Vercel Dashboard**
2. **Click on your project**
3. **View deployment logs**
4. **Test your application**

### **Custom Domain:**
1. **Go to Settings → Domains**
2. **Add your domain**
3. **Configure DNS**
4. **SSL automatically enabled**

## 🎉 **Success!**

Your Hydrogen Dashboard is now deployed on Vercel!

**URL**: `https://your-project.vercel.app`

---

**Perfect for Vercel deployment!** All configuration files are ready! 🚀
