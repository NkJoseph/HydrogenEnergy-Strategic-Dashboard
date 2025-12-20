# Hydrogen Dashboard Deployment Guide

This guide covers multiple free deployment options for your hydrogen dashboard project.

## 🚀 Quick Deploy Options

### 1. **Render.com** (Recommended)

**Pros:**
- Free tier: 750 hours/month
- Easy Git-based deployment
- Automatic HTTPS
- Perfect for FastAPI + Dash

**Steps:**
1. Push your code to GitHub
2. Go to [render.com](https://render.com) and sign up
3. Click "New +" → "Blueprint"
4. Connect your GitHub repository
5. Render will automatically detect the `render.yaml` file
6. Deploy both services (API + Dashboard)

**URLs after deployment:**
- API: `https://hydrogen-api.onrender.com`
- Dashboard: `https://hydrogen-dashboard.onrender.com`

### 2. **Railway.app**

**Pros:**
- Free tier: $5 credit monthly
- Simple deployment
- Good for full-stack apps

**Steps:**
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repo
3. Deploy as a new service
4. Set environment variables:
   - `PORT`: 8000
   - `HYDROGEN_API_URL`: Your API URL

### 3. **Heroku** (Alternative)

**Pros:**
- Established platform
- Good documentation

**Steps:**
1. Install Heroku CLI
2. Run: `heroku create your-app-name`
3. Run: `git push heroku main`
4. Set environment variables in Heroku dashboard

## 🔧 Environment Variables

Set these in your deployment platform:

```bash
PORT=8000
HYDROGEN_API_URL=https://your-api-url.com
DEBUG=False
```

## 📁 Project Structure

```
Dashboard/
├── api.py              # FastAPI backend
├── dashboard.py        # Dash frontend
├── requirements.txt    # Python dependencies
├── render.yaml         # Render.com config
├── Procfile           # Heroku config
└── pretrained-models/ # ML models
```

## 🐛 Troubleshooting

### Common Issues:

1. **Port binding error:**
   - Ensure your app binds to `0.0.0.0` and uses `$PORT`

2. **API connection error:**
   - Check `HYDROGEN_API_URL` environment variable
   - Verify API service is running

3. **Model loading error:**
   - Ensure `pretrained-models/` directory is included
   - Check file permissions

### Debug Commands:

```bash
# Check if API is running
curl https://your-api-url.com/

# Check environment variables
echo $PORT
echo $HYDROGEN_API_URL
```

## 🔄 Continuous Deployment

All platforms support automatic deployment:
- Push to `main` branch
- Platform automatically rebuilds and deploys
- No manual intervention needed

## 📊 Monitoring

- **Render:** Built-in logs and metrics
- **Railway:** Real-time logs
- **Heroku:** Application logs via CLI

## 💰 Cost Optimization

- **Render:** Free tier sufficient for development
- **Railway:** $5 credit covers small projects
- **Heroku:** Basic dynos available

## 🚀 Next Steps

1. Choose your preferred platform
2. Follow the deployment steps
3. Test your application
4. Share your live URL!

---

**Need help?** Check the platform-specific documentation or create an issue in your repository. 