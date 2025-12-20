# Docker Deployment Troubleshooting Guide

## Common Docker Build Issues and Solutions

### 1. Package Installation Failures

**Error**: `process "/bin/sh -c apt-get update && apt-get install -y ..." did not complete successfully: exit code: 100`

**Causes**:
- Network connectivity issues
- Package repository problems
- Base image issues
- Too many packages being installed

**Solutions**:

#### Option A: Use Minimal Dockerfile
```bash
# Use the minimal Dockerfile
docker build -f Dockerfile.minimal -t hydrogen-dashboard .
```

#### Option B: Use Fixed Dockerfile
```bash
# Use the fixed Dockerfile with retry logic
docker build -f Dockerfile.fixed -t hydrogen-dashboard .
```

#### Option C: Update Original Dockerfile
The original Dockerfile has been updated with:
- Retry logic for package installation
- Minimal package requirements
- Better error handling

### 2. Alternative Docker Builds

#### Minimal Build (Recommended)
```bash
# Use minimal requirements
cp requirements-docker-minimal.txt requirements.txt
docker build -f Dockerfile.minimal -t hydrogen-dashboard .
```

#### Alpine-based Build
```dockerfile
# Alternative: Use Alpine Linux (smaller, more reliable)
FROM python:3.11-alpine

# Install minimal dependencies
RUN apk add --no-cache curl

# Rest of the Dockerfile...
```

### 3. Network Issues

If you're behind a corporate firewall or have network issues:

```bash
# Use different DNS
docker build --dns=8.8.8.8 -t hydrogen-dashboard .

# Or use host network
docker build --network=host -t hydrogen-dashboard .
```

### 4. Memory Issues

If you're running out of memory during build:

```bash
# Increase Docker memory limit
# In Docker Desktop: Settings > Resources > Memory > 4GB+

# Or use build with less memory
docker build --memory=2g -t hydrogen-dashboard .
```

### 5. Platform-Specific Issues

#### Windows WSL2 Issues
```bash
# Ensure WSL2 is updated
wsl --update

# Restart Docker Desktop
```

#### macOS Issues
```bash
# Reset Docker Desktop
# Docker Desktop > Troubleshoot > Reset to factory defaults
```

### 6. Step-by-Step Debugging

#### Check Base Image
```bash
# Test if base image works
docker run --rm python:3.11-slim echo "Base image works"
```

#### Test Package Installation
```bash
# Test package installation in interactive mode
docker run -it python:3.11-slim bash
apt-get update
apt-get install -y curl
```

#### Build with Verbose Output
```bash
# Build with detailed output
docker build --progress=plain --no-cache -t hydrogen-dashboard .
```

### 7. Alternative Deployment Methods

#### Use Docker Compose (Recommended)
```bash
# Use docker-compose which handles networking better
docker-compose up --build
```

#### Use Pre-built Images
```bash
# Use official Python image with pre-installed packages
FROM python:3.11-slim-bullseye
```

### 8. Quick Fixes

#### Fix 1: Clean Docker Cache
```bash
docker system prune -a
docker build --no-cache -t hydrogen-dashboard .
```

#### Fix 2: Use Different Base Image
```dockerfile
# Try different Python base image
FROM python:3.11-bullseye
# or
FROM python:3.11-bookworm
```

#### Fix 3: Skip System Packages
```dockerfile
# Skip system package installation entirely
FROM python:3.11-slim

# Just install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt
```

### 9. Production Deployment

#### Use Multi-stage Build
```dockerfile
# Use the production Dockerfile
docker build -f Dockerfile.prod -t hydrogen-dashboard .
```

#### Use Docker Compose for Production
```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

### 10. Emergency Deployment

If Docker continues to fail, use Render.com instead:

1. **Go to [render.com](https://render.com)**
2. **Connect your GitHub repository**
3. **Use the `render.yaml` configuration**
4. **Deploy without Docker**

### 11. Testing Commands

#### Test API Service
```bash
# Test API container
docker run -p 8000:8000 hydrogen-dashboard python start_api.py
```

#### Test Dashboard Service
```bash
# Test Dashboard container
docker run -p 8050:8050 hydrogen-dashboard python dashboard.py
```

#### Test Full Stack
```bash
# Test with docker-compose
docker-compose up --build
```

### 12. Logs and Debugging

#### View Build Logs
```bash
# Build with detailed logs
docker build --progress=plain -t hydrogen-dashboard . 2>&1 | tee build.log
```

#### View Container Logs
```bash
# View running container logs
docker logs <container_id>
```

#### Interactive Debugging
```bash
# Run container in interactive mode
docker run -it hydrogen-dashboard bash
```

## Quick Solutions Summary

1. **Try minimal Dockerfile**: `docker build -f Dockerfile.minimal -t hydrogen-dashboard .`
2. **Use docker-compose**: `docker-compose up --build`
3. **Clean Docker cache**: `docker system prune -a`
4. **Use Render.com**: Skip Docker entirely
5. **Check network**: Ensure stable internet connection
6. **Update Docker**: Ensure latest Docker Desktop version
