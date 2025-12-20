# Docker Deployment Guide

This guide explains how to deploy the Hydrogen Simulation Dashboard using Docker.

## Quick Start

### Development Environment

1. **Build and run with docker-compose:**
   ```bash
   docker-compose up --build
   ```

2. **Access the services:**
   - Dashboard: http://localhost:8050
   - API: http://localhost:8000
   - API Health: http://localhost:8000/health

### Production Environment

1. **Build and run with production compose:**
   ```bash
   docker-compose -f docker-compose.prod.yml up --build -d
   ```

2. **Access the services:**
   - Dashboard: http://localhost:8050
   - API: http://localhost:8000
   - With Nginx: http://localhost (dashboard) and http://localhost/api/ (API)

## Docker Files Overview

### Dockerfile
- **Purpose**: Development and basic deployment
- **Features**: Single-stage build, includes both API and dashboard
- **Use case**: Quick testing and development

### Dockerfile.prod
- **Purpose**: Production deployment
- **Features**: Multi-stage build, non-root user, optimized layers
- **Use case**: Production environments

### docker-compose.yml
- **Purpose**: Development environment
- **Features**: Separate API and dashboard services
- **Use case**: Local development and testing

### docker-compose.prod.yml
- **Purpose**: Production environment
- **Features**: Resource limits, health checks, persistent volumes
- **Use case**: Production deployment

## Service Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │    │      API        │
│   (Port 8050)   │◄───┤   (Port 8000)   │
│                 │    │                 │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────────────────┘
                    │
            ┌─────────────────┐
            │     Nginx       │
            │  (Port 80/443)  │
            └─────────────────┘
```

## Environment Variables

### API Service
- `HYDROGEN_API_URL`: API base URL (default: http://127.0.0.1:8000)
- `PYTHONUNBUFFERED`: Python output buffering (default: 1)

### Dashboard Service
- `HYDROGEN_API_URL`: API base URL for dashboard to connect to
- `PORT`: Dashboard port (default: 8050)
- `PYTHONUNBUFFERED`: Python output buffering (default: 1)

## Volumes

The following volumes are mounted for data persistence:

- `pretrained-models/`: ML model files
- `dataset/`: Training and test datasets
- `processed_data/`: Processed data files

## Health Checks

Both services include health checks:

- **API**: `GET /health` endpoint
- **Dashboard**: `GET /` endpoint

## Resource Limits (Production)

### API Service
- Memory: 1-2GB
- CPU: 0.5-1.0 cores

### Dashboard Service
- Memory: 512MB-1GB
- CPU: 0.25-0.5 cores

## Deployment Commands

### Development
```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production
```bash
# Build and start production
docker-compose -f docker-compose.prod.yml up --build -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop services
docker-compose -f docker-compose.prod.yml down
```

### Individual Services
```bash
# Run only API
docker-compose up api

# Run only dashboard
docker-compose up dashboard
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8000 and 8050 are available
2. **Memory issues**: Increase Docker memory allocation
3. **Model loading errors**: Check that pretrained models are in the correct directory
4. **API connection errors**: Verify HYDROGEN_API_URL environment variable

### Debug Commands

```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs [service_name]

# Execute commands in container
docker-compose exec [service_name] bash

# Check resource usage
docker stats
```

### Logs

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs api
docker-compose logs dashboard

# Follow logs in real-time
docker-compose logs -f
```

## Security Considerations

1. **Non-root user**: Production Dockerfile runs as non-root user
2. **Resource limits**: Prevents resource exhaustion
3. **Health checks**: Ensures service availability
4. **Network isolation**: Services communicate through internal network

## Scaling

To scale the services:

```bash
# Scale dashboard (if needed)
docker-compose up --scale dashboard=2

# Scale with production compose
docker-compose -f docker-compose.prod.yml up --scale dashboard=2
```

## Monitoring

Monitor the services using:

1. **Health check endpoints**
2. **Docker stats**: `docker stats`
3. **Container logs**: `docker-compose logs`
4. **Resource usage**: Monitor CPU and memory usage

## Backup and Recovery

### Backup Data
```bash
# Backup volumes
docker run --rm -v hydrogen-dashboard_pretrained-models:/data -v $(pwd):/backup alpine tar czf /backup/pretrained-models-backup.tar.gz -C /data .
```

### Restore Data
```bash
# Restore volumes
docker run --rm -v hydrogen-dashboard_pretrained-models:/data -v $(pwd):/backup alpine tar xzf /backup/pretrained-models-backup.tar.gz -C /data
```
