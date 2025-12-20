#!/usr/bin/env python3
"""
Render.com optimized startup script
Handles service detection and proper startup
"""

import os
import sys
import time
from pathlib import Path

def setup_environment():
    """Setup environment for Render deployment"""
    print("Setting up environment for Render...")
    
    # Create necessary directories
    directories = [
        "pretrained-models/global",
        "pretrained-models/regional", 
        "processed_data",
        "dataset"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Set environment variables
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

def start_api():
    """Start API service"""
    print("Starting API service...")
    try:
        import uvicorn
        import api
        
        port = int(os.environ.get("PORT", 8000))
        print(f"Starting API on port {port}")
        
        uvicorn.run(
            api.app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            workers=1,
            access_log=True
        )
    except Exception as e:
        print(f"Error starting API: {e}")
        sys.exit(1)

def start_dashboard():
    """Start Dashboard service"""
    print("Starting Dashboard service...")
    try:
        import dashboard
        
        port = int(os.environ.get("PORT", 8050))
        print(f"Starting Dashboard on port {port}")
        
        # Configure for production
        dashboard.app.run_server(
            host="0.0.0.0",
            port=port,
            debug=False,
            dev_tools_hot_reload=False,
            dev_tools_ui=False
        )
    except Exception as e:
        print(f"Error starting Dashboard: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    print("=" * 50)
    print("Hydrogen Simulation Dashboard - Render Startup")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Determine service type
    service_type = os.environ.get("SERVICE_TYPE", "api")
    
    print(f"Service type: {service_type}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Environment variables:")
    for key, value in os.environ.items():
        if key.startswith(("PORT", "PYTHON", "HYDROGEN")):
            print(f"  {key}={value}")
    
    print("=" * 50)
    
    # Start appropriate service
    if service_type == "api":
        start_api()
    elif service_type == "dashboard":
        start_dashboard()
    else:
        print(f"Unknown service type: {service_type}")
        print("Available types: api, dashboard")
        sys.exit(1)

if __name__ == "__main__":
    main()
