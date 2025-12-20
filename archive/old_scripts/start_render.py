#!/usr/bin/env python3
"""
Render.com startup script for Hydrogen Dashboard
Handles both API and Dashboard services based on environment variables
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

def start_api():
    """Start the API server"""
    print("Starting API server...")
    try:
        import uvicorn
        import api
        
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run(
            api.app, 
            host="0.0.0.0", 
            port=port, 
            log_level="info",
            workers=1
        )
    except Exception as e:
        print(f"Error starting API: {e}")
        sys.exit(1)

def start_dashboard():
    """Start the Dashboard server"""
    print("Starting Dashboard server...")
    try:
        import dashboard
        
        port = int(os.environ.get("PORT", 8050))
        dashboard.app.run_server(
            host="0.0.0.0",
            port=port,
            debug=False
        )
    except Exception as e:
        print(f"Error starting Dashboard: {e}")
        sys.exit(1)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main startup function"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get service type from environment
    service_type = os.environ.get("SERVICE_TYPE", "api")
    
    print(f"Starting {service_type} service...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Create necessary directories
    Path("pretrained-models/global").mkdir(parents=True, exist_ok=True)
    Path("pretrained-models/regional").mkdir(parents=True, exist_ok=True)
    Path("processed_data").mkdir(parents=True, exist_ok=True)
    
    if service_type == "api":
        start_api()
    elif service_type == "dashboard":
        start_dashboard()
    else:
        print(f"Unknown service type: {service_type}")
        sys.exit(1)

if __name__ == "__main__":
    main()
