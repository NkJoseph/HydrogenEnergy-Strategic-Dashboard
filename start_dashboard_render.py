#!/usr/bin/env python3
"""
Render.com startup script for Dashboard
Ensures proper host and port binding for Render deployment
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Setup environment for Render deployment"""
    print("Setting up Dashboard environment for Render...")
    
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

def start_dashboard():
    """Start Dashboard service with proper Render binding"""
    print("Starting Dashboard service for Render...")
    
    try:
        import dashboard
        
        # Get port from environment (Render sets this)
        port = int(os.environ.get("PORT", 8050))
        host = "0.0.0.0"  # Must bind to 0.0.0.0 for Render
        
        print(f"Starting Dashboard on {host}:{port}")
        print(f"Environment: PORT={port}, HOST={host}")
        
        # Start the dashboard with proper binding
        dashboard.app.run_server(
            host=host,
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
    print("Hydrogen Dashboard - Render Startup")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Environment variables:")
    for key, value in os.environ.items():
        if key.startswith(("PORT", "PYTHON", "HYDROGEN")):
            print(f"  {key}={value}")
    
    print("=" * 50)
    
    # Start dashboard
    start_dashboard()

if __name__ == "__main__":
    main()
