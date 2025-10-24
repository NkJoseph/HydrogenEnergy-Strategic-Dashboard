#!/usr/bin/env python3
"""
Test script to verify Render.com port binding
Run this to test if the application binds correctly
"""

import os
import sys
from flask import Flask

def test_binding():
    """Test if the application can bind to the correct host and port"""
    app = Flask(__name__)
    
    @app.route('/')
    def health():
        return {"status": "healthy", "message": "Render binding test successful"}
    
    @app.route('/health')
    def health_check():
        return {"status": "healthy", "port": os.environ.get("PORT", "unknown")}
    
    # Get port from environment
    port = int(os.environ.get("PORT", 8050))
    host = "0.0.0.0"
    
    print(f"Testing binding to {host}:{port}")
    print(f"Environment PORT: {os.environ.get('PORT')}")
    print(f"Starting test server...")
    
    try:
        app.run(host=host, port=port, debug=False)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_binding()
