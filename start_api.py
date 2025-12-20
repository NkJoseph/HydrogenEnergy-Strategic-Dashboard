#!/usr/bin/env python3
"""
Simple script to start the API server
"""
import os
import uvicorn
import api

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    print(f"Starting API server on http://{host}:{port}")
    uvicorn.run(api.app, host=host, port=port, log_level="info") 