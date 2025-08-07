#!/usr/bin/env python3
"""
Simple script to start the API server
"""
import uvicorn
import api

if __name__ == "__main__":
    print("Starting API server on http://127.0.0.1:8000")
    uvicorn.run(api.app, host="127.0.0.1", port=8000, log_level="info") 