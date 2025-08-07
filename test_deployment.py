#!/usr/bin/env python3
"""
Test script for deployment verification
"""
import requests
import time
import sys

def test_local_deployment():
    """Test the local deployment"""
    print("🧪 Testing local deployment...")
    
    try:
        # Test API health endpoint
        response = requests.get("http://localhost:8000/ping", timeout=5)
        if response.status_code == 200:
            print("✅ API health check passed")
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
            
        # Test dashboard accessibility
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard accessible")
        else:
            print(f"❌ Dashboard not accessible: {response.status_code}")
            return False
            
        print("🎉 Local deployment test passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to localhost:8000")
        print("   Make sure the app is running with: python main.py")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_remote_deployment(url):
    """Test the remote deployment"""
    print(f"🧪 Testing remote deployment at {url}...")
    
    try:
        # Test API health endpoint
        response = requests.get(f"{url}/ping", timeout=10)
        if response.status_code == 200:
            print("✅ Remote API health check passed")
        else:
            print(f"❌ Remote API health check failed: {response.status_code}")
            return False
            
        # Test dashboard accessibility
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print("✅ Remote dashboard accessible")
        else:
            print(f"❌ Remote dashboard not accessible: {response.status_code}")
            return False
            
        print("🎉 Remote deployment test passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to remote URL")
        return False
    except Exception as e:
        print(f"❌ Remote test failed with error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test remote deployment
        remote_url = sys.argv[1]
        test_remote_deployment(remote_url)
    else:
        # Test local deployment
        test_local_deployment() 