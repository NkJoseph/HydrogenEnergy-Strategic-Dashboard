"""
Test script to verify all API endpoints work correctly with pretrained models
"""
import requests
import json
import time
import sys
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_endpoint(endpoint, method="GET", data=None, description=""):
    """Test a single endpoint and return result"""
    try:
        url = f"{API_BASE}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            return False, f"Unknown method: {method}"
        
        if response.status_code == 200:
            result = response.json()
            return True, result
        else:
            return False, f"HTTP {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {e}"

def main():
    print("🧪 Testing Hydrogen Forecasting API Endpoints")
    print("=" * 50)
    
    # Test basic endpoints
    endpoints_to_test = [
        ("/", "GET", None, "Root endpoint"),
        ("/available_regions", "GET", None, "Available regions"),
        ("/scenarios", "GET", None, "Available scenarios"),
        ("/models", "GET", None, "Available models"),
    ]
    
    for endpoint, method, data, description in endpoints_to_test:
        print(f"\n📍 Testing: {description}")
        success, result = test_endpoint(endpoint, method, data, description)
        if success:
            print(f"✅ SUCCESS: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ FAILED: {result}")
    
    # Test forecast endpoint with different parameters
    print(f"\n📍 Testing: Global forecast")
    success, result = test_endpoint("/forecast?region=Global&start=2025&end=2030&scenario=BAU")
    if success:
        years = result.get('years', []) if isinstance(result, dict) else []
        print(f"✅ SUCCESS: Got {len(years)} years of data")
        print(f"   Sample data: {result}")
    else:
        print(f"❌ FAILED: {result}")
    
    # Test regional forecast
    print(f"\n📍 Testing: Regional forecast (Europe)")
    success, result = test_endpoint("/forecast?region=Europe&start=2025&end=2027&scenario=Optimistic")
    if success:
        years = result.get('years', []) if isinstance(result, dict) else []
        print(f"✅ SUCCESS: Got {len(years)} years of data")
        print(f"   Sample data: {result}")
    else:
        print(f"❌ FAILED: {result}")
    
    # Test optimization endpoint
    print(f"\n📍 Testing: Optimization endpoint")
    opt_data = {
        "region": "Global",
        "start": 2025,
        "end": 2030,
        "target": 430.0,
        "scenario": "BAU",
        "curve_dynamic": "early",
        "return_ci": True
    }
    success, result = test_endpoint("/optimize", "POST", opt_data)
    if success:
        print(f"✅ SUCCESS: Got optimization result with CI")
        if isinstance(result, dict):
            cumulative = result.get('cumulative', [])
            final_value = cumulative[-1] if cumulative else 'N/A'
            print(f"   Total cumulative: {final_value}")
        else:
            print(f"   Result: {result}")
    else:
        print(f"❌ FAILED: {result}")
    
    # Test map endpoint
    print(f"\n📍 Testing: Map endpoint")
    success, result = test_endpoint("/map?year=2025&scenario=BAU&strategic=false")
    if success:
        if isinstance(result, list):
            print(f"✅ SUCCESS: Got map data for {len(result)} regions")
            sample_regions = [r.get('region', 'unknown') for r in result[:3] if isinstance(r, dict)]
            print(f"   Sample regions: {sample_regions}")
        else:
            print(f"✅ SUCCESS: {result}")
    else:
        print(f"❌ FAILED: {result}")
    
    # Test strategic map
    print(f"\n📍 Testing: Strategic map endpoint")
    success, result = test_endpoint("/map?year=2025&scenario=BAU&strategic=true")
    if success:
        if isinstance(result, list):
            print(f"✅ SUCCESS: Got strategic map data for {len(result)} regions")
            sample_values = [r.get('value', 0) for r in result[:3] if isinstance(r, dict)]
            print(f"   Sample values: {sample_values}")
        else:
            print(f"✅ SUCCESS: {result}")
    else:
        print(f"❌ FAILED: {result}")
    
    print(f"\n📍 Testing: Model file verification")
    # Check if model files exist and are properly sized
    model_dir = Path("pretrained-models")
    global_models = list((model_dir / "global").glob("*.pt"))
    regional_models = list((model_dir / "regional").glob("*.pt"))
    
    print(f"✅ Found {len(global_models)} global models")
    print(f"✅ Found {len(regional_models)} regional models")
    
    # Check some specific model files
    key_files = [
        "pretrained-models/global/transformer.pt",
        "pretrained-models/regional/Europe_transformer.pt",
        "pretrained-models/regional/USA_autoformer.pt"
    ]
    
    for file_path in key_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {file_path} exists ({size} bytes)")
        else:
            print(f"❌ {file_path} missing")
    
    print("\n" + "=" * 50)
    print("🎯 API Endpoint Testing Complete!")
    print("If all tests passed, your API is ready to use with the dashboard.")

if __name__ == "__main__":
    main() 