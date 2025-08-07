"""
Comprehensive test to verify dashboard seamlessly functions with API and pretrained models
"""
import requests
import time
import subprocess
import sys
import json
from pathlib import Path

API_URL = "http://localhost:8000"
DASHBOARD_URL = "http://localhost:8050"

def test_api_connection():
    """Test if API is running and responsive"""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def test_dashboard_connection():
    """Test if dashboard is running and responsive"""
    try:
        response = requests.get(DASHBOARD_URL, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def test_api_functionality():
    """Test key API endpoints that dashboard uses"""
    tests = []
    
    # Test regions endpoint
    try:
        response = requests.get(f"{API_URL}/available_regions", timeout=5)
        regions = response.json()
        tests.append(("Regions endpoint", len(regions) == 14, f"Got {len(regions)} regions"))
    except Exception as e:
        tests.append(("Regions endpoint", False, str(e)))
    
    # Test scenarios endpoint
    try:
        response = requests.get(f"{API_URL}/scenarios", timeout=5)
        scenarios = response.json()
        tests.append(("Scenarios endpoint", len(scenarios) == 3, f"Got {len(scenarios)} scenarios"))
    except Exception as e:
        tests.append(("Scenarios endpoint", False, str(e)))
    
    # Test forecast endpoint (global)
    try:
        response = requests.get(f"{API_URL}/forecast?region=Global&start=2025&end=2030&scenario=BAU", timeout=10)
        forecast = response.json()
        has_data = "years" in forecast and len(forecast["years"]) == 6
        tests.append(("Global forecast", has_data, f"Got forecast with {len(forecast.get('years', []))} years"))
    except Exception as e:
        tests.append(("Global forecast", False, str(e)))
    
    # Test forecast endpoint (regional)
    try:
        response = requests.get(f"{API_URL}/forecast?region=Europe&start=2025&end=2027&scenario=Optimistic", timeout=10)
        forecast = response.json()
        has_data = "years" in forecast and len(forecast["years"]) == 3
        tests.append(("Regional forecast", has_data, f"Got forecast with {len(forecast.get('years', []))} years"))
    except Exception as e:
        tests.append(("Regional forecast", False, str(e)))
    
    # Test optimization endpoint
    try:
        opt_data = {
            "region": "Global",
            "start": 2025,
            "end": 2030,
            "target": 430.0,
            "scenario": "BAU",
            "curve_dynamic": "early",
            "return_ci": True
        }
        response = requests.post(f"{API_URL}/optimize", json=opt_data, timeout=10)
        optimize = response.json()
        has_ci = "ci_low" in optimize and "ci_high" in optimize
        tests.append(("Optimization with CI", has_ci, f"Got optimization result with CI: {has_ci}"))
    except Exception as e:
        tests.append(("Optimization with CI", False, str(e)))
    
    # Test map endpoint
    try:
        response = requests.get(f"{API_URL}/map?year=2025&scenario=BAU&strategic=false", timeout=10)
        map_data = response.json()
        has_all_regions = len(map_data) == 14
        tests.append(("Map data", has_all_regions, f"Got map data for {len(map_data)} regions"))
    except Exception as e:
        tests.append(("Map data", False, str(e)))
    
    return tests

def test_model_files():
    """Test that model files exist and are properly sized"""
    tests = []
    
    # Check global models
    global_dir = Path("pretrained-models/global")
    global_models = list(global_dir.glob("*.pt"))
    tests.append(("Global models exist", len(global_models) == 7, f"Found {len(global_models)} global models"))
    
    # Check regional models
    regional_dir = Path("pretrained-models/regional")
    regional_models = list(regional_dir.glob("*.pt"))
    tests.append(("Regional models exist", len(regional_models) > 50, f"Found {len(regional_models)} regional models"))
    
    # Check specific key models
    key_models = [
        "pretrained-models/global/transformer.pt",
        "pretrained-models/global/autoformer.pt",
        "pretrained-models/regional/Europe_transformer.pt",
        "pretrained-models/regional/USA_autoformer.pt"
    ]
    
    for model_path in key_models:
        path = Path(model_path)
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        tests.append((f"Model {path.name}", exists and size > 0, f"Size: {size} bytes"))
    
    return tests

def simulate_dashboard_scenarios():
    """Simulate typical dashboard usage scenarios"""
    tests = []
    
    # Scenario 1: Global forecast for different scenarios
    scenarios = ["BAU", "Optimistic", "Pessimistic"]
    for scenario in scenarios:
        try:
            response = requests.get(f"{API_URL}/forecast?region=Global&start=2025&end=2030&scenario={scenario}", timeout=10)
            forecast = response.json()
            success = response.status_code == 200 and "years" in forecast
            tests.append((f"Global {scenario} scenario", success, f"Status: {response.status_code}"))
        except Exception as e:
            tests.append((f"Global {scenario} scenario", False, str(e)))
    
    # Scenario 2: Regional forecasts for key regions
    key_regions = ["Europe", "USA", "China", "India", "Australia"]
    for region in key_regions:
        try:
            response = requests.get(f"{API_URL}/forecast?region={region}&start=2025&end=2027&scenario=BAU", timeout=10)
            forecast = response.json()
            success = response.status_code == 200 and "years" in forecast
            tests.append((f"Regional {region} forecast", success, f"Status: {response.status_code}"))
        except Exception as e:
            tests.append((f"Regional {region} forecast", False, str(e)))
    
    # Scenario 3: Strategic optimization for different targets
    targets = [155, 430, 523]  # IPCC, IEA, IRENA targets
    for target in targets:
        try:
            opt_data = {
                "region": "Global",
                "start": 2025,
                "end": 2030,
                "target": float(target),
                "scenario": "BAU",
                "curve_dynamic": "early",
                "return_ci": False
            }
            response = requests.post(f"{API_URL}/optimize", json=opt_data, timeout=10)
            optimize = response.json()
            success = response.status_code == 200 and "cumulative" in optimize
            final_cumulative = optimize.get("cumulative", [])[-1] if optimize.get("cumulative") else 0
            tests.append((f"Optimize for {target} Mt", success, f"Final: {final_cumulative:.1f} Mt"))
        except Exception as e:
            tests.append((f"Optimize for {target} Mt", False, str(e)))
    
    return tests

def main():
    print("🔍 Comprehensive Dashboard Integration Test")
    print("=" * 60)
    
    # Check API connection
    print("\n📡 API Connection Test")
    api_running = test_api_connection()
    print(f"{'✅' if api_running else '❌'} API Server: {'Running' if api_running else 'Not running'}")
    
    if not api_running:
        print("❌ API server is not running. Please start it with: uvicorn api:app --host 0.0.0.0 --port 8000")
        return
    
    # Check dashboard connection
    print("\n🖥️  Dashboard Connection Test")
    dashboard_running = test_dashboard_connection()
    print(f"{'✅' if dashboard_running else '❌'} Dashboard: {'Running' if dashboard_running else 'Not running'}")
    
    # Test API functionality
    print("\n🔧 API Functionality Tests")
    api_tests = test_api_functionality()
    for test_name, success, details in api_tests:
        print(f"{'✅' if success else '❌'} {test_name}: {details}")
    
    # Test model files
    print("\n📁 Model File Tests")
    model_tests = test_model_files()
    for test_name, success, details in model_tests:
        print(f"{'✅' if success else '❌'} {test_name}: {details}")
    
    # Simulate dashboard scenarios
    print("\n🎭 Dashboard Usage Scenarios")
    scenario_tests = simulate_dashboard_scenarios()
    for test_name, success, details in scenario_tests:
        print(f"{'✅' if success else '❌'} {test_name}: {details}")
    
    # Summary
    all_tests = api_tests + model_tests + scenario_tests
    passed = sum(1 for _, success, _ in all_tests if success)
    total = len(all_tests)
    
    print("\n" + "=" * 60)
    print(f"🎯 TEST SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your dashboard is ready for seamless operation!")
        if dashboard_running:
            print(f"🌐 Dashboard available at: {DASHBOARD_URL}")
        else:
            print(f"💡 Start dashboard with: python dashboard.py")
    else:
        print(f"⚠️  {total-passed} tests failed. Please review the issues above.")
    
    print("\n📋 Dashboard Features Tested:")
    print("   ✅ Global and Regional forecasting")
    print("   ✅ Multiple scenario support (BAU, Optimistic, Pessimistic)")
    print("   ✅ Strategic optimization with targets")
    print("   ✅ Map data for choropleth visualization")
    print("   ✅ Confidence intervals")
    print("   ✅ All 14 regions supported")
    print("   ✅ All 7 model types available")

if __name__ == "__main__":
    main() 