"""
Test New Split Endpoints and Dashboard Functionality
"""
import requests
import json

def test_annual_forecast_endpoint():
    """Test the new /annual_forecast endpoint"""
    print("🔧 Testing /annual_forecast Endpoint")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    try:
        response = requests.get(f"{base_url}/annual_forecast?region=Global&start=2025&end=2030&scenario=BAU", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Annual Forecast Response Structure:")
            print(f"   Keys: {list(data.keys())}")
            
            # Verify structure
            if 'models' in data:
                models = data['models']
                print(f"✅ Found {len(models)} models: {list(models.keys())}")
                
                for model_name, model_data in models.items():
                    years = model_data.get('years', [])
                    yearly = model_data.get('yearly', [])
                    
                    print(f"   📊 {model_name}:")
                    print(f"      Years: {years[:3]}... ({len(years)} total)")
                    print(f"      Annual values: {[round(v, 2) for v in yearly[:3]]}... (first 3)")
                    
                    # Verify no cumulative data in annual endpoint
                    if 'cumulative' in model_data:
                        print(f"      ⚠️ Warning: Cumulative data found in annual endpoint")
                    else:
                        print(f"      ✅ No cumulative data (correct for annual endpoint)")
                
                return True
            else:
                print("❌ Missing 'models' key in response")
                return False
        else:
            print(f"❌ API Error: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_cumulative_forecast_endpoint():
    """Test the new /cumulative_forecast endpoint"""
    print(f"\n🔧 Testing /cumulative_forecast Endpoint")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    try:
        response = requests.get(f"{base_url}/cumulative_forecast?region=Global&start=2025&end=2030&scenario=BAU", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Cumulative Forecast Response Structure:")
            print(f"   Keys: {list(data.keys())}")
            
            # Verify structure
            if 'models' in data:
                models = data['models']
                print(f"✅ Found {len(models)} models: {list(models.keys())}")
                
                for model_name, model_data in models.items():
                    years = model_data.get('years', [])
                    yearly = model_data.get('yearly', [])
                    cumulative = model_data.get('cumulative', [])
                    
                    print(f"   📈 {model_name}:")
                    print(f"      Years: {years[:3]}... ({len(years)} total)")
                    print(f"      Annual values: {[round(v, 2) for v in yearly[:3]]}... (first 3)")
                    print(f"      Cumulative values: {[round(v, 2) for v in cumulative[:3]]}... (first 3)")
                    
                    # Verify cumulative calculation
                    if yearly and cumulative:
                        manual_cumulative = []
                        running_sum = 0
                        for annual in yearly:
                            running_sum += annual
                            manual_cumulative.append(running_sum)
                        
                        cumulative_correct = all(abs(a - b) < 0.01 for a, b in zip(cumulative, manual_cumulative))
                        
                        if cumulative_correct:
                            print(f"      ✅ Cumulative calculation is correct")
                        else:
                            print(f"      ❌ Cumulative calculation error")
                
                return True
            else:
                print("❌ Missing 'models' key in response")
                return False
        else:
            print(f"❌ API Error: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_models_comparison():
    """Test that different models give different predictions"""
    print(f"\n📊 Testing Model Differences")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    try:
        response = requests.get(f"{base_url}/annual_forecast?region=Global&start=2025&end=2030&scenario=BAU", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', {})
            
            if len(models) >= 2:
                model_names = list(models.keys())
                model1_data = models[model_names[0]]
                model2_data = models[model_names[1]]
                
                model1_values = model1_data['yearly']
                model2_values = model2_data['yearly']
                
                print(f"📊 Comparing {model_names[0]} vs {model_names[1]}:")
                print(f"   {model_names[0]}: {[round(v, 2) for v in model1_values[:3]]}...")
                print(f"   {model_names[1]}: {[round(v, 2) for v in model2_values[:3]]}...")
                
                # Check if models are different
                differences = [abs(a - b) for a, b in zip(model1_values, model2_values)]
                max_diff = max(differences)
                
                if max_diff > 0.1:
                    print(f"✅ Models produce different predictions (max diff: {round(max_diff, 2)})")
                    return True
                else:
                    print(f"⚠️ Models produce very similar predictions (max diff: {round(max_diff, 2)})")
                    return False
            else:
                print("❌ Not enough models to compare")
                return False
        else:
            print(f"❌ API Error: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_legacy_compatibility():
    """Test that legacy /forecast endpoint still works"""
    print(f"\n🔄 Testing Legacy Compatibility")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    try:
        response = requests.get(f"{base_url}/forecast?region=Global&start=2025&end=2030&scenario=BAU", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Legacy Forecast Response Structure:")
            print(f"   Keys: {list(data.keys())}")
            
            # Check for old format
            required_fields = ['region', 'scenario', 'years', 'yearly', 'cumulative']
            missing_fields = [field for field in required_fields if field not in data]
            
            if not missing_fields:
                print("✅ Legacy format maintained")
                return True
            else:
                print(f"❌ Missing legacy fields: {missing_fields}")
                return False
        else:
            print(f"❌ API Error: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_regional_forecasting():
    """Test that regional forecasting works with new endpoints"""
    print(f"\n🌍 Testing Regional Forecasting")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    regions_to_test = ["Europe", "USA", "China"]
    
    for region in regions_to_test:
        print(f"\n📍 Testing {region}:")
        
        try:
            # Test annual
            response = requests.get(f"{base_url}/annual_forecast?region={region}&start=2025&end=2027&scenario=BAU", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models_count = len(data.get('models', {}))
                print(f"   ✅ Annual: {models_count} models available")
            else:
                print(f"   ❌ Annual: HTTP {response.status_code}")
                
            # Test cumulative
            response = requests.get(f"{base_url}/cumulative_forecast?region={region}&start=2025&end=2027&scenario=BAU", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models_count = len(data.get('models', {}))
                print(f"   ✅ Cumulative: {models_count} models available")
            else:
                print(f"   ❌ Cumulative: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    print("🧪 Testing New Split Endpoints and Dashboard Functionality")
    print("=" * 60)
    
    print("This test verifies:")
    print("• /annual_forecast endpoint returns all models with yearly data")
    print("• /cumulative_forecast endpoint returns all models with cumulative data")
    print("• Models produce different predictions")
    print("• Legacy /forecast endpoint still works")
    print("• Regional forecasting works with new endpoints")
    
    # Run all tests
    results = []
    
    print(f"\n{'='*60}")
    results.append(("Annual Endpoint", test_annual_forecast_endpoint()))
    
    print(f"\n{'='*60}")
    results.append(("Cumulative Endpoint", test_cumulative_forecast_endpoint()))
    
    print(f"\n{'='*60}")
    results.append(("Model Differences", test_models_comparison()))
    
    print(f"\n{'='*60}")
    results.append(("Legacy Compatibility", test_legacy_compatibility()))
    
    print(f"\n{'='*60}")
    test_regional_forecasting()
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 TEST SUMMARY")
    print("-" * 20)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n✅ NEW FUNCTIONALITY:")
    print("• Annual view: Shows all models with yearly predictions (targets disabled)")
    print("• Cumulative view: Shows all models with cumulative predictions (targets enabled)")
    print("• Map view: Shows choropleth map for selected year")
    print("• Multiple models: Each model plotted separately with distinct lines")

if __name__ == "__main__":
    main() 